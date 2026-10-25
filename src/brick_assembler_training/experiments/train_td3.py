"""Training script that currently trains a model against the Pendulum environment."""

from __future__ import annotations

import copy
import random

import gymnasium as gym
import numpy as np
import torch
import wandb
from pydantic import BaseModel, Field
from safetensors.torch import save_model
from tqdm import tqdm

from ..algorithms.replay_buffer import ReplayBuffer
from ..algorithms.td3 import train_td3
from ..models.pendulum import PendulumActorNet, PendulumValueNet
from ..utils import create_directory, parse_args


class Args(BaseModel):
    num_envs: int = Field(32, description="Number of parallel Pendulum environments")
    train_steps: int = Field(4, description="Vector environment steps per iteration")
    iterations: int = Field(100_000, description="Collection/training iterations")
    train_iters: int = Field(1, description="Gradient updates per iteration")
    train_batch_size: int = Field(256, description="Training batch size")
    grad_steps: int = Field(
        1,
        description="Number of training iterations to perform before taking a gradient step. train_iters must be > 1.",
    )
    discount: float = Field(0.99, ge=0, le=1, description="Discount factor")
    q_lr: float = Field(0.001, description="Learning rate of the value network")
    p_lr: float = Field(0.001, description="Learning rate of the critic network")
    exploration_noise: float = Field(
        0.1, ge=0, description="Action exploration noise stddev"
    )
    noise_scale: float = Field(
        0.2, ge=0, description="Target policy smoothing noise stddev"
    )
    noise_clip: float = Field(0.5, ge=0, description="Target smoothing noise limit")
    polyak: float = Field(0.005, le=1, description="Source weight in target updates")
    update_policy_every: int = Field(
        2, description="Critic updates per policy/target update"
    )
    warmup_steps: int = Field(
        500,
        ge=0,
        description="Number of iterations at the start where random actions are performed + no learning occurs",
    )
    buffer_size: int = Field(100_000, description="Size of the replay buffer")
    eval_iters: int = Field(8, description="Eval episodes")
    max_eval_steps: int = Field(200, description="Maximum steps per eval episode")
    eval_every: int = Field(100, description="Iterations before eval")
    save_every: int = Field(100, description="Iterations before saving")
    seed: int = 0
    runs_dir: str = Field("./runs", description="Where all the runs are stored")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def evaluate(
    actor: PendulumActorNet,
    env: gym.Env,
    device: torch.device,
    episodes: int,
    max_steps: int,
) -> float:
    actor.eval()
    reward_total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action = actor(torch.as_tensor(obs, dtype=torch.float32, device=device))
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            reward_total += float(reward)
            if terminated or truncated:
                break
    return reward_total / episodes


def main() -> None:
    # Parse args
    args = parse_args(Args)
    device = torch.device(args.device)
    random.seed(args.seed)

    # Set up experiment
    metadata = args
    wandb.init(
        project="brick_assembler",
        config=metadata.model_dump(),
    )
    chkpt_path = create_directory(args.runs_dir, metadata)

    # Set up envs
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("Pendulum-v1") for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    test_env = gym.make("Pendulum-v1")

    obs_space = env.single_observation_space
    act_space = env.single_action_space
    assert isinstance(obs_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Box)
    act_low = torch.as_tensor(act_space.low, dtype=torch.float32, device=device)
    act_high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)

    # Set up models
    q_net_1 = PendulumValueNet(obs_space.shape[0], act_space.shape[0]).to(device)
    q_net_2 = PendulumValueNet(obs_space.shape[0], act_space.shape[0]).to(device)
    p_net = PendulumActorNet(obs_space.shape[0], act_low, act_high).to(device)
    q_net_1_target = copy.deepcopy(q_net_1).requires_grad_(False)
    q_net_2_target = copy.deepcopy(q_net_2).requires_grad_(False)
    p_net_target = copy.deepcopy(p_net).requires_grad_(False)
    q_1_opt = torch.optim.Adam(q_net_1.parameters(), lr=args.q_lr)
    q_2_opt = torch.optim.Adam(q_net_2.parameters(), lr=args.q_lr)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=args.p_lr)

    # Set up buffer
    buffer = ReplayBuffer(
        obs_space.shape,
        (0,),
        args.buffer_size,
        torch.float32,
        action_shape=act_space.shape,
    )
    obs, _ = env.reset()
    test_env.reset()
    p_net.eval()

    for iteration in tqdm(range(args.iterations)):
        # Collect samples
        with torch.inference_mode():
            for _ in range(args.train_steps):
                if iteration < args.warmup_steps:
                    actions = env.action_space.sample()
                else:
                    actions_tensor = p_net(
                        torch.as_tensor(obs, dtype=torch.float32, device=device)
                    )
                    actions_tensor += (
                        torch.randn_like(actions_tensor) * args.exploration_noise
                    )
                    actions = actions_tensor.clamp(act_low, act_high).cpu().numpy()
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                replay_next_obs = next_obs.copy()
                for index in np.flatnonzero(terminated | truncated):
                    replay_next_obs[index] = infos["final_obs"][index]
                buffer.insert_step(
                    torch.from_numpy(obs),
                    torch.from_numpy(replay_next_obs),
                    torch.from_numpy(actions),
                    list(rewards),
                    list(terminated),
                    None,
                    None,
                )
                obs = next_obs

        if iteration < args.warmup_steps or len(buffer) < args.train_batch_size:
            continue

        # Train models
        q_loss, p_loss = train_td3(
            q_net_1=q_net_1,
            q_net_1_target=q_net_1_target,
            q_1_opt=q_1_opt,
            q_net_2=q_net_2,
            q_net_2_target=q_net_2_target,
            q_2_opt=q_2_opt,
            p_net=p_net,
            p_net_target=p_net_target,
            p_opt=p_opt,
            buffer=buffer,
            device=device,
            train_iters=args.train_iters,
            train_batch_size=args.train_batch_size,
            discount=args.discount,
            polyak=args.polyak,
            noise_clip=args.noise_clip,
            noise_scale=args.noise_scale,
            act_low=act_low,
            act_high=act_high,
            update_policy_every=args.update_policy_every,
        )

        # Log metrics
        log_dict = {
            "avg_q_loss": q_loss / (2 * args.train_iters),
            "avg_p_loss": p_loss / (args.train_iters // args.update_policy_every),
            "q_lr": q_1_opt.param_groups[0]["lr"],
            "p_lr": p_opt.param_groups[0]["lr"],
        }
        is_finished = iteration + 1 == args.iterations
        if (iteration + 1) % args.eval_every == 0 or is_finished:
            log_dict["avg_eval_episode_reward"] = evaluate(
                p_net, test_env, device, args.eval_iters, args.max_eval_steps
            )
        wandb.log(log_dict)

        # Save models
        if (iteration + 1) % args.save_every == 0 or is_finished:
            for label in [str(iteration), "latest"]:
                save_model(q_net_1, str(chkpt_path / f"{label}.safetensors"))
                save_model(q_net_2, str(chkpt_path / f"{label}.safetensors"))
                save_model(p_net, str(chkpt_path / f"{label}.safetensors"))


if __name__ == "__main__":
    main()

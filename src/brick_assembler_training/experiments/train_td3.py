"""
TODO: This currently assumes a DQN training setup, needs to be updated to use TD3.
"""

from __future__ import annotations

import copy
import random
from functools import reduce
from typing import Any

import numpy as np  # type: ignore
import torch
import wandb
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from pydantic import BaseModel
from torch import nn
from tqdm import tqdm

from brick_assembler_training.utils import parse_args

from ..algorithms.replay_buffer import ReplayBuffer
from ..algorithms.td3 import train_td3


class Args(BaseModel):
    num_envs: int = (
        32  # Number of environments to step through at once during sampling.
    )
    train_steps: int = 4  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
    iterations: int = 100000  # Number of sample/train iterations.
    train_iters = 1  # Number of passes over the samples collected.
    train_batch_size = 512  # Minibatch size while training models.
    discount = 0.999  # Discount factor applied to rewards.
    q_epsilon = (
        0.1  # Epsilon for epsilon greedy strategy. This gets annealed over time.
    )
    eval_steps = 8  # Number of eval runs to average over.
    max_eval_steps = 300  # Max number of steps to take during each eval run.
    q_lr = 0.0001  # Learning rate of the q net.
    warmup_steps = 500  # For the first n number of steps, we will only sample randomly.
    buffer_size = 10000  # Number of elements that can be stored in the buffer.
    target_update = 500  # Number of iterations before updating Q target.
    device = torch.device("cuda")


def main() -> None:
    args = parse_args(Args)
    metadata = args
    wandb.init(
        project="tests",
        config=metadata.model_dump(),
    )

    env = envpool.make("CartPole-v1", "gym", num_envs=num_envs)
    test_env = CartPoleEnv()

    # Initialize Q network
    obs_space = env.observation_space
    act_space = env.action_space
    q_net = QNet(obs_space.shape, int(act_space.n))
    q_net_target = copy.deepcopy(q_net)
    q_net_target.to(device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)

    # Initialize buffer
    buffer = ReplayBuffer(
        torch.Size(obs_space.shape),
        torch.Size((int(act_space.n),)),
        buffer_size,
    )

    # Train loop
    obs = torch.Tensor(env.reset()[0])
    done = False
    for step in tqdm(range(iterations), position=0):
        percent_done = step / iterations

        # Collect experience
        with torch.no_grad():
            for _ in range(train_steps):
                if (
                    random.random() < q_epsilon * max(1.0 - percent_done, 0.05)
                    or step < warmup_steps
                ):
                    actions_list = [
                        random.randrange(0, int(act_space.n)) for _ in range(num_envs)
                    ]
                    actions_ = np.array(actions_list)
                else:
                    q_vals = q_net(obs)
                    actions_ = q_vals.argmax(1).numpy()
                obs_, rewards, dones, truncs, _ = env.step(actions_)
                next_obs = torch.from_numpy(obs_)
                buffer.insert_step(
                    obs,
                    next_obs,
                    torch.from_numpy(actions_).squeeze(0),
                    rewards,
                    dones,
                    None,
                    None,
                )
                obs = next_obs

        # Train
        if buffer.filled:
            total_q_loss = train_dqn(
                q_net,
                q_net_target,
                q_opt,
                buffer,
                device,
                train_iters,
                train_batch_size,
                discount,
            )

            # Evaluate the network's performance after this training iteration.
            eval_done = False
            with torch.no_grad():
                reward_total = 0
                pred_reward_total = 0
                obs_, info = test_env.reset()
                eval_obs = torch.from_numpy(np.array(obs_)).float()
                for _ in range(eval_steps):
                    steps_taken = 0
                    score = 0
                    for _ in range(max_eval_steps):
                        q_vals = q_net(eval_obs.unsqueeze(0)).squeeze()
                        action = q_vals.argmax(0).item()
                        pred_reward_total += (
                            q_net(eval_obs.unsqueeze(0)).squeeze().max(0).values.item()
                        )
                        obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                        eval_obs = torch.from_numpy(np.array(obs_)).float()
                        steps_taken += 1
                        reward_total += reward
                        if eval_done or eval_trunc:
                            obs_, info = test_env.reset()
                            eval_obs = torch.from_numpy(np.array(obs_)).float()
                            break

            wandb.log(
                {
                    "avg_eval_episode_reward": reward_total / eval_steps,
                    "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
                    "avg_q_loss": total_q_loss / train_iters,
                    "q_lr": q_opt.param_groups[-1]["lr"],
                }
            )

            # Update Q target
            if (step + 1) % target_update == 0:
                q_net_target.load_state_dict(q_net.state_dict())

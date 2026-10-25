from __future__ import annotations

import torch
from shape_extensions import Int, IntTuple, IntVar
from torch import Tensor


class ReplayBuffer[Cap: IntVar, StateShape: IntTuple, ActionMasksShape: IntTuple]:
    """
    Stores transitions and generates mini batches.
    """

    def __init__(
        self,
        state_shape: StateShape,
        action_masks_shape: ActionMasksShape,
        capacity: Int[Cap],
        action_type: type[float | int],
    ) -> None:
        k = torch.float
        batch_state_shape = (capacity, *state_shape)
        batch_action_masks_shape = (capacity, *action_masks_shape)
        self.capacity = capacity
        self.next = 0
        d = torch.device("cpu")
        self.states: Tensor[[Cap, *StateShape]] = torch.zeros(
            batch_state_shape, dtype=k, device=d, requires_grad=False
        )
        self.next_states: Tensor[[Cap, *StateShape]] = torch.zeros(
            batch_state_shape, dtype=k, device=d, requires_grad=False
        )
        self.actions = torch.zeros(
            (capacity,), dtype=action_type, device=d, requires_grad=False
        )
        self.rewards = torch.zeros((capacity,), dtype=k, device=d, requires_grad=False)
        # Technically this is the "terminated" flag
        self.dones = torch.zeros((capacity,), dtype=k, device=d, requires_grad=False)
        self.filled = False
        self.masks: Tensor[[Cap, *ActionMasksShape]] = torch.zeros(
            batch_action_masks_shape, dtype=torch.int, device=d, requires_grad=False
        )
        self.next_masks: Tensor[[Cap, *ActionMasksShape]] = torch.zeros(
            batch_action_masks_shape, dtype=torch.int, device=d, requires_grad=False
        )

    def insert_step[B: IntVar](
        self,
        states: Tensor[[B, *StateShape]],
        next_states: Tensor[[B, *StateShape]],
        actions: Tensor[[B]],
        rewards: list[float],
        dones: list[bool],
        masks: Tensor | None,
        next_masks: Tensor | None,
    ):
        """
        Inserts a transition from each environment into the buffer. Make sure
        more data than steps aren't inserted.
        """
        batch_size = len(dones)
        d = torch.device("cpu")
        with torch.no_grad():
            indices = torch.arange(
                self.next,
                (self.next + batch_size),
            ).remainder(self.capacity)
            self.states.index_copy_(0, indices, states)
            self.next_states.index_copy_(0, indices, next_states)
            self.actions.index_copy_(0, indices, actions)
            self.rewards.index_copy_(
                0, indices, torch.tensor(rewards, dtype=torch.float, device=d)
            )
            self.dones.index_copy_(
                0, indices, torch.tensor(dones, dtype=torch.float, device=d)
            )
            if masks is not None:
                self.masks.index_copy_(0, indices, masks.to(torch.int))
            if next_masks is not None:
                self.next_masks.index_copy_(0, indices, next_masks.to(torch.int))
        self.next = (self.next + batch_size) % self.capacity
        if self.next == 0:
            self.filled = True

    def sample[B: IntVar](
        self, batch_size: Int[B]
    ) -> tuple[
        Tensor[[B, *StateShape]],  # States
        Tensor[[B, *StateShape]],  # Next States
        Tensor[[B]],  # Actions
        Tensor[[B]],  # Rewards
        Tensor[[B]],  # Dones
        Tensor[[B, *ActionMasksShape]],  # Action Masks
        Tensor[[B, *ActionMasksShape]],  # Next Action Masks
    ]:
        """
        Generates a minibatch of experience.
        """
        with torch.no_grad():
            indices = torch.randint(
                self.capacity,
                (batch_size,),
                dtype=torch.int,
            )
            rand_states = self.states.index_select(0, indices)
            rand_next_states = self.next_states.index_select(0, indices)
            rand_actions = self.actions.index_select(0, indices)
            rand_rewards = self.rewards.index_select(0, indices)
            rand_dones = self.dones.index_select(0, indices)
            rand_masks = self.masks.index_select(0, indices)
            rand_next_masks = self.next_masks.index_select(0, indices)
            return (
                rand_states,
                rand_next_states,
                rand_actions,
                rand_rewards,
                rand_dones,
                rand_masks,
                rand_next_masks,
            )

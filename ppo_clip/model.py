import torch
import numpy as np
from collections import defaultdict
# import loss
from torch.nn import MSELoss
from utils import BOARD_LEN
from utils import reward_fn

class PPOClipAgent:
    def __init__(self, action_space_shape):
        self.action_space_shape = action_space_shape
        self.game_state_shape = (BOARD_LEN, BOARD_LEN, 27)

        self._actor = Actor(self.game_state_shape, self.action_space_shape)
        self._critic = Critic(self.game_state_shape)

        self._num_trajectories = 100
        self._epochs = 10
        self._trajectories = defaultdict(list)

    def run(self, id, game_state, valid_actions):
        for i in range(self._epochs * self._num_trajectories):
            
            # Collect trajectory
            action = self.get_action(game_state, valid_actions)
            self._trajectories[id].append((game_state, action, reward_fn(game_state)))

            yield action
            
            if i % self._num_trajectories == 0:
                self._update(game_state, valid_actions)

    def rewards_to_go(self, rewards, dones, gamma=0.99):
        """
        Compute rewards-to-go for each timestep.
        
        Args:
            rewards (torch.Tensor): Tensor of rewards for each timestep
            values (torch.Tensor): Tensor of value estimates for each timestep
            dones (torch.Tensor): Tensor of done flags for each timestep
            gamma (float): Discount factor (default: 0.99)
        
        Returns:
            torch.Tensor: Rewards-to-go for each timestep
        """
        rewards_to_go = torch.zeros_like(rewards)
        running_sum = 0
        
        # Iterate backwards through the rewards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_sum = 0
            running_sum = rewards[t] + gamma * running_sum
            rewards_to_go[t] = running_sum
            
        return rewards_to_go

    def advantage(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        """
        Calculate advantage using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards (torch.Tensor): Tensor of rewards for each timestep
            values (torch.Tensor): Tensor of value estimates for each timestep
            dones (torch.Tensor): Tensor of done flags for each timestep
            gamma (float): Discount factor (default: 0.99)
            lambda_ (float): GAE parameter for variance-bias tradeoff (default: 0.95)
        
        Returns:
            torch.Tensor: Advantages for each timestep
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = values[-1]
        
        # Iterate backwards through the rewards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                last_advantage = 0
                last_value = values[t]
                
            # Calculate TD error
            delta = rewards[t] + gamma * last_value * (1 - dones[t]) - values[t]
            
            # Calculate advantage using GAE
            advantage = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
            
            advantages[t] = advantage
            last_advantage = advantage
            last_value = values[t]
            
        return advantages

    def _update(self, game_state, valid_actions):
        # Rewards-to-go
        rewards = torch.tensor([t[2] for t in self._trajectories])
        values = self._critic.forward(game_state)
        dones = torch.tensor([t[2] for t in self._trajectories])

        rewards_to_go = self.rewards_to_go(rewards, values, dones)

        # Advantage
        advantages = self.advantage(rewards, values, dones)

        # Actor: SGD with Adam optimizer
        # Maximize PPO-clip objective
        optimizer = torch.optim.Adam(self._actor.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss = self.ppo_clip_objective(rewards_to_go, advantages)
        loss.backward()
        optimizer.step()

        # Critic: SGD with Adam optimizer
        # Minimize MSE loss
        optimizer = torch.optim.Adam(self._critic.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss = MSELoss(rewards_to_go, values)
        loss.backward()
        optimizer.step()

    def get_action(self, game_state, valid_actions):
        action_space_logits = self._actor.forward(game_state)

        print(valid_actions)
        mask = create_multidimensional_mask(torch.tensor(valid_actions), self.action_space_shape)
        print("Mask shape:", mask.shape)

        # use the mask to filter the valid actions by
        valid_action_space = action_space_logits * mask
        # count nonzero elements
        print("Number of nonzero elements:", torch.count_nonzero(valid_action_space), "Number of 1s in the mask:", torch.sum(mask))

        # softmax and choose the action
        valid_action_space = torch.softmax(valid_action_space, dim=len(self.action_space_shape) - 1)

        # Flatten tensor and get argmax
        flat_index = torch.argmax(valid_action_space.flatten())
        
        # Convert flat index back to multi-dimensional indices
        action = np.unravel_index(flat_index.item(), self.action_space_shape)
        # this is a np arr of np.int64, i want to convert it to a list of ints
        action = [int(i) for i in action]

        print("Action: ", action)
        return action

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        pass

    def forward(self, x):
        pass

class Critic(torch.nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        pass

    def forward(self, x):
        pass

def create_multidimensional_mask(coordinates, shape):
    """
    Create a multi-dimensional mask with zeros and set specified coordinates to 1.

    Parameters:
    - coordinates: List of coordinate lists/tuples, where each inner list/tuple
                   represents a coordinate across all dimensions
    - shape: Tuple specifying the dimensions of the mask

    Returns:
    - NumPy array mask with 1s at specified coordinates
    """
    # Create a zero matrix with the specified shape
    mask = torch.zeros(shape, dtype=torch.int)

    # Set the specified coordinates to 1
    for coord in coordinates:
        # Ensure the coordinate is within the matrix bounds
        if len(coord) == len(shape) and all(0 <= c < s for c, s in zip(coord, shape)):
            mask[tuple(coord)] = 1

    return mask

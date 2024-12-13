import torch
import numpy as np
from collections import defaultdict
# import loss
from torch.nn import MSELoss
from utils import BOARD_LEN
from utils import reward_fn
from torch import optim

class PPOClipAgent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self._actor = Actor(self.input_size, self.output_size)
        self._critic = Critic(self.input_size)

        lr = 0.0001

        self.actor_optimizer = optim.Adam(self._actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self._critic.parameters(), lr=lr)

        self._num_trajectories = 100
        self._epochs = 10
        self._trajectories = []

    def to(self, device):
        """
        Move the model to the specified device.
        """
        self._actor = self._actor.to(device)
        self._critic = self._critic.to(device)
        self.device = device
        return self
        
    def run(self, id, game_state, valid_actions):
        old_log_probs = torch.randn(self.output_size)

        for i in range(self._epochs * self._num_trajectories):
            
            # Collect trajectory
            action = self.get_action(game_state, valid_actions)
            self._trajectories.append((game_state, action, reward_fn(game_state)))

            yield action
            
            if i % self._num_trajectories == 0:
                new_log_probs = self._update(game_state, valid_actions, old_log_probs)
                old_log_probs = new_log_probs

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
        self.actor_optimizer.zero_grad()

        new_action_space_logits = self._actor.forward(game_state)

        mask = create_multidimensional_mask(torch.tensor(valid_actions), self.output_size)
        masked_action_space_logits = new_action_space_logits * mask

        # regular softmax
        masked_action_space_probs = torch.softmax(masked_action_space_logits, dim=len(self.output_size) - 1)

        dist = Categorical(masked_action_space_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()    

        # TODO: old_log_probs is not defined
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic: SGD with Adam optimizer
        # Minimize MSE loss
        self.critic_optimizer.zero_grad()
        loss = MSELoss(rewards_to_go, values)
        loss.backward()
        self.critic_optimizer.step()

        return new_log_probs

    def get_action(self, game_state, valid_actions):
        action_space_logits = self._actor.forward(game_state)

        print(valid_actions)
        mask = create_multidimensional_mask(torch.tensor(valid_actions), self.output_size)
        print("Mask shape:", mask.shape)

        # use the mask to filter the valid actions by
        masked_action_space_logits = action_space_logits * mask
        # count nonzero elements
        print("Number of nonzero elements:", torch.count_nonzero(masked_action_space_logits), "Number of 1s in the mask:", torch.sum(mask))

        # softmax and choose the action
        masked_action_space_probs = torch.softmax(masked_action_space_logits, dim=len(self.output_size) - 1)

        # Flatten tensor and get argmax
        flat_index = torch.argmax(masked_action_space_probs.flatten())
        
        # Convert flat index back to multi-dimensional indices
        action = np.unravel_index(flat_index.item(), self.output_size)
        # this is a np arr of np.int64, i want to convert it to a list of ints
        action = [int(i) for i in action]

        print("Action: ", action)
        return action

class Actor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        # Return a tensor of shape output_size
        return torch.randn(self.output_size)

class Critic(torch.nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        return torch.randn(1)

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

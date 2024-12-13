import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
# import loss
from torch.nn import MSELoss
from utils import BOARD_LEN
from utils import reward_fn
from torch import optim
import math
from vectorize_game_state import game_state_to_vector
from torch.distributions import Categorical

# To reduce duplicate code, this is used for both the actor and the critic
# NOTE: we are not batching the input for now
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # CNN for spatial features
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=27, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # MLP for global features
        self.global_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
    def forward(self, spatial, global_features):
        print("=== Feature Extractor Forward ===")
        # Rearrange spatial for CNN
        spatial = spatial.permute(2, 0, 1)  # -> (27, board_size, board_size)
        
        # Process spatial
        spatial_out = self.conv_net(spatial.unsqueeze(0))  # Add a dummy batch dimension for conv layers
        spatial_out = spatial_out.flatten()  # -> (64*board_size*board_size)

        # Process global
        global_out = self.global_net(global_features)  # -> (16)
        
        # Combine
        combined = torch.cat([spatial_out, global_out], dim=0)  # Concatenate along the feature dimension
        return combined

class Actor(nn.Module):
    def __init__(self, board_size, output_size):
        super(Actor, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.output_size = output_size
        
        spatial_flat_size = 64 * board_size * board_size
        combined_size = spatial_flat_size + 16

        # Flatten output size since it's like (3, 22, 11, 11, 11, 11, 24)
        flat_output_size = math.prod(output_size)
        
        self.policy_head = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, flat_output_size)
        )

    def forward(self, spatial, global_features):
        print("=== Actor Forward ===")
        combined = self.feature_extractor(spatial, global_features)
        output = self.policy_head(combined)
        # Reshape the output to the desired dimensions
        return output.view(self.output_size)

class Critic(nn.Module):
    def __init__(self, board_size):
        super(Critic, self).__init__()
        self.feature_extractor = FeatureExtractor()
        
        spatial_flat_size = 64 * board_size * board_size
        combined_size = spatial_flat_size + 16
        
        self.value_head = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single value
        )

    def forward(self, spatial, global_features):
        print("=== Critic Forward ===")
        combined = self.feature_extractor(spatial, global_features)
        return self.value_head(combined)  # -> single value

class PPOClipAgent:
    def __init__(self, input_size, output_size):
        print("Initializing PPOClipAgent", output_size)
        self.input_size = input_size
        self.output_size = output_size

        self._actor = Actor(BOARD_LEN, self.output_size)
        self._critic = Critic(BOARD_LEN)

        lr = 0.0001

        self.actor_optimizer = optim.Adam(self._actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self._critic.parameters(), lr=lr)

        self._batch_size = 4
        self._epochs = 10
        self._trajectories = []
        self._counter = 0

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
        self._counter += 1

        # Collect trajectory
        action = self.get_action(game_state, valid_actions)
        self._trajectories.append((game_state, action, reward_fn(game_state)))
        
        if self._counter % self._batch_size == 0:
            new_log_probs = self._update(game_state, valid_actions, old_log_probs)
            old_log_probs = new_log_probs

        return action

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
            
        # Compute the returns as the sum of the advantages and the values since A = Q - V (to be used as target for the critic)
        returns = advantages + values
        return returns, advantages

    def _update(self, game_state, valid_actions, old_log_probs):
        print("=== Update ===")
        # Rewards-to-go
        rewards = torch.tensor([t[2] for t in self._trajectories])

        # Extract spatial and global features
        spatial_tensor, global_info = game_state_to_vector(game_state)

        values = self._critic.forward(spatial_tensor, global_info)
        dones = torch.tensor([t[2] for t in self._trajectories])

        # rewards_to_go = self.rewards_to_go(rewards, values, dones)

        # Advantage
        rewards_to_go, advantages = self.advantage(rewards, values, dones)

        print("\nrewards:", rewards)
        print("\nvalues:", values) 
        print("\ndones:", dones)
        print("\nrewards_to_go:", rewards_to_go)
        print("\nadvantages:", advantages)

        # Extract actions from trajectories
        actions = torch.tensor([t[1] for t in self._trajectories])

        print("\n=== Actor Update ===")
        print("1. Zeroing actor gradients...")
        self.actor_optimizer.zero_grad()

        print("2. Getting new action logits from actor network...")
        new_action_space_logits = self._actor.forward(spatial_tensor, global_info)

        print("3. Creating and applying action mask...")
        mask = create_multidimensional_mask(torch.tensor(valid_actions), self.output_size)
        masked_action_space_logits = new_action_space_logits * mask

        print("4. Computing action probabilities...")
        # regular softmax
        masked_action_space_probs = torch.softmax(masked_action_space_logits, dim=len(self.output_size) - 1)

        print("5. Creating probability distribution...")
        dist = Categorical(masked_action_space_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()    

        print("6. Computing PPO ratios and losses...")
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        print("7. Performing actor backprop...")
        actor_loss.backward()
        self.actor_optimizer.step()

        print("\n=== Critic Update ===") 
        print("1. Zeroing critic gradients...")
        self.critic_optimizer.zero_grad()
        
        print("2. Computing critic loss...")
        loss = MSELoss(rewards_to_go, values)
        
        print("3. Performing critic backprop...")
        loss.backward()
        self.critic_optimizer.step()

        return new_log_probs

    def get_action(self, game_state, valid_actions):
        print("=== Getting Action ===")
        spatial_tensor, global_info = game_state_to_vector(game_state)
        action_space_logits = self._actor.forward(spatial_tensor, global_info)

        # print(valid_actions)
        mask = create_multidimensional_mask(torch.tensor(valid_actions), self.output_size)
        # print("Mask shape:", mask.shape)

        # use the mask to filter the valid actions by
        masked_action_space_logits = action_space_logits * mask
        # count nonzero elements
        # print("Number of nonzero elements:", torch.count_nonzero(masked_action_space_logits), "Number of 1s in the mask:", torch.sum(mask))

        # softmax and choose the action
        masked_action_space_probs = torch.softmax(masked_action_space_logits, dim=len(self.output_size) - 1)

        # Flatten tensor and get argmax
        flat_index = torch.argmax(masked_action_space_probs.flatten())
        
        # Convert flat index back to multi-dimensional indices
        action = np.unravel_index(flat_index.item(), self.output_size)
        # this is a np arr of np.int64, i want to convert it to a list of ints
        action = [int(i) for i in action]

        return action

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

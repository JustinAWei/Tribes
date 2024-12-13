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
        spatial = spatial.permute(0, 3, 1, 2)  # -> (batch_size, 27, board_size, board_size)
        
        # Process spatial
        spatial_out = self.conv_net(spatial)
        spatial_out = spatial_out.flatten(start_dim=1)  # Flatten from the second dimension onwards

        # Process global
        global_out = self.global_net(global_features)  # -> (batch_size, 16)
        
        # Combine
        combined = torch.cat([spatial_out, global_out], dim=1)  # Concatenate along the feature dimension
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
        return output.view(-1, *self.output_size)  # Ensure batch dimension is preserved

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
        return self.value_head(combined)  # Output should be (batch_size, 1)

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

        self._batch_size = 1
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
        print("=== Advantage ===")
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
        dones = torch.tensor([t[2] for t in self._trajectories])

        # Extract spatial and global features
        game_states = [t[0] for t in self._trajectories]
        spatial_tensors, global_infos = game_state_to_vector(game_states) # -> (batch_size, board_size, board_size, 27), (batch_size, 2)
        values = self._critic.forward(spatial_tensors, global_infos)

        print("\nrewards:", rewards)
        print("\nvalues:", values) 
        print("\ndones:", dones)

        # Advantage
        rewards_to_go, advantages = self.advantage(rewards, values, dones)

        print("\nrewards_to_go:", rewards_to_go)
        print("\nadvantages:", advantages)

        # Extract actions from trajectories
        actions = torch.tensor([t[1] for t in self._trajectories])

        print("\n=== Actor Update ===")
        print("1. Zeroing actor gradients...")
        self.actor_optimizer.zero_grad()

        print("2. Getting new action logits from actor network...")
        new_action_space_logits = self._actor.forward(spatial_tensors, global_infos)

        print("3. Creating and applying action mask...")
        # Create mask tensor of -inf everywhere
        mask = torch.full(self.output_size, float('-inf'))

        # Expand mask to match the batch dimension of logits
        mask = mask.unsqueeze(0)

        # Debug: Print the shape of logits and mask
        print("Logits shape:", new_action_space_logits.shape)
        print("Mask shape:", mask.shape)

        # Set valid action coordinates to 0 in mask
        # TODO: this is not batched right now, only assuming single game
        print("valid_actions:", valid_actions)
        for action in valid_actions:
            mask[0, action[0], action[1], action[2], action[3], action[4], action[5], action[6]] = 0.0  # Apply mask across the batch dimension

        # Add mask to logits to keep valid actions and mask out invalid ones
        masked_logits = new_action_space_logits + mask

        print("4. Computing action probabilities...")

        # Flatten logits for softmax
        # print dimensions of masked_action_space_logits
        masked_action_space_probs = torch.softmax(masked_logits.flatten(), dim=0)
        print("masked_action_space_probs:", masked_action_space_probs)
        
        print("5. Creating probability distribution...")
        dist = Categorical(masked_action_space_probs)
        new_log_probs = dist.log_prob(actions.view(-1))
        entropy = dist.entropy()  

        print("6. Computing PPO ratios and losses...") 
        # Flatten old_log_probs to match new_log_probs
        old_log_probs = old_log_probs.view(-1)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surrogate1 = ratio * advantages.view(-1)
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.view(-1)
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
        # Get state features
        [spatial_tensor, global_info] = game_state_to_vector([game_state])
        
        # Get action logits from actor network
        logits = self._actor.forward(spatial_tensor, global_info)
        
        # Create mask tensor of -inf everywhere
        mask = torch.full(self.output_size, float('-inf'))
        
        # Set valid action coordinates to 0 in mask
        for action in valid_actions:
            mask[tuple(action)] = 0.0
            
        # Add mask to logits to keep valid actions and mask out invalid ones
        masked_logits = logits + mask

        # Get action probabilities and select action
        probs = torch.softmax(masked_logits.flatten(), dim=0)
        dist = Categorical(probs)
        flat_index = dist.sample()
        
        # Convert to multi-dimensional action
        action = [int(i) for i in np.unravel_index(flat_index.item(), self.output_size)]
        
        return action
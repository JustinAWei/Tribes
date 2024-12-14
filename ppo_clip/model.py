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
from utils import timing_decorator

# To reduce duplicate code, this is used for both the actor and the critic
# NOTE: we are not batching the input for now
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # CNN for spatial features
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=27, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # MLP for global features
        self.global_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
    @timing_decorator
    def forward(self, spatial, global_features):
        # print("=== Feature Extractor Forward ===")
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
        
        spatial_flat_size = 32 * board_size * board_size
        combined_size = spatial_flat_size + 8

        # Flatten output size since it's like (3, 22, 11, 11, 11, 11, 24)
        flat_output_size = math.prod(output_size)
        
        self.policy_head = nn.Sequential(
            nn.Linear(combined_size, 16),
            nn.ReLU(),
            nn.Linear(16, flat_output_size)
        )

    @timing_decorator
    def forward(self, spatial, global_features):
        # print("=== Actor Forward ===")
        combined = self.feature_extractor(spatial, global_features)
        output = self.policy_head(combined)
        # Reshape the output to the desired dimensions
        return output.view(-1, *self.output_size)  # Ensure batch dimension is preserved

class Critic(nn.Module):
    def __init__(self, board_size):
        super(Critic, self).__init__()
        self.feature_extractor = FeatureExtractor()
        
        spatial_flat_size = 32 * board_size * board_size
        combined_size = spatial_flat_size + 8
        
        self.value_head = nn.Sequential(
            nn.Linear(combined_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output a single value
        )

    @timing_decorator
    def forward(self, spatial, global_features):
        # print("=== Critic Forward ===")
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

        self.epsilon = 0.2
        self._batch_size = 200
        self._epochs = 2
#         self._base_trajectories = {
#             "spatial_tensor": torch.empty((0, BOARD_LEN, BOARD_LEN, 27), dtype=torch.float),
#             "global_info": torch.empty((0, 2), dtype=torch.float),
# 
#             "actions": torch.empty((0, 6), dtype=torch.long),  # [batch_size, action_dims] from actions shape: torch.Size([B, num_dimensions])
#             "rewards": torch.empty((0, 1), dtype=torch.float),   # [batch_size] from rewards shape: torch.Size([1])
#             "probs": torch.empty((0, np.prod(self.output_size)), dtype=torch.float),  # [batch_size, flattened_action_space] from probs shape shown in get_action()
#             "masks": torch.empty((0, *self.output_size), dtype=torch.float)  # [batch_size, *output_size] from mask shape: torch.Size([1, *output_size])
#         }
        self._base_trajectories = {
            "spatial_tensor": [],
            "global_info": [],
            "actions": [],
            "rewards": [],
            "probs": [],
            "masks": []
        }
        self._trajectories = self._base_trajectories.copy()
        self._counter = 0

    def to(self, device):
        """
        Move the model to the specified device.
        """
        self._actor = self._actor.to(device)
        self._critic = self._critic.to(device)
        self.device = device
        return self
        
    @timing_decorator
    @profile
    def run(self, id, game_state, valid_actions):
        self._counter += 1

        # game_states = [batch_size, 1]
        # valid action = [batch_size, N, 6]

        # Get state features
        [spatial_tensor, global_info] = game_state_to_vector([game_state]) # -> (batch_size, board_size, board_size, 27), (batch_size, 2)

        # print("spatial_tensor shape:", spatial_tensor.shape)
        # print("global_info shape:", global_info.shape)

        batch_size = spatial_tensor.shape[0]

        # Create mask tensor of -inf everywhere
        masks = torch.full((batch_size, *self.output_size), float('-inf')) # -> (batch_size, action_space_size)
        # print("mask shape:", masks.shape)
        # Convert valid actions to tensor indices and set to 0 in mask
        # Convert valid actions to indices in flattened action space
        for batch_idx, actions in enumerate([valid_actions]):
            # Each action is [action_type, x1, y1, x2, y2, extra]
            for action in actions:
                # print("action:", action)
                # Set corresponding position in mask to 0 to allow this action
                masks[batch_idx][action[0]][action[1]][action[2]][action[3]][action[4]] = 0

        # compute rewards
        rewards = torch.tensor([[reward_fn(game_state) for game_state in [game_state]]])
        # print("rewards shape:", rewards.shape)

        # Collect trajectory
        actions, probs = self.get_action(spatial_tensor, global_info, masks)
        # print("ALL SHAPES")
        # print("actions shape:", actions.shape)
        # print("probs shape:", probs.shape)
        # print("masks shape:", masks.shape)
        # print("rewards shape:", rewards.shape)

        # self._trajectories = {
        #     "spatial_tensor": torch.cat((self._trajectories["spatial_tensor"], spatial_tensor), dim=0),
        #     "global_info": torch.cat((self._trajectories["global_info"], global_info), dim=0),
        #     "actions": torch.cat((self._trajectories["actions"], actions), dim=0),
        #     "rewards": torch.cat((self._trajectories["rewards"], rewards), dim=0),
        #     "probs": torch.cat((self._trajectories["probs"], probs), dim=0),
        #     "masks": torch.cat((self._trajectories["masks"], masks), dim=0) 
        # }
        self._trajectories["spatial_tensor"].append(spatial_tensor)
        self._trajectories["global_info"].append(global_info)
        self._trajectories["actions"].append(actions)
        self._trajectories["rewards"].append(rewards)
        self._trajectories["probs"].append(probs)
        self._trajectories["masks"].append(masks)


        # print("\n=== Trajectory Shapes ===")
        # print("spatial_tensor shape:", self._trajectories["spatial_tensor"].shape)
        # print("global_info shape:", self._trajectories["global_info"].shape)
        # print("actions shape:", self._trajectories["actions"].shape) 
        # print("rewards shape:", self._trajectories["rewards"].shape)
        # print("probs shape:", self._trajectories["probs"].shape)
        # print("mask shape:", self._trajectories["masks"].shape)

        if self._counter % 100 == 0:
            print("=== Trajectory Shapes ===")
            print("spatial_tensor shape:", len(self._trajectories["spatial_tensor"]), self._trajectories["spatial_tensor"][0].shape)
            print("global_info shape:", len(self._trajectories["global_info"]), self._trajectories["global_info"][0].shape)
            print("actions shape:", len(self._trajectories["actions"]), self._trajectories["actions"][0].shape) 
            print("rewards shape:", len(self._trajectories["rewards"]), self._trajectories["rewards"][0].shape)
            print("probs shape:", len(self._trajectories["probs"]), self._trajectories["probs"][0].shape)
            print("mask shape:", len(self._trajectories["masks"]), self._trajectories["masks"][0].shape)
        
        if self._counter % self._batch_size == 0:
            self._update()
            self._trajectories = self._base_trajectories.copy()

        return actions[0].tolist()

    @timing_decorator
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
        # print("=== Advantage ===")
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

    @timing_decorator
    @profile
    def _update(self):
        # print("=== Update ===")

        probs = torch.cat(self._trajectories["probs"], dim=0)
        actions = torch.cat(self._trajectories["actions"], dim=0)

        # print("actions shape:", actions.shape)
        # print("probs shape:", probs.shape)

        dist = Categorical(probs)

        # print("dist:", dist)

        # translate actions to flat indices using ravel_multi_index
        flat_actions = np.ravel_multi_index(tuple(actions.T), self.output_size)
        flat_actions = torch.tensor(flat_actions)
        old_log_probs = dist.log_prob(flat_actions)
        # print("old_log_probs shape:", old_log_probs.shape)

        for i in range(self._epochs):
            # print("=== Epoch", i, "===")
            # Get rewards and dones from trajectories
            rewards = torch.cat(self._trajectories["rewards"], dim=0)
            dones = (rewards != 0).float()
            spatial_tensor = torch.cat(self._trajectories["spatial_tensor"], dim=0)
            global_info = torch.cat(self._trajectories["global_info"], dim=0)
            masks = torch.cat(self._trajectories["masks"], dim=0)

            values = self._critic.forward(spatial_tensor, global_info)

            # print("\nrewards:", rewards)
            # print("\nvalues:", values) 
            # print("\ndones:", dones)

            # Advantage
            rewards_to_go, advantages = self.advantage(rewards, values.detach(), dones.detach())

            # print("\nrewards_to_go:", rewards_to_go)
            # print("\nadvantages:", advantages)


            # print("\n=== Actor Update ===")
            self.actor_optimizer.zero_grad()

            # print("4. Computing action probabilities...")

            _, new_probs = self.get_action(spatial_tensor, global_info, masks)

            # Flatten logits for softmax
            # print("old_log_probs shape:", old_log_probs.shape)

            dist = Categorical(new_probs)
            flat_actions = np.ravel_multi_index(tuple(actions.T), self.output_size)
            flat_actions = torch.tensor(flat_actions)
            new_log_probs = dist.log_prob(flat_actions)

            # entropy = probs.entropy()  

            # Ensure old_log_probs and new_log_probs have the same shape as advantages
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # print("7. Performing actor backprop...")
            actor_loss.backward()
            self.actor_optimizer.step()

            # print("\n=== Critic Update ===") 
            # print("1. Zeroing critic gradients...")
            self.critic_optimizer.zero_grad()
            
            # print("2. Computing critic loss...")
            criterion = nn.MSELoss()
            loss = criterion(values, rewards_to_go)
            
            # print("3. Performing critic backprop...")
            loss.backward()
            self.critic_optimizer.step()

        return new_log_probs

    @timing_decorator
    @profile
    def get_action(self, spatial_tensor, global_info, masks):
        try:
            # Get action logits from actor network
            logits = self._actor.forward(spatial_tensor, global_info) # -> (batch_size, action_space_size)
            # print("logits shape:", logits.shape)
            
            # Add mask to logits to keep valid actions and mask out invalid ones
            masked_logits = logits.add(masks)  # Using in-place batch addition
            # print("masked_logits shape:", masked_logits.shape)

            # Get action probabilities and select action
            probs = torch.softmax(masked_logits.flatten(start_dim=1), dim=1)  # Keep batch dimension

            # print("probs shape:", probs.shape)
            dist = Categorical(probs)
            flat_indices = dist.sample()  # Will sample for each item in batch

            # print("flat", flat_indices)
            # print("flat_indices shape:", flat_indices.shape)

            # Convert flat indices [B, 1] to multi-dimensional indices using torch.unravel_index
            actions = torch.stack(
                torch.unravel_index(
                    flat_indices,
                    self._actor.output_size
                ),
                dim=1
            )  # Shape: [B, num_dimensions]

            # print("actions shape:", actions.shape)

            return actions, probs
        except Exception as e:
            print(f"Error in get_action: {e}")
            print(f"Logits shape: {logits.shape}")
            print(f"Masks shape: {masks.shape}")
            print(f"Flat indices: {flat_indices}")
            print(f"Actor output size: {self._actor.output_size}")

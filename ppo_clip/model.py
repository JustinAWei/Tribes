import copy
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
from utils import timing_decorator, serialize_trajectories
from modal_functions.update import remote_update
import requests
import modal
from torch.utils.tensorboard import SummaryWriter
import time

# Endpoint for the update function
url = "https://kev2010--ppo-clip-update-trigger-update.modal.run"

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

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
        self.device = DEVICE
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

    
    def forward(self, spatial, global_features):
        # Ensure inputs are on GPU
        spatial = spatial.to(self.device)
        global_features = global_features.to(self.device)

        # print("=== Actor Forward ===")
        combined = self.feature_extractor(spatial, global_features)
        output = self.policy_head(combined)
        # Reshape the output to the desired dimensions
        return output.view(-1, *self.output_size)  # Ensure batch dimension is preserved

class Critic(nn.Module):
    def __init__(self, board_size):
        super(Critic, self).__init__()
        self.device = DEVICE
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

    
    def forward(self, spatial, global_features):
        # Ensure inputs are on GPU
        spatial = spatial.to(self.device)
        global_features = global_features.to(self.device)
        
        # print("=== Critic Forward ===")
        combined = self.feature_extractor(spatial, global_features)
        return self.value_head(combined)  # Output should be (batch_size, 1)

class PPOClipAgent:
    def __init__(self, load_path, save_path, input_size, output_size):
        self.device = DEVICE
        print(f"Using device: {self.device}")
        print("Initializing PPOClipAgent", output_size)

        self._load_path = load_path
        self._save_path = save_path

        self.input_size = input_size
        self.output_size = output_size

        self._actor = Actor(BOARD_LEN, self.output_size).to(self.device)
        self._critic = Critic(BOARD_LEN).to(self.device).to(self.device)

        lr = 0.0001

        self.actor_optimizer = optim.Adam(self._actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self._critic.parameters(), lr=lr)

        self.epsilon = 0.2
        self._batch_size = 2048
        self._epochs = 3
        self.multipliers = torch.tensor([
            np.prod(self.output_size[i+1:]) 
            for i in range(len(self.output_size))
        ], device=self.device, dtype=torch.float32)

        self._base_trajectories = {
            "spatial_tensor": [],
            "global_info": [],
            "actions": [],
            "rewards": [],
            "probs": [],
            "masks": []
        }
        self._trajectories = copy.deepcopy(self._base_trajectories)
        self._counter = 0

        # Tracking information
        self.writer = SummaryWriter(f'ppo_clip/runs/ppo_clip_{time.strftime("%Y%m%d-%H%M%S")}')
        self.update_count = 0
        
    def save_weights(self):
        """Save actor and critic model weights to files
        
        Args:
            path (str, optional): Path to save weights to. If None, uses default path.
        """
        torch.save(self._actor.state_dict(), f"{self._save_path}/{self._counter}/actor.pth")
        torch.save(self._critic.state_dict(), f"{self._save_path}/{self._counter}/critic.pth") 
        print(f"Saved model weights to {self._save_path}")

    def load_weights(self):
        """Load actor and critic model weights from files"""
        try:
            self._actor.load_state_dict(torch.load(f"{self._load_path}/actor.pth", map_location=self.device))
            self._critic.load_state_dict(torch.load(f"{self._load_path}/critic.pth", map_location=self.device))
            print(f"Loaded model weights from {self._load_path}")
        except FileNotFoundError as e:
            print(f"Could not load weights from {self._load_path}: {e}")
    
    def game_ended(self, game_state):
        # who won
        # set the last reward to 1 if win, -1 if lose, 0 if not done
        # last reward is the reward for the last end turn
        # the very last index should be the reward for the last action and should be set to 1

        print("=== Game Ended ===")

        # Handle each game state in the batch
        for game_idx, gs in enumerate(game_state):
            rank_1_tribe_id = gs['ranking'][0]["id"]
            tribes = gs['board']['tribes']
            for tribe in tribes:
                active_tribe_id = tribe['actorId']
                # find the last action for the tribe in global_info
                for i in range(len(self._trajectories["global_info"])-1, -1, -1):
                    global_info = self._trajectories["global_info"][i]
                    if global_info[game_idx, 0] == active_tribe_id:
                        # This is ranked "WIN/LOSE" first.
                        winner = rank_1_tribe_id == active_tribe_id
                        self._trajectories["rewards"][i] = torch.tensor([[1 if winner else -1]], device=self.device)
                        print("Setting reward for tribe", active_tribe_id, "at index", i, "to", self._trajectories["rewards"][i])
                        break

        # the last turn for the tribe that didn't win should be set to 0
    
    # @profile
    def run(self, id, game_state, valid_actions):
        self._counter += 1

        # game_states = [batch_size, 1]
        # valid action = [batch_size, N, 6]

        # Get state features
        [spatial_tensor, global_info] = game_state_to_vector([game_state], self.device) # -> (batch_size, board_size, board_size, 27), (batch_size, 2)
        spatial_tensor = spatial_tensor.to(self.device)
        global_info = global_info.to(self.device)

        # print("spatial_tensor shape:", spatial_tensor.shape)
        # print("global_info shape:", global_info.shape)

        batch_size = spatial_tensor.shape[0]

        # Create mask tensor of -inf everywhere
        masks = torch.full((batch_size, *self.output_size), float('-inf')).to(self.device) # -> (batch_size, action_space_size)
        # print("mask shape:", masks.shape)
        # Convert valid actions to tensor indices and set to 0 in mask
        # Convert valid actions to indices in flattened action space
        for batch_idx, actions in enumerate([valid_actions]):
            # Each action is [action_type, x1, y1, x2, y2, extra]
            for action in actions:
                # print("action:", action)
                # Set corresponding position in mask to 0 to allow this action
                masks[batch_idx][action[0]][action[1]][action[2]] = 0

        # compute rewards
        rewards = torch.tensor([[reward_fn(game_state) for game_state in [game_state]]]).to(self.device)
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

        if self._counter % 512 == 0:
            print(self._counter)
        #     print("=== Trajectory Shapes ===")
        #     print("spatial_tensor shape:", len(self._trajectories["spatial_tensor"]), self._trajectories["spatial_tensor"][0].shape)
        #     print("global_info shape:", len(self._trajectories["global_info"]), self._trajectories["global_info"][0].shape)
        #     print("actions shape:", len(self._trajectories["actions"]), self._trajectories["actions"][0].shape) 
        #     print("rewards shape:", len(self._trajectories["rewards"]), self._trajectories["rewards"][0].shape)
        #     print("probs shape:", len(self._trajectories["probs"]), self._trajectories["probs"][0].shape)
        #     print("mask shape:", len(self._trajectories["masks"]), self._trajectories["masks"][0].shape)
        
        if self._counter % self._batch_size == 0:
            # Sort trajectories by tribe ID

            # Convert list of tensors to single tensor for easier manipulation
            global_info_tensor = torch.cat(self._trajectories["global_info"], dim=0)
            
            # Get tribe IDs from first column
            tribe_ids = global_info_tensor[:, 0]
            # print("tribe_ids:", tribe_ids)

            # Get indices that would sort by tribe ID while preserving order
            sorted_indices = torch.argsort(tribe_ids, stable=True)
            # print("sorted_indices:", sorted_indices)
            
            # Apply sorting to all trajectory components
            self._trajectories["spatial_tensor"] = [self._trajectories["spatial_tensor"][i] for i in sorted_indices]
            self._trajectories["global_info"] = [self._trajectories["global_info"][i] for i in sorted_indices]
            self._trajectories["actions"] = [self._trajectories["actions"][i] for i in sorted_indices]
            self._trajectories["rewards"] = [self._trajectories["rewards"][i] for i in sorted_indices]
            self._trajectories["probs"] = [self._trajectories["probs"][i] for i in sorted_indices]
            self._trajectories["masks"] = [self._trajectories["masks"][i] for i in sorted_indices]

            # print("sorted_indices values:", self._trajectories["global_info"])

            self._update()

            if self._counter % (5 * self._batch_size) == 0:
                self.save_weights()

            self._trajectories = copy.deepcopy(self._base_trajectories)

        return actions[0].tolist()

    
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

    
    # @profile
    def _update(self):
#         remote_update = modal.Function.lookup("ppo_clip_update", "remote_update")
# 
#         # Call the remote function asynchronously
#         return await remote_update.remote(
#             serialized_trajectories=serialize_trajectories(self._trajectories),
#             output_size=self.output_size,
#             epochs=self._epochs,
#             epsilon=self.epsilon,
#             actor_optimizer=self.actor_optimizer.state_dict(),
#             critic_optimizer=self.critic_optimizer.state_dict(),
#             critic=self._critic.state_dict(),
#             advantage=self.advantage,
#             get_action=self.get_action
#         )
    
        # print("=== Update ===")

                # Move tensors to GPU
        probs = torch.cat(self._trajectories["probs"], dim=0).to(self.device).float()
        actions = torch.cat(self._trajectories["actions"], dim=0).to(self.device).float()
        rewards = torch.cat(self._trajectories["rewards"], dim=0).to(self.device).float()
        dones = (rewards != 0).float().to(self.device)
        spatial_tensor = torch.cat(self._trajectories["spatial_tensor"], dim=0).to(self.device).float()
        global_info = torch.cat(self._trajectories["global_info"], dim=0).to(self.device).float()
        masks = torch.cat(self._trajectories["masks"], dim=0).to(self.device).float()


        # print("actions shape:", actions.shape)
        # print("probs shape:", probs.shape)

        dist = Categorical(probs)

        # print("dist:", dist)

        # # translate actions to flat indices using ravel_multi_index
        # flat_actions = np.ravel_multi_index(tuple(actions.T), self.output_size)
        # flat_actions = torch.tensor(flat_actions, device=self.device)
        # Replace np.ravel_multi_index with PyTorch operations
        multipliers = torch.tensor([
            np.prod(self.output_size[i+1:]) 
            for i in range(len(self.output_size))
        ], device=self.device, dtype=torch.float32)

        flat_actions = (actions.float() * multipliers).sum(dim=1)
        old_log_probs = dist.log_prob(flat_actions)
        # print("old_log_probs shape:", old_log_probs.shape)

        for _ in range(self._epochs):
            # print("=== Epoch", i, "===")

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
            # Replace np.ravel_multi_index with PyTorch operations
            multipliers = torch.tensor([
                np.prod(self.output_size[i+1:]) 
                for i in range(len(self.output_size))
            ], device=self.device, dtype=torch.float32)

            flat_actions = (actions * multipliers).sum(dim=1)
            # flat_actions = np.ravel_multi_index(tuple(actions.T), self.output_size)
            # flat_actions = torch.tensor(flat_actions).to(self.device)
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

            self._log_training_metrics(
                actor_loss=actor_loss,
                critic_loss=loss,
                ratio=ratio,
                advantages=advantages,
                values=values,
                rewards_to_go=rewards_to_go
            )

        return new_log_probs

    
    # @profile
    def get_action(self, spatial_tensor, global_info, masks):
        try:
            # Ensure inputs are on GPU
            spatial_tensor = spatial_tensor.to(self.device).float()
            global_info = global_info.to(self.device).float()
            masks = masks.to(self.device).float()

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

            # # Convert flat indices [B, 1] to multi-dimensional indices using torch.unravel_index
            # actions = torch.stack(
            #     torch.unravel_index(
            #         flat_indices,
            #         self._actor.output_size
            #     ),
            #     dim=1
            # )  # Shape: [B, num_dimensions]
            # Convert flat indices to multi-dimensional indices using pure PyTorch
            actions = (flat_indices.unsqueeze(1) // self.multipliers) % torch.tensor(
                self.output_size, device=self.device
            )

            return actions.long(), probs
        
        except Exception as e:
            print(f"Error in get_action: {e}")
    
    def _log_training_metrics(self, actor_loss, critic_loss, ratio, advantages, values, rewards_to_go):
        """
        Log training metrics to TensorBoard.
        
        Args:
            actor_loss (torch.Tensor): The actor's loss value
            critic_loss (torch.Tensor): The critic's loss value
            ratio (torch.Tensor): The policy ratio
            advantages (torch.Tensor): The calculated advantages
            values (torch.Tensor): The predicted values
            rewards_to_go (torch.Tensor): The calculated returns
        """
        # Log losses
        self.writer.add_scalar('Loss/actor', actor_loss.item(), self.update_count)
        self.writer.add_scalar('Loss/critic', critic_loss.item(), self.update_count)
        
        # Log gradients
        for name, param in self._actor.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/actor_{name}', param.grad, self.update_count)
        
        for name, param in self._critic.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/critic_{name}', param.grad, self.update_count)

        # Log other useful metrics
        self.writer.add_scalar('Policy/ratio', ratio.mean().item(), self.update_count)
        self.writer.add_scalar('Policy/advantages', advantages.mean().item(), self.update_count)
        self.writer.add_scalar('Values/predicted', values.mean().item(), self.update_count)
        self.writer.add_scalar('Values/returns', rewards_to_go.mean().item(), self.update_count)

        self.update_count += 1

import copy
import os
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
import requests
import modal
from torch.utils.tensorboard import SummaryWriter
import time
import json

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# To reduce duplicate code, this is used for both the actor and the critic
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
    def __init__(self, 
                 load_path, 
                 save_path, 
                 input_size, 
                 output_size, 
                 lr=0.0001, 
                 clip_ratio=0.2, 
                 batch_size=2048, 
                 n_epochs=3, 
                 gamma=0.99, 
                 gae_lambda=0.95):
        self.device = DEVICE
        print(f"Using device: {self.device}")
        print("Initializing PPOClipAgent", output_size)

        self._load_path = load_path
        self._save_path = save_path

        self.input_size = input_size
        self.output_size = output_size

        self._actor = Actor(BOARD_LEN, self.output_size).to(self.device)
        self._critic = Critic(BOARD_LEN).to(self.device).to(self.device)

        self._learning_rate = lr

        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._learning_rate)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=self._learning_rate)

        self._clip_ratio = clip_ratio
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._gae_lambda = gae_lambda

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
        """Save actor and critic model weights, optimizer states, and hyperparameters to files"""
        print(f"Saving model data to {self._save_path}")
        save_dir = f"{self._save_path}/{self._counter}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self._actor.state_dict(), f"{save_dir}/actor.pth")
        torch.save(self._critic.state_dict(), f"{save_dir}/critic.pth")
        
        # Save optimizer states
        torch.save(self._actor_optimizer.state_dict(), f"{save_dir}/actor_optimizer.pth")
        torch.save(self._critic_optimizer.state_dict(), f"{save_dir}/critic_optimizer.pth")
        
        # Save hyperparameters
        hyperparams = {
            'learning_rate': self._learning_rate,
            'clip_ratio': self._clip_ratio,
            'batch_size': self._batch_size,
            'n_epochs': self._n_epochs,
            # Add any other hyperparameters you want to save
        }
        
        with open(f"{save_dir}/hyperparams.json", 'w') as f:
            json.dump(hyperparams, f, indent=4)
        
        print(f"Saved model data to {save_dir}")

    def load_weights(self):
        """Load actor and critic model weights, optimizer states, and hyperparameters from files"""
        try:
            print(f"Loading model data from {self._load_path}")
            
            # Load model weights
            self._actor.load_state_dict(torch.load(f"{self._load_path}/actor.pth", map_location=self.device))
            self._critic.load_state_dict(torch.load(f"{self._load_path}/critic.pth", map_location=self.device))
            
            # Load optimizer states
            self._actor_optimizer.load_state_dict(torch.load(f"{self._load_path}/actor_optimizer.pth", map_location=self.device))
            self._critic_optimizer.load_state_dict(torch.load(f"{self._load_path}/critic_optimizer.pth", map_location=self.device))
            
            # Load hyperparameters
            with open(f"{self._load_path}/hyperparams.json", 'r') as f:
                hyperparams = json.load(f)
                
            # Update instance variables with loaded hyperparameters
            self._learning_rate = hyperparams['learning_rate']
            self._clip_ratio = hyperparams['clip_ratio']
            self._batch_size = hyperparams['batch_size']
            self._n_epochs = hyperparams['n_epochs']
            # Update any other hyperparameters you saved
            
            print(f"Loaded model data from {self._load_path}")
            
        except FileNotFoundError as e:
            print(f"Could not load data from {self._load_path}: {e}")
        except json.JSONDecodeError as e:
            print(f"Could not parse hyperparameters file: {e}")
    
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

        batch_size = spatial_tensor.shape[0]

        # Create mask tensor of -inf everywhere
        masks = torch.full((batch_size, *self.output_size), float('-inf')).to(self.device) # -> (batch_size, action_space_size)
        for batch_idx, actions in enumerate([valid_actions]):
            # Each action is [action_type, x1, y1, x2, y2, extra]
            for action in actions:
                # Set corresponding position in mask to 0 to allow this action
                masks[batch_idx][action[0]][action[1]][action[2]] = 0

        # compute rewards
        rewards = torch.tensor([[reward_fn(game_state) for game_state in [game_state]]]).to(self.device)

        # Collect trajectory
        actions, probs = self.get_action(spatial_tensor, global_info, masks)

        self._trajectories["spatial_tensor"].append(spatial_tensor)
        self._trajectories["global_info"].append(global_info)
        self._trajectories["actions"].append(actions)
        self._trajectories["rewards"].append(rewards)
        self._trajectories["probs"].append(probs)
        self._trajectories["masks"].append(masks)

        # if self._counter % 512 == 0:
        #     print("=== Trajectory Shapes ===")
        #     print("spatial_tensor shape:", len(self._trajectories["spatial_tensor"]), self._trajectories["spatial_tensor"][0].shape)
        #     print("global_info shape:", len(self._trajectories["global_info"]), self._trajectories["global_info"][0].shape)
        #     print("actions shape:", len(self._trajectories["actions"]), self._trajectories["actions"][0].shape) 
        #     print("rewards shape:", len(self._trajectories["rewards"]), self._trajectories["rewards"][0].shape)
        #     print("probs shape:", len(self._trajectories["probs"]), self._trajectories["probs"][0].shape)
        #     print("mask shape:", len(self._trajectories["masks"]), self._trajectories["masks"][0].shape)
        
        if self._counter % self._batch_size == 0:
            # Convert list of tensors to single tensor for easier manipulation
            global_info_tensor = torch.cat(self._trajectories["global_info"], dim=0)
            
            # Get tribe IDs from first column
            tribe_ids = global_info_tensor[:, 0]

            # Get indices that would sort by tribe ID while preserving order
            sorted_indices = torch.argsort(tribe_ids, stable=True)
            
            # Apply sorting to all trajectory components
            self._trajectories["spatial_tensor"] = [self._trajectories["spatial_tensor"][i] for i in sorted_indices]
            self._trajectories["global_info"] = [self._trajectories["global_info"][i] for i in sorted_indices]
            self._trajectories["actions"] = [self._trajectories["actions"][i] for i in sorted_indices]
            self._trajectories["rewards"] = [self._trajectories["rewards"][i] for i in sorted_indices]
            self._trajectories["probs"] = [self._trajectories["probs"][i] for i in sorted_indices]
            self._trajectories["masks"] = [self._trajectories["masks"][i] for i in sorted_indices]

            self._update()

            if self._counter % (5 * self._batch_size) == 0:
                self.save_weights()

            self._trajectories = copy.deepcopy(self._base_trajectories)

        return actions[0].tolist()

    
    def advantage(self, rewards, values, dones):
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
            delta = rewards[t] + self._gamma * last_value * (1 - dones[t]) - values[t]
            
            # Calculate advantage using GAE
            advantage = delta + self._gamma * self._gae_lambda * (1 - dones[t]) * last_advantage
            
            advantages[t] = advantage
            last_advantage = advantage
            last_value = values[t]
            
        # Compute the returns as the sum of the advantages and the values since A = Q - V (to be used as target for the critic)
        returns = advantages + values
        return returns, advantages

    
    # @profile
    def _update(self):
        # Move tensors to GPU
        probs = torch.cat(self._trajectories["probs"], dim=0).to(self.device).float()
        actions = torch.cat(self._trajectories["actions"], dim=0).to(self.device).float()
        rewards = torch.cat(self._trajectories["rewards"], dim=0).to(self.device).float()
        dones = (rewards != 0).float().to(self.device)
        spatial_tensor = torch.cat(self._trajectories["spatial_tensor"], dim=0).to(self.device).float()
        global_info = torch.cat(self._trajectories["global_info"], dim=0).to(self.device).float()
        masks = torch.cat(self._trajectories["masks"], dim=0).to(self.device).float()


        dist = Categorical(probs)
        flat_actions = (actions.float() * self.multipliers).sum(dim=1)
        old_log_probs = dist.log_prob(flat_actions)

        for _ in range(self._n_epochs):
            # print("=== Epoch", i, "===")
            values = self._critic.forward(spatial_tensor, global_info)

            # Advantage
            rewards_to_go, advantages = self.advantage(rewards, values.detach(), dones.detach())

            self._actor_optimizer.zero_grad()
            _, new_probs = self.get_action(spatial_tensor, global_info, masks)
            dist = Categorical(new_probs)
            flat_actions = (actions * self.multipliers).sum(dim=1)
            new_log_probs = dist.log_prob(flat_actions)

            # entropy = probs.entropy()  

            # Actor loss
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            actor_loss.backward()
            self._actor_optimizer.step()

            # Critic loss
            self._critic_optimizer.zero_grad()
            criterion = nn.MSELoss()
            critic_loss = criterion(values, rewards_to_go)
            critic_loss.backward()
            self._critic_optimizer.step()

            self._log_training_metrics(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
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
            
            # Add mask to logits to keep valid actions and mask out invalid ones
            masked_logits = logits.add(masks)  # Using in-place batch addition

            # Get action probabilities and select action
            probs = torch.softmax(masked_logits.flatten(start_dim=1), dim=1)  # Keep batch dimension

            dist = Categorical(probs)
            flat_indices = dist.sample()  # Will sample for each item in batch

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

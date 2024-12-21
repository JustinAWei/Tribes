"""
TODO: SPLIT THIS FILE UP SINCE IT'S TOO BIG
"""
import copy
import os
import torch
import torch.nn as nn
import numpy as np
from utils import BOARD_LEN
from utils import reward_fn
from torch import optim
import math
from vectorize_game_state import game_state_to_vector
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time
import json

# DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
DEVICE = 'cpu' # for trajectory collection
TRAINING_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu' # for training

# Network architecture constants
INITIAL_CHANNELS = 27        # Input channels for spatial features
CONV_CHANNELS = 64          # Number of channels in conv layers
GLOBAL_INPUT_SIZE = 2       # Size of global features input
GLOBAL_HIDDEN_SIZE = 64
GLOBAL_OUTPUT_SIZE = 32

POLICY_HIDDEN_SIZES = [128, 64]
VALUE_HIDDEN_SIZES = [128, 64]

# Calculate spatial dimensions after convolutions
def calculate_conv_output_size(size, kernel_size=3, stride=1, padding=1):
    return (size + 2*padding - kernel_size)//stride + 1

def calculate_spatial_size(board_size):
    # First two conv layers (stride=1) maintain size
    size = board_size
    # First stride-2 conv
    size = calculate_conv_output_size(size, stride=2)
    return size

BOARD_SIZE = 11  # This should match your game board size
FINAL_SPATIAL_SIZE = calculate_spatial_size(BOARD_SIZE)
SPATIAL_FLAT_SIZE = CONV_CHANNELS * FINAL_SPATIAL_SIZE * FINAL_SPATIAL_SIZE
COMBINED_FEATURE_SIZE = SPATIAL_FLAT_SIZE + GLOBAL_OUTPUT_SIZE

def calculate_total_parameters(output_size):
    # Convolutional layers
    conv1_params = (INITIAL_CHANNELS * 3 * 3 * CONV_CHANNELS) + CONV_CHANNELS  # weights + biases
    conv2_params = (CONV_CHANNELS * 3 * 3 * CONV_CHANNELS) + CONV_CHANNELS    # weights + biases

    # Global feature network
    global1_params = (GLOBAL_INPUT_SIZE * GLOBAL_HIDDEN_SIZE) + GLOBAL_HIDDEN_SIZE
    global2_params = (GLOBAL_HIDDEN_SIZE * GLOBAL_OUTPUT_SIZE) + GLOBAL_OUTPUT_SIZE

    # Policy head
    policy_head_params = 0
    input_size = CONV_CHANNELS * FINAL_SPATIAL_SIZE * FINAL_SPATIAL_SIZE + GLOBAL_OUTPUT_SIZE
    for hidden_size in POLICY_HIDDEN_SIZES:
        policy_head_params += (input_size * hidden_size) + hidden_size  # weights + biases
        input_size = hidden_size
    policy_head_params += (input_size * math.prod(output_size)) + math.prod(output_size)

    # Value head
    value_head_params = 0
    input_size = CONV_CHANNELS * FINAL_SPATIAL_SIZE * FINAL_SPATIAL_SIZE + GLOBAL_OUTPUT_SIZE
    for hidden_size in VALUE_HIDDEN_SIZES:
        value_head_params += (input_size * hidden_size) + hidden_size  # weights + biases
        input_size = hidden_size
    value_head_params += (input_size * 1) + 1  # Final layer outputs a single value

    total_params = conv1_params + conv2_params + global1_params + global2_params + policy_head_params + value_head_params

    print(f"Total number of parameters: {total_params}")
    print(f"Breakdown:")
    print(f"  Conv layers: {conv1_params + conv2_params}")
    print(f"  Global network: {global1_params + global2_params}")
    print(f"  Policy head: {policy_head_params}")
    print(f"  Value head: {value_head_params}")

# To reduce duplicate code, this is used for both the actor and the critic
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # CNN for spatial features
        self.conv_net = nn.Sequential(
            nn.Conv2d(INITIAL_CHANNELS, CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # MLP for global features processing
        self.global_net = nn.Sequential(
            nn.Linear(GLOBAL_INPUT_SIZE, GLOBAL_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(GLOBAL_HIDDEN_SIZE, GLOBAL_OUTPUT_SIZE),
            nn.ReLU()
        )
    
    def forward(self, spatial, global_features):
        spatial = spatial.permute(0, 3, 1, 2)
        spatial_out = self.conv_net(spatial)
        spatial_out = spatial_out.flatten(start_dim=1)
        global_out = self.global_net(global_features)
        return torch.cat([spatial_out, global_out], dim=1)

class Actor(nn.Module):
    def __init__(self, board_size, output_size):
        super(Actor, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.output_size = output_size
        
        # Calculate flattened output size
        self.flat_output_size = math.prod(output_size)
        
        # Deeper policy head
        layers = []
        input_size = COMBINED_FEATURE_SIZE
        for hidden_size in POLICY_HIDDEN_SIZES:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.flat_output_size))
        
        self.policy_head = nn.Sequential(*layers)
    
    def forward(self, spatial, global_features):
        combined = self.feature_extractor(spatial, global_features)
        output = self.policy_head(combined)
        return output.view(-1, *self.output_size)

class Critic(nn.Module):
    def __init__(self, board_size):
        super(Critic, self).__init__()
        self.feature_extractor = FeatureExtractor()
        
        # Build value head layers dynamically
        layers = []
        input_size = COMBINED_FEATURE_SIZE
        for hidden_size in VALUE_HIDDEN_SIZES:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        
        self.value_head = nn.Sequential(*layers)
    
    def forward(self, spatial, global_features):
        combined = self.feature_extractor(spatial, global_features)
        return self.value_head(combined)

class PPOClipAgent:
    def __init__(self, 
                 save_path, 
                 input_size, 
                 output_size, 
                 lr=0.0001, 
                 clip_ratio=0.15, 
                 batch_size=1600,
                 n_epochs=5, 
                 gamma=0.9999, 
                 gae_lambda=0.75,
                 value_coef=0.5,
                 entropy_coef=0.05,
                 checkpoint_path=""):
        self.device = DEVICE
        self.training_device = TRAINING_DEVICE
        print(f"Using device: {self.device} for inference, {self.training_device} for training")
        calculate_total_parameters(output_size)

        self.input_size = input_size
        self.output_size = output_size

        self._actor = Actor(BOARD_LEN, self.output_size).to(self.device)
        self._critic = Critic(BOARD_LEN).to(self.device)

        self._learning_rate = lr

        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._learning_rate)
        self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=self._learning_rate)

        self._clip_ratio = clip_ratio
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._entropy_coef = entropy_coef
        self._value_coef = value_coef

        self.multipliers = torch.tensor([
            np.prod(self.output_size[i+1:]) 
            for i in range(len(self.output_size))
        ], device=self.device, dtype=torch.float32)

        self._base_trajectories = {
            "game_id": [],
            "spatial_tensor": [],
            "global_info": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "masks": []
        }
        self._trajectories = copy.deepcopy(self._base_trajectories)
        self._counter = 0

        # Tracking information
        self.writer = SummaryWriter(f'ppo_clip/runs/ppo_clip_{time.strftime("%Y%m%d-%H%M%S")}')
        self.update_count = 0

        # Generate a unique directory name based on hyperparameters and timestamp
        if checkpoint_path:
            # Extract the base name of the checkpoint file without extension
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkpoint_name = os.path.basename(checkpoint_dir)
            actions_number = os.path.splitext(os.path.basename(checkpoint_path))[0].split('_')[-1]
            unique_dir_name = (
                f"{save_path}/{checkpoint_name}_actions_{actions_number}_{time.strftime('%Y%m%d-%H%M%S')}"
            )
        else:
            unique_dir_name = (
                f"{save_path}/lr_{lr}_clip_{clip_ratio}_bs_{batch_size}_epochs_{n_epochs}_"
                f"gamma_{gamma}_gae_{gae_lambda}_entropy_{entropy_coef}_value_{value_coef}_{time.strftime('%Y%m%d-%H%M%S')}"
            )
        
        self._checkpoint_path = checkpoint_path
        self._save_dir = unique_dir_name
        os.makedirs(self._save_dir, exist_ok=True)

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(self._checkpoint_path)
    
    def save_checkpoint(self):
        agent_save_path = os.path.join(self._save_dir, f"actions_{self._counter}.pth")
        model_dict = {
            'actor': self._actor.state_dict(),
            'critic': self._critic.state_dict(),
            'actor_optimizer': self._actor_optimizer.state_dict(),
            'critic_optimizer': self._critic_optimizer.state_dict(),
            'learning_rate': self._learning_rate,
            'gamma': self._gamma,
            'clip_ratio': self._clip_ratio,
            'batch_size': self._batch_size,
            'n_epochs': self._n_epochs,
            'gae_lambda': self._gae_lambda,
            'entropy_coef': self._entropy_coef,
            'value_coef': self._value_coef,
            'input_size': self.input_size,
            'output_size': self.output_size 
        }
        torch.save(model_dict, agent_save_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._actor.load_state_dict(checkpoint['actor'])
        self._critic.load_state_dict(checkpoint['critic'])
        self._actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self._critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self._learning_rate = checkpoint['learning_rate']
        self._gamma = checkpoint['gamma']
        self._clip_ratio = checkpoint['clip_ratio']
        self._batch_size = checkpoint['batch_size']
        self._n_epochs = checkpoint['n_epochs']
        self._gae_lambda = checkpoint['gae_lambda']
        self._entropy_coef = checkpoint['entropy_coef']
        self._value_coef = checkpoint['value_coef']
        self.input_size = checkpoint['input_size']
        self.output_size = checkpoint['output_size']
        print(f"Loaded checkpoint from {checkpoint_path}")

    def game_ended(self, game_state):
        # who won
        # set the last reward to 1 if win, -1 if lose, 0 if not done
        # last reward is the reward for the last end turn
        # the very last index should be the reward for the last action and should be set to 1

        print("=== Game Ended ===")

        # Handle each game state in the batch
        for game_idx, gs in enumerate(game_state):
            tribes = gs['board']['tribes']
            for tribe in tribes:
                active_tribe_id = tribe['actorId']
                # find the last action for the tribe in global_info
                for i in range(len(self._trajectories["global_info"])-1, -1, -1):
                    global_info = self._trajectories["global_info"][i]
                    if global_info[game_idx, 0] == active_tribe_id:
                        self._trajectories["rewards"][i] = torch.tensor([[reward_fn(gs, active_tribe_id)]], device=self.device)
                        print("Setting reward for tribe", active_tribe_id, "at index", i, "to", self._trajectories["rewards"][i])
                        break

        # the last turn for the tribe that didn't win should be set to 0
    
    # @profile
    def run(self, game_id, game_state, valid_actions):
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

        # Collect trajectory
        actions, log_probs = self.get_action(spatial_tensor, global_info, masks, device=self.device)

        self._trajectories["game_id"].append(game_id)
        self._trajectories["spatial_tensor"].append(spatial_tensor)
        self._trajectories["global_info"].append(global_info)
        self._trajectories["actions"].append(actions)

        # Rewards are calculated after the game ends
        self._trajectories["rewards"].append(torch.tensor([[0]], device=self.device))
        self._trajectories["log_probs"].append(log_probs)
        self._trajectories["masks"].append(masks)
        
        if self._counter % self._batch_size == 0:
            # Convert list of tensors to single tensor for easier manipulation
            global_info_tensor = torch.cat(self._trajectories["global_info"], dim=0)
            game_ids = torch.tensor(self._trajectories["game_id"], device=global_info_tensor.device)
            
            # Get tribe IDs from first column
            tribe_ids = global_info_tensor[:, 0]

            # Create a composite key for sorting: game_id * max_tribe_id + tribe_id
            max_tribe_id = tribe_ids.max() + 1
            sort_keys = game_ids * max_tribe_id + tribe_ids

            # Get indices that would sort by tribe ID while preserving order
            sorted_indices = torch.argsort(sort_keys, stable=True)
            
            # Apply sorting to all trajectory components
            self._trajectories["game_id"] = [self._trajectories["game_id"][i] for i in sorted_indices]
            self._trajectories["spatial_tensor"] = [self._trajectories["spatial_tensor"][i] for i in sorted_indices]
            self._trajectories["global_info"] = [self._trajectories["global_info"][i] for i in sorted_indices]
            self._trajectories["actions"] = [self._trajectories["actions"][i] for i in sorted_indices]
            self._trajectories["rewards"] = [self._trajectories["rewards"][i] for i in sorted_indices]
            self._trajectories["log_probs"] = [self._trajectories["log_probs"][i] for i in sorted_indices]
            self._trajectories["masks"] = [self._trajectories["masks"][i] for i in sorted_indices]

            # print("\n=== Trajectory Sorting Debug ===")
            # print("Sequence of (Game ID, Tribe ID) pairs:")
            # 
            # # Show the first 20 and last 20 entries to verify sorting
            # entries = list(zip(
            #     [int(gid) for gid in self._trajectories["game_id"]], 
            #     [int(self._trajectories["global_info"][i][0, 0].item()) for i in range(len(self._trajectories["game_id"]))]
            # ))
            # 
            # print("\nFirst 20 entries:")
            # for i, (game_id, tribe_id) in enumerate(entries[:20]):
            #     print(f"{i:4d}: Game {game_id:2d}, Tribe {tribe_id:2d}")
            # 
            # print("\n... middle entries omitted ...")
            # 
            # print("\nLast 20 entries:")
            # for i, (game_id, tribe_id) in enumerate(entries[-20:], start=len(entries)-20):
            #     print(f"{i:4d}: Game {game_id:2d}, Tribe {tribe_id:2d}")
            # 
            # print("\nTotal trajectories:", len(entries))
            # print("===========================\n")

            self._update()

            if self._counter % (25 * self._batch_size) == 0:
                self.save_checkpoint()

            self._trajectories = copy.deepcopy(self._base_trajectories)

        return actions[0].tolist()

    
    def calculate_advantages(self, rewards, values, dones):
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
        # Move tensors to CPU for calculation — much faster than GPU
        rewards = rewards.cpu()
        values = values.cpu()
        dones = dones.cpu()

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

        # Move results back to the original device
        return returns.to(self.training_device), advantages.to(self.training_device)

    
    # @profile
    def _update(self):
        # Move models to training device before update
        self._actor.to(self.training_device)
        self._critic.to(self.training_device)

        # Move tensors to training device
        old_log_probs = torch.cat(self._trajectories["log_probs"], dim=0).to(self.training_device).float()
        actions = torch.cat(self._trajectories["actions"], dim=0).to(self.training_device).float()
        rewards = torch.cat(self._trajectories["rewards"], dim=0).to(self.training_device).float()
        dones = (rewards != 0).float().to(self.training_device)
        spatial_tensor = torch.cat(self._trajectories["spatial_tensor"], dim=0).to(self.training_device).float()
        global_info = torch.cat(self._trajectories["global_info"], dim=0).to(self.training_device).float()
        masks = torch.cat(self._trajectories["masks"], dim=0).to(self.training_device).float()

        # Move multipliers to training device
        multipliers = self.multipliers.to(self.training_device)

        # Calculate values, advantages, and returns once before the epoch loop
        # NOTE: We calculate the advantages BEFORE the updates because it represents the empirical advantage from our taken actions
        values = self._critic.forward(spatial_tensor, global_info)
        rewards_to_go, advantages = self.calculate_advantages(rewards, values.detach(), dones.detach())
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages since rewards are extremely sparse

        for _ in range(self._n_epochs):
            # print("\nInput verification:")
            # print(f"old_log_probs requires_grad: {old_log_probs.requires_grad}")
            # print(f"advantages non-zero elements: {(advantages != 0).sum().item()}")
            # print(f"advantages mean: {advantages.mean().item()}")

            # Get new logits directly from actor and apply masking
            new_logits = self._actor.forward(spatial_tensor, global_info)
            # new_masked_logits = new_logits.add(masks)
            new_masked_logits = new_logits + masks
            new_probs = torch.softmax(new_masked_logits.flatten(start_dim=1), dim=1)

            # print("\nDEBUG:")
            # print(f"New logits range: {new_logits.min().item():.4f} to {new_logits.max().item():.4f}")
            # print(f"Masked logits range: {new_masked_logits.min().item():.4f} to {new_masked_logits.max().item():.4f}")
            # print(f"New probs range: {new_probs.min().item():.4f} to {new_probs.max().item():.4f}")


            dist = Categorical(new_probs)
            flat_actions = (actions * multipliers).sum(dim=1)
            new_log_probs = dist.log_prob(flat_actions)
            # print(f"New log probs shape: {new_log_probs.shape}")

#             # Debug values before ratio calculation
#             print("\nPre-ratio values:")
#             print(f"new_log_probs: {new_log_probs[:5]}")  # Show first 5 values
#             print(f"old_log_probs: {old_log_probs[:5]}")
# 
#             # More debugging
#             print(f"New log probs range: {new_log_probs.min().item():.4f} to {new_log_probs.max().item():.4f}")
#             print(f"Old log probs range: {old_log_probs.min().item():.4f} to {old_log_probs.max().item():.4f}")

            # entropy = probs.entropy()  

            # Actor loss
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            print(f"Ratio range: {ratio.min().item():.4f} to {ratio.max().item():.4f}")
            print(f"Advantages range: {advantages.min().item():.4f} to {advantages.max().item():.4f}")

            surrogate1 = ratio * advantages
            print(f"Mean of surrogate1: {surrogate1.mean().item()}")
            surrogate2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantages
            print(f"Mean of surrogate2: {surrogate2.mean().item()}")
            actor_loss = -torch.min(surrogate1, surrogate2).mean() - self._entropy_coef * dist.entropy().mean()

            print(f"Normalized advantages: {advantages[:5]}")
            print(f"Surrogate 1: {surrogate1[:10]}")
            print(f"Surrogate 2: {surrogate2[:10]}")

            print(f"Actor loss: {actor_loss.item():.6f}")

            self._actor_optimizer.zero_grad()
            actor_loss.backward()

            # # Check gradients of intermediate values
            # print("\nGradient flow check:")
            # print(f"new_logits grad: {new_logits.grad is not None}")
            # print(f"new_masked_logits grad: {new_masked_logits.grad is not None}")
            # print(f"new_probs grad: {new_probs.grad is not None}")
            # print(f"new_log_probs grad: {new_log_probs.grad is not None}")
            # print(f"ratio grad: {ratio.grad is not None}")

            print("\nGradient check:")
            for name, param in self._actor.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"{name} grad norm: {grad_norm:.6f}")
                else:
                    print(f"{name} has no gradient")
            # print("\nParameter change check:")
            # for name, param in self._actor.named_parameters():
            #     print(f"{name} param norm: {param.norm().item():.6f}")

            # Critic loss
            new_values = self._critic.forward(spatial_tensor, global_info)
            critic_loss = nn.MSELoss()(new_values, rewards_to_go) * self._value_coef

            self._critic_optimizer.zero_grad()
            critic_loss.backward()

            # Log training metrics
            if self._counter % (self._batch_size * 5) == 0:
                self._log_training_metrics(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    ratio=ratio,
                    advantages=advantages,
                    values=values,
                    rewards_to_go=rewards_to_go
                )

                self.save_critical_info(
                    counter=self._counter,
                    actor_loss=actor_loss,
                    value_loss=critic_loss,
                    surrogate1=surrogate1,
                    surrogate2=surrogate2,
                    ratio=ratio,
                    advantages=advantages,
                    returns=rewards_to_go
                )
                
            self._actor_optimizer.step()
            self._critic_optimizer.step()
            
        # Move models back to CPU after update
        self._actor.to(self.device)
        self._critic.to(self.device)

    
    # @profile
    def get_action(self, spatial_tensor, global_info, masks, device=None):
        try:
            # Use provided device or default to self.device
            device = device or self.device

            # Ensure inputs are on GPU
            spatial_tensor = spatial_tensor.to(device).float()
            global_info = global_info.to(device).float()
            masks = masks.to(device).float()

            # Get action logits from actor network
            logits = self._actor.forward(spatial_tensor, global_info) # -> (batch_size, action_space_size)
            
            # Add mask to logits to keep valid actions and mask out invalid ones
            masked_logits = logits.add(masks)  # Using in-place batch addition

            # Get action probabilities and select action
            probs = torch.softmax(masked_logits.flatten(start_dim=1), dim=1)  # Keep batch dimension

            dist = Categorical(probs)
            flat_indices = dist.sample()  # Will sample for each item in batch

            # Calculate log_prob of chosen action immediately
            log_probs = dist.log_prob(flat_indices)

            # Ensure multipliers are on the correct device
            multipliers = self.multipliers.to(device)

            # Ensure the output size tensor is on the correct device
            output_size_tensor = torch.tensor(self.output_size, device=device)

            actions = (flat_indices.unsqueeze(1) // multipliers) % output_size_tensor

            return actions.long(), log_probs
        
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
    
    def save_critical_info(self, counter: int, actor_loss: torch.Tensor, value_loss: torch.Tensor, 
                        surrogate1: torch.Tensor, surrogate2: torch.Tensor, ratio: torch.Tensor, 
                        advantages: torch.Tensor, returns: torch.Tensor):
        """
        Log critical information about the training process every 10 games
        """
        log_path = os.path.join(self._save_dir, '_training_log.txt')
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write('')  # Create an empty file

        with open(log_path, 'a') as f:
            f.write(f"\n{'='*20} Count {counter} {'='*20}\n")
            f.write(f"Actor Loss: {actor_loss.item():.6f}\n")
            f.write(f"Value Loss: {value_loss.item():.6f}\n")
            f.write(f"Ratio range: {ratio.min().item():.4f} to {ratio.max().item():.4f}\n")
            f.write(f"Advantages range: {advantages.min().item():.4f} to {advantages.max().item():.4f}\n")
            f.write(f"Mean of surrogate1: {surrogate1.mean().item()}\n")
            f.write(f"Mean of surrogate2: {surrogate2.mean().item()}\n")
            
            # Log sample of advantages
            num_elements = 7
            truncated_advantages = [round(a.item(), 6) for a in advantages[:num_elements]] + ["..."] + [round(a.item(), 6) for a in advantages[-num_elements:]]
            f.write(f"Advantages sample: {truncated_advantages}\n")
            
            # Log returns statistics
            f.write(f"Returns: {returns.mean().item():.6f} (mean), {returns.std().item():.6f} (std)\n")
            
            # Log actor network parameters
            f.write("\nActor Network Parameters:\n")
            for name, param in self._actor.named_parameters():
                param_norm = param.norm().item()
                f.write(f"{name} norm: {param_norm:.6f}\n")
            
            # Log critic network parameters
            f.write("\nCritic Network Parameters:\n")
            for name, param in self._critic.named_parameters():
                param_norm = param.norm().item()
                f.write(f"{name} norm: {param_norm:.6f}\n")
            
            # Log actor gradients
            f.write("\nActor Gradients:\n")
            for name, param in self._actor.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    f.write(f"{name} grad norm: {grad_norm:.6f}\n")
                else:
                    f.write(f"{name} has no gradient\n")
            
            # Log critic gradients
            f.write("\nCritic Gradients:\n")
            for name, param in self._critic.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    f.write(f"{name} grad norm: {grad_norm:.6f}\n")
                else:
                    f.write(f"{name} has no gradient\n")

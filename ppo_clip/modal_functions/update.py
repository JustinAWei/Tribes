import modal
import torch
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import joblib

# Define a Modal App
app = modal.App(
    "ppo_clip_update",
    image=modal.Image.debian_slim().pip_install("torch", "joblib")
)

@app.function()
@modal.web_endpoint(method="POST")
async def trigger_update(request):
    # Extract parameters from the request
    data = await request.json()
    # Call the GPU function asynchronously
    result = await remote_update.call(
        trajectories=data['trajectories'],
        output_size=data['output_size'],
        epochs=data['epochs'],
        epsilon=data['epsilon'],
        actor_optimizer=data['actor_optimizer'],
        critic_optimizer=data['critic_optimizer'],
        critic=data['critic'],
        advantage=data['advantage'],
        get_action=data['get_action']
    )
    return result

@app.function(gpu="t4")
async def remote_update(serialized_trajectories, output_size, epochs, epsilon, actor_optimizer_state, critic_optimizer_state, critic, advantage, get_action):
    # Ensure all tensors and models are moved to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models to GPU
    critic.to(device)

    # Initialize optimizers and load their states
    actor_optimizer = optim.Adam(critic.parameters())  # Initialize with the same parameters
    critic_optimizer = optim.Adam(critic.parameters())  # Initialize with the same parameters
    actor_optimizer.load_state_dict(actor_optimizer_state)
    critic_optimizer.load_state_dict(critic_optimizer_state)

    # Deserialize the data back to numpy arrays
    numpy_trajectories = joblib.loads(serialized_trajectories)
    
    # Convert numpy arrays back to PyTorch tensors
    trajectories = {
        key: [torch.tensor(array) for array in arrays]
        for key, arrays in numpy_trajectories.items()
    }
    
    # Unpack trajectories and move to GPU
    probs = torch.cat(trajectories["probs"], dim=0).to(device)
    actions = torch.cat(trajectories["actions"], dim=0).to(device)

    dist = Categorical(probs)
    flat_actions = np.ravel_multi_index(tuple(actions.T), output_size)
    flat_actions = torch.tensor(flat_actions).to(device)
    old_log_probs = dist.log_prob(flat_actions)

    for i in range(epochs):
        rewards = torch.cat(trajectories["rewards"], dim=0).to(device)
        dones = (rewards != 0).float().to(device)
        spatial_tensor = torch.cat(trajectories["spatial_tensor"], dim=0).to(device)
        global_info = torch.cat(trajectories["global_info"], dim=0).to(device)
        masks = torch.cat(trajectories["masks"], dim=0).to(device)

        values = critic.forward(spatial_tensor, global_info)
        rewards_to_go, advantages = advantage(rewards, values.detach(), dones.detach())

        actor_optimizer.zero_grad()
        _, new_probs = get_action(spatial_tensor, global_info, masks)
        dist = Categorical(new_probs)
        flat_actions = np.ravel_multi_index(tuple(actions.T), output_size)
        flat_actions = torch.tensor(flat_actions).to(device)
        new_log_probs = dist.log_prob(flat_actions)

        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(values, rewards_to_go)
        loss.backward()
        critic_optimizer.step()

    return new_log_probs
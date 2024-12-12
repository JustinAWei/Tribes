import torch
import numpy as np

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

def ppo_clip(game_state, valid_actions, action_space_shape):

    # output matrix in the action space
    # rand matrix between 0MASK of action_space_shape
    # TODO: actually implement the PPO-clip
    action_space = torch.rand(action_space_shape)

    print(valid_actions)
    mask = create_multidimensional_mask(torch.tensor(valid_actions), action_space_shape)
    print("Mask shape:", mask.shape)

    # use the mask to filter the valid actions by
    valid_action_space = action_space * mask
    # count nonzero elements
    print("Number of nonzero elements:", torch.count_nonzero(valid_action_space), "Number of 1s in the mask:", torch.sum(mask))

    # softmax and choose the action
    valid_action_space = torch.softmax(valid_action_space, dim=len(action_space_shape) - 1)

    # Flatten tensor and get argmax
    flat_index = torch.argmax(valid_action_space.flatten())
    
    # Convert flat index back to multi-dimensional indices
    action = np.unravel_index(flat_index.item(), action_space_shape)
    # this is a np arr of np.int64, i want to convert it to a list of ints
    action = [int(i) for i in action]

    print("Action: ", action)

    return action
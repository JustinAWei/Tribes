from fastapi import FastAPI, Request
import uvicorn
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import json
from pprint import pprint
from datetime import datetime
import torch

from utils import ACTION_CATEGORIES, ACTION_TYPES, TECH_TYPES, BUILDING_TYPES, UNIT_TYPES, MASK, BOARD_LEN

# Create FastAPI app
app = FastAPI()

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

def get_actor_x_y(actor_id, gs):
    # in board -> gameActors
    game_actors = gs.get('board', {}).get('gameActors', {})
    actor = game_actors.get(str(actor_id), {})
    if not actor:
        return MASK, MASK

    position = actor.get('position', {})
    x = position.get('x', 0)
    y = position.get('y', 0)

    # print("actor_id", actor_id)
    # print(f"x: {x} | y: {y}")

    return x, y

def ppo_clip(game_state, valid_actions):
    NUM_CATEGORIES = 32
    action_space_shape = (len(ACTION_CATEGORIES), max(ACTION_TYPES.values()) + 1, BOARD_LEN, BOARD_LEN, BOARD_LEN, BOARD_LEN, NUM_CATEGORIES)

    # output matrix in the action space
    # rand matrix between 0MASK of action_space_shape
    # TODO: actually implement the PPO-clip
    action_space = torch.rand(action_space_shape)

    # mask
    coordinates = torch.tensor(valid_actions)
    print(coordinates)
    mask = create_multidimensional_mask(coordinates, action_space_shape)
    print("Mask shape:", mask.shape)
    print("Number of 1s in the mask:", torch.sum(mask))

    # use the mask to filter the valid actions by
    valid_action_space = action_space * mask
    # count nonzero elements
    print("Number of nonzero elements:", torch.count_nonzero(valid_action_space))

    # softmax and choose the action
    valid_action_space = torch.softmax(valid_action_space, dim=len(action_space_shape) - 1)

    # Flatten tensor and get argmax
    flat_index = torch.argmax(valid_action_space.flatten())
    
    # Convert flat index back to multi-dimensional indices
    action = np.unravel_index(flat_index.item(), action_space_shape)
    # this is a np arr of np.int64, i want to convert it to a list of ints
    action = [int(i) for i in action]

    print("Action:")
    print(action)

    return action

@app.post("/receive")
async def receive_data(request: Request):
    """
    Process and filter city actions from game state
    """
    # Extract request data
    data = await request.json()

    try:
        # Parse game state
        gs = json.loads(data['gameState']) if isinstance(data['gameState'], str) else data['gameState']
        # pprint(gs)

        # Save game state only every 20th request
        # if request_counter % 20 == 0:
        #     # Create timestamped filename
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     filename = f'game_state_{timestamp}.json'
        #     
        #     # Save game state to file
        #     with open(filename, 'w') as f:
        #         json.dump(gs, f, indent=4)
        #     print(f"Game state saved to '{filename}'")

        BOARD_LEN = len(gs['board']['terrains'])
        BOARD_SIZE = BOARD_LEN ** 2

        print("board len", BOARD_LEN)
        print("board size", BOARD_SIZE)

        # Allowed action types (both numbers and strings)
        allowed_action_types = list(ACTION_TYPES.keys()) + list(ACTION_TYPES.values())

        # List to store filtered city actions
        filtered_city_actions = []

        valid_actions = []

        # this is just a list
        tribe_actions = gs.get('tribeActions', [])
        print("tribe_actions")
        print(tribe_actions)
        filtered_tribe_actions = [
            action for action in tribe_actions
            if action.get('actionType') in allowed_action_types
        ]
        print("filtered_tribe_actions")
        print(filtered_tribe_actions)
        for action in filtered_tribe_actions:
            if action.get('actionType') == 'RESEARCH_TECH':
                tech_type = TECH_TYPES[action.get('tech')]
                valid_actions.append([ACTION_CATEGORIES["TRIBE"], ACTION_TYPES[action.get('actionType')], MASK, MASK, MASK, MASK, tech_type])
            else:
                valid_actions.append([ACTION_CATEGORIES["TRIBE"], ACTION_TYPES[action.get('actionType')], MASK, MASK, MASK, MASK, MASK])

        # Process city actions
        city_actions = gs.get('cityActions', {})
        print("city_actions")
        print(city_actions)
        for city_id, actions in city_actions.items():
            # Filter actions for this city
            filtered_city_actions = [
                action for action in actions
                if action.get('actionType') in allowed_action_types
            ]
            print('city_id', 'filtered_city_actions')
            print(city_id, filtered_city_actions)
            for action in filtered_city_actions:
                x2, y2 = MASK, MASK
                if 'targetPos' in action:
                    x2 = action['targetPos']['x']
                    y2 = action['targetPos']['y']

                x1, y1 = get_actor_x_y(int(city_id), gs)

                if action.get('actionType') == 'BUILD':
                    building_type = BUILDING_TYPES[action.get('buildingType')]
                    valid_actions.append([ACTION_CATEGORIES["CITY"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2, building_type])
                elif action.get('actionType') == 'SPAWN':
                    unit_type = UNIT_TYPES[action.get('unit_type')]
                    valid_actions.append([ACTION_CATEGORIES["CITY"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2, unit_type])
                else:
                    valid_actions.append([ACTION_CATEGORIES["CITY"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2, MASK])

        unit_actions = gs.get('unitActions', {})
        print("unit_actions")
        print(unit_actions)
        for unit_id, actions in unit_actions.items():
            # Filter actions for this city
            filtered_unit_actions = [
                action for action in actions
                if action.get('actionType') in allowed_action_types
            ]
            print('unit_id', 'filtered_unit_actions')
            print(unit_id, filtered_unit_actions)
            for action in filtered_unit_actions:
                x1, y1 = get_actor_x_y(int(unit_id), gs)

                x2, y2 = MASK, MASK
                if 'destination' in action:
                    x2 = action['destination']['x']
                    y2 = action['destination']['y']

                elif 'targetId' in action:
                    x2, y2 = get_actor_x_y(int(action['targetId']), gs)

                elif 'cityId' in action:
                    x2, y2 = get_actor_x_y(int(action['cityId']), gs)

                elif action.get('actionType') == 'CAPTURE':
                    # Assumption: Captures are always to the same position as the unit
                    x2, y2 = x1, y1


                valid_actions.append([ACTION_CATEGORIES["UNIT"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2, MASK])

        action = ppo_clip(gs, valid_actions)

        return {
            "status": 200, 
            "action": action
        }

    except Exception as e:
        print(f"Error processing game state: {e}")
        return {"status": 500, "message": str(e)}


@app.get("/")
async def root():
    """
    Root endpoint to confirm server is running
    """
    return {"message": "City Actions Filter Server is running! Send POST requests to /receive."}

if __name__ == "__main__":
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)
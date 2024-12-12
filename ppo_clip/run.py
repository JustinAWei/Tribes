from fastapi import FastAPI, Request
import uvicorn
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import json
from pprint import pprint

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
    mask = np.zeros(shape, dtype=int)

    # Set the specified coordinates to 1
    for coord in coordinates:
        # Ensure the coordinate is within the matrix bounds
        if len(coord) == len(shape) and all(0 <= c < s for c, s in zip(coord, shape)):
            mask[tuple(coord)] = 1

    return mask

# Reversed dictionary for action types
ACTION_TYPES = {
    "RESOURCE_GATHERING": 6,
    "SPAWN": 7,
    "END_TURN": 9,
    "ATTACK": 13,
    "CAPTURE": 14,
    "MOVE": 20
}

# Reversed dictionary for action categories
ACTION_CATEGORIES = {
    "TRIBE": 0,
    "CITY": 1,
    "UNIT": 2
}

BOARD_LEN = 8
BOARD_SIZE = BOARD_LEN ** 2

class FilteredActions(BaseModel):
    """
    Model to structure filtered city actions
    """
    city_id: str
    filtered_actions: List[Dict[str, Any]]

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
        filtered_tribe_actions = [
            action for action in tribe_actions
            if action.get('actionType') in allowed_action_types
        ]
        print("filtered_tribe_actions")
        print(filtered_tribe_actions)
        for action in filtered_tribe_actions:
            valid_actions.append([ACTION_CATEGORIES["TRIBE"], action.get('tribeId'), ACTION_TYPES[action.get('actionType')], 0, 0])

        # Process city actions
        city_actions = gs.get('cityActions', {})
        for city_id, actions in city_actions.items():
            # Filter actions for this city
            filtered_city_actions = [
                action for action in actions
                if action.get('actionType') in allowed_action_types
            ]
            print('city_id', 'filtered_city_actions')
            print(city_id, filtered_city_actions)
            for action in filtered_city_actions:
                if 'targetPos' in action:
                    x = action['targetPos']['x']
                    y = action['targetPos']['y']

                valid_actions.append([ACTION_CATEGORIES["CITY"], int(city_id), ACTION_TYPES[action.get('actionType')], x, y])

        unit_actions = gs.get('unitActions', {})
        for unit_id, actions in unit_actions.items():
            # Filter actions for this city
            filtered_unit_actions = [
                action for action in actions
                if action.get('actionType') in allowed_action_types
            ]
            print('unit_id', 'filtered_unit_actions')
            print(unit_id, filtered_unit_actions)
            for action in filtered_unit_actions:
                # TODO: get x,y from targetId and cityId
                x,y = 0,0
                if 'destination' in action:
                    x = action['destination']['x']
                    y = action['destination']['y']

                valid_actions.append([ACTION_CATEGORIES["UNIT"], int(unit_id), ACTION_TYPES[action.get('actionType')], x, y])


        coordinates = np.array(valid_actions)

        print(coordinates)

        # Assumption: you will never create more units than 4x board_size
        MAX_ID = BOARD_SIZE * 4
        matrix_shape = (len(ACTION_CATEGORIES), MAX_ID, max(ACTION_TYPES.values()) + 1, BOARD_LEN, BOARD_LEN)

        try:
            mask = create_multidimensional_mask(coordinates, matrix_shape)
            print("Mask created successfully.")
            print("Mask shape:", mask.shape)
            print("Number of 1s in the mask:", np.sum(mask))
        except Exception as e:
            print("Error creating mask:", e)

        return {
            "status": "Data processed",
            "filtered_city_actions": valid_actions
        }

    except Exception as e:
        print(f"Error processing game state: {e}")
        return {"status": "Error", "message": str(e)}


@app.get("/")
async def root():
    """
    Root endpoint to confirm server is running
    """
    return {"message": "City Actions Filter Server is running! Send POST requests to /receive."}

if __name__ == "__main__":
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)
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
    "LEVEL_UP": 5,
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

def get_actor_x_y(actor_id, gs):
    # in board -> gameActors
    game_actors = gs.get('board', {}).get('gameActors', {})
    actor = game_actors.get(str(actor_id), {})
    if not actor:
        return -1, -1

    position = actor.get('position', {})
    x = position.get('x', 0)
    y = position.get('y', 0)

    print("actor_id", actor_id)
    print(f"x: {x} | y: {y}")

    return x, y

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
        print("tribe_actions")
        print(tribe_actions)
        filtered_tribe_actions = [
            action for action in tribe_actions
            if action.get('actionType') in allowed_action_types
        ]
        print("filtered_tribe_actions")
        print(filtered_tribe_actions)
        for action in filtered_tribe_actions:
            valid_actions.append([ACTION_CATEGORIES["TRIBE"], ACTION_TYPES[action.get('actionType')], -1, -1, -1, -1])

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
                if 'targetPos' in action:
                    x2 = action['targetPos']['x']
                    y2 = action['targetPos']['y']

                x1, y1 = get_actor_x_y(int(city_id), gs)

                valid_actions.append([ACTION_CATEGORIES["CITY"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2])

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
                # TODO: get x,y from targetId and cityId
                x2, y2 = 0, 0
                if 'destination' in action:
                    x2 = action['destination']['x']
                    y2 = action['destination']['y']

                elif 'targetId' in action:
                    x2, y2 = get_actor_x_y(int(action['targetId']), gs)

                elif 'cityId' in action:
                    x2, y2 = get_actor_x_y(int(action['cityId']), gs)

                x1, y1 = get_actor_x_y(int(unit_id), gs)

                valid_actions.append([ACTION_CATEGORIES["UNIT"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2])


        coordinates = np.array(valid_actions)
        print(coordinates)

        matrix_shape = (len(ACTION_CATEGORIES), max(ACTION_TYPES.values()) + 1, BOARD_LEN, BOARD_LEN, BOARD_LEN, BOARD_LEN)

        try:
            mask = create_multidimensional_mask(coordinates, matrix_shape)
            print("Mask created successfully.")
            print("Mask shape:", mask.shape)
            print("Number of 1s in the mask:", np.sum(mask))
        except Exception as e:
            print("Error creating mask:", e)

        random_action = valid_actions[np.random.randint(len(valid_actions))]
        return {
            "status": "Data processed", 
            "action": random_action
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
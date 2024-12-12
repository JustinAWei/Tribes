from fastapi import FastAPI, Request
import uvicorn
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import json
from pprint import pprint
from datetime import datetime

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
    "BUILD": 0,
    "LEVEL_UP": 5,
    "RESOURCE_GATHERING": 6,
    "SPAWN": 7,

    "END_TURN": 9,
    "RESEARCH_TECH": 10,

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


# Unit type mapping
UNIT_TYPES = {
    "WARRIOR": 0,
    "RIDER": 1, 
    "DEFENDER": 2,
    "SWORDMAN": 3,
    "ARCHER": 4,
    "CATAPULT": 5,
    "KNIGHT": 6,
    "MIND_BENDER": 7,
    "BOAT": 8,
    "SHIP": 9,
    "BATTLESHIP": 10,
    "SUPERUNIT": 11
}

# Building type mapping
BUILDING_TYPES = {
    "PORT": 0,
    "MINE": 1,
    "FORGE": 2,
    "FARM": 3,
    "WINDMILL": 4,
    "CUSTOMS_HOUSE": 5,
    "LUMBER_HUT": 6,
    "SAWMILL": 7,
    "TEMPLE": 8,
    "WATER_TEMPLE": 9,
    "FOREST_TEMPLE": 10,
    "MOUNTAIN_TEMPLE": 11,
    "ALTAR_OF_PEACE": 12,
    "EMPERORS_TOMB": 13,
    "EYE_OF_GOD": 14,
    "GATE_OF_POWER": 15,
    "GRAND_BAZAR": 16,
    "PARK_OF_FORTUNE": 17,
    "TOWER_OF_WISDOM": 18
}
# Technology type mapping
TECH_TYPES = {
    "CLIMBING": 0,
    "FISHING": 1,
    "HUNTING": 2,
    "ORGANIZATION": 3,
    "RIDING": 4,
    "ARCHERY": 5,
    "FARMING": 6,
    "FORESTRY": 7,
    "FREE_SPIRIT": 8,
    "MEDITATION": 9,
    "MINING": 10,
    "ROADS": 11,
    "SAILING": 12,
    "SHIELDS": 13,
    "WHALING": 14,
    "AQUATISM": 15,
    "CHIVALRY": 16,
    "CONSTRUCTION": 17,
    "MATHEMATICS": 18,
    "NAVIGATION": 19,
    "SMITHERY": 20,
    "SPIRITUALISM": 21,
    "TRADE": 22,
    "PHILOSOPHY": 23
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

# request_counter = 0

@app.post("/receive")
async def receive_data(request: Request):
    """
    Process and filter city actions from game state
    """
    # global request_counter
    # request_counter += 1

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
                valid_actions.append([ACTION_CATEGORIES["TRIBE"], ACTION_TYPES[action.get('actionType')], -1, -1, -1, -1, tech_type])
            else:
                valid_actions.append([ACTION_CATEGORIES["TRIBE"], ACTION_TYPES[action.get('actionType')], -1, -1, -1, -1, -1])

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
                x2, y2 = -1, -1
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
                    valid_actions.append([ACTION_CATEGORIES["CITY"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2, -1])

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

                x2, y2 = -1, -1
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


                valid_actions.append([ACTION_CATEGORIES["UNIT"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2, -1])


        coordinates = np.array(valid_actions)
        print(coordinates)
        NUM_CATEGORIES = 32
        matrix_shape = (len(ACTION_CATEGORIES), max(ACTION_TYPES.values()) + 1, BOARD_LEN, BOARD_LEN, BOARD_LEN, BOARD_LEN, NUM_CATEGORIES)

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
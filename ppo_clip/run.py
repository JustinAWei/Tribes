from fastapi import FastAPI, Request
import uvicorn
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import json
from pprint import pprint
from datetime import datetime

from utils import ACTION_CATEGORIES, ACTION_TYPES, TECH_TYPES, BUILDING_TYPES, UNIT_TYPES, MASK, get_actor_x_y
from model import ppo_clip

# Create FastAPI app
app = FastAPI()

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
        
        # Building types, unit types, tech types, etc.
        MAX_EXTRA_VARS = 32
        action_space_shape = (len(ACTION_CATEGORIES), max(ACTION_TYPES.values()) + 1, BOARD_LEN, BOARD_LEN, BOARD_LEN, BOARD_LEN, MAX_EXTRA_VARS)

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
            elif action.get('actionType') == 'BUILD_ROAD':
                # get position
                x, y = action.get('position').get('x'), action.get('position').get('y')
                valid_actions.append([ACTION_CATEGORIES["TRIBE"], ACTION_TYPES[action.get('actionType')], x, y, MASK, MASK, MASK])
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
                elif action.get('actionType') == 'LEVEL_UP':
                    bonus_type = BONUS_TYPES[action.get('bonus')]
                    valid_actions.append([ACTION_CATEGORIES["CITY"], ACTION_TYPES[action.get('actionType')], x1, y1, MASK, MASK, bonus_type])
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

                elif action.get('actionType') in ['CAPTURE', 'RECOVER', 'EXAMINE', 'MAKE_VETERAN']:
                    # Assumption: These are always to the same position as the unit
                    x2, y2 = x1, y1

                valid_actions.append([ACTION_CATEGORIES["UNIT"], ACTION_TYPES[action.get('actionType')], x1, y1, x2, y2, MASK])

        action = ppo_clip(gs, valid_actions, action_space_shape)

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
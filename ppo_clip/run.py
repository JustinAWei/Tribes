from fastapi import FastAPI, Request
import uvicorn
import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import json

# Create FastAPI app
app = FastAPI()

# Enum-like dictionary for action types
ACTION_TYPES = {
    6: "RESOURCE_GATHERING",
    7: "SPAWN",
    9: "END_TURN",
    13: "ATTACK",
    14: "CAPTURE",
    20: "MOVE"
}

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

        # Allowed action types (both numbers and strings)
        allowed_action_types = list(ACTION_TYPES.keys()) + list(ACTION_TYPES.values())

        # List to store filtered city actions
        filtered_city_actions = []

        # Process city actions
        city_actions = gs.get('cityActions', {})
        for city_id, actions in city_actions.items():
            # Filter actions for this city
            city_filtered_actions = [
                action for action in actions
                if action.get('actionType') in allowed_action_types
            ]

            # If there are filtered actions, add to the list
            if city_filtered_actions:
                filtered_city_actions.append({
                    "city_id": city_id,
                    "filtered_actions": city_filtered_actions
                })

        # Log the filtered actions
        print("Filtered City Actions:")
        for city_action_set in filtered_city_actions:
            print(f"City ID: {city_action_set['city_id']}")
            for action in city_action_set['filtered_actions']:
                print(f"  Action: {action}")

        return {
            "status": "Data processed",
            "filtered_city_actions": filtered_city_actions
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
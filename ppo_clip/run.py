from fastapi import FastAPI, Request
import uvicorn
import numpy as np
from pydantic import BaseModel, Field
from typing import Any
import json
# Create FastAPI app
app = FastAPI()

'''
City
Spawn unit, city, unit type,
Gather resource, city id, target position

Tribe
End turn, tribe id

Unit
Attack unitId, targetid
Capture, unitid, cityid
Move, unitid, x,y


[x1,y1, type, id, action]
'''


from pydantic import BaseModel


@app.post("/receive")
async def receive_data(request: Request):
    # Extract and print request data

    # Filter actions for relevant ones
    data = await request.json()
    gs = data['gameState']
    print(gs)

    BOARD_SIZE = 8 * 8
    # RESOURCE_GATHERING: 6, SPAWN: 7, END_TURN: 9, ATTACK: 13, CAPTURE: 14, MOVE: 20,
    allowed_action_types_nums = [6, 7, 9, 13, 14, 20]
    allowed_action_types = ["RESOURCE_GATHERING", "SPAWN", "END_TURN", "ATTACK", "CAPTURE", "MOVE"]

    gs = json.loads(gs)
    print(gs)

    city_actions = gs['cityActions']
    tribe_actions = gs['tribeActions']
    unit_actions = gs['unitActions']

    print('city_actions')
    print(city_actions)
    print('tribe_actions')
    print(tribe_actions)

    # filter out city actions
    # for each city
    for city, actions in city_actions.items():
        print("city", "actions")
        print(city, actions)
        for action in actions:
            print("action", action)
            if action['actionType'] in allowed_action_types:
                print(action)

#     print("Received data:", data)
    return {"status": "Data received", "received_data": data}

@app.get("/")
async def root():
    return {"message": "Server is running! Send POST requests to /receive."}

if __name__ == "__main__":
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)

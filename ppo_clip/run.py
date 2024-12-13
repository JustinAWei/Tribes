from fastapi import FastAPI, Request
import uvicorn
import json

from utils import ACTION_CATEGORIES, ACTION_TYPES, MAX_EXTRA_VARS, filter_actions
from model import PPOClipAgent
from utils import BOARD_LEN

game_state_shape = (BOARD_LEN, BOARD_LEN, 27)
action_space_shape = (max(ACTION_TYPES.values()), BOARD_LEN, BOARD_LEN, BOARD_LEN, BOARD_LEN, MAX_EXTRA_VARS)

agent = PPOClipAgent(game_state_shape, action_space_shape)

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

        # BOARD_LEN = len(gs['board']['terrains'])
        # BOARD_SIZE = BOARD_LEN ** 2
        
        valid_actions = filter_actions(gs)
        
        action = agent.run(0, gs, valid_actions)
        print("Action: ", action)

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
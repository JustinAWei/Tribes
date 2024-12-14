from fastapi import FastAPI, Request
import uvicorn
import json

from utils import filter_actions, action_tuples, index_to_action_tuple
from model import PPOClipAgent
from utils import BOARD_LEN

game_state_shape = (BOARD_LEN, BOARD_LEN, 27)
action_space_shape = (BOARD_LEN, BOARD_LEN, BOARD_LEN, BOARD_LEN, len(action_tuples))

print(action_space_shape)

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
        # print("Action: ", action)

        # convert last action to (action_type, extra_var)
        action_type, extra_var = index_to_action_tuple(action[-1])
        print("Action Type: ", action_type)
        print("Extra Var: ", extra_var)

        unpacked_action = [action[0], action[1], action[2], action[3], action_type, extra_var]

        return {
            "status": 200, 
            "action": unpacked_action
        }

    except Exception as e:
        print(f"Error processing game state: {e}")
        print("spatial_tensor shape:", len(agent._trajectories["spatial_tensor"]), agent._trajectories["spatial_tensor"][0].shape)
        print("global_info shape:", len(agent._trajectories["global_info"]), agent._trajectories["global_info"][0].shape)
        print("actions shape:", len(agent._trajectories["actions"]), agent._trajectories["actions"][0].shape) 
        print("rewards shape:", len(agent._trajectories["rewards"]), agent._trajectories["rewards"][0].shape)
        print("probs shape:", len(agent._trajectories["probs"]), agent._trajectories["probs"][0].shape)
        print("mask shape:", len(agent._trajectories["masks"]), agent._trajectories["masks"][0].shape)
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
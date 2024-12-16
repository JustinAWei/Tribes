from collections import defaultdict
from fastapi import FastAPI, Request
import uvicorn
import json
from utils import filter_actions, action_tuples, index_to_action_tuple, ACTION_TYPES

from model import PPOClipAgent
from utils import BOARD_LEN
from pprint import pprint
game_state_shape = (BOARD_LEN, BOARD_LEN, 27)
action_space_shape = (BOARD_LEN, BOARD_LEN, len(action_tuples))

print(action_space_shape)

agent = PPOClipAgent(load_path="ppo_clip/weights", save_path="ppo_clip/weights", input_size=game_state_shape, output_size=action_space_shape)

# Create FastAPI app
app = FastAPI()

current_game_id = 0
action_type_stats = defaultdict(lambda: defaultdict(int))

@app.post("/end_game")
async def end_game(request: Request):
    global current_game_id
    global action_type_stats

    data = await request.json()

    agent.game_ended([data['gameState']])

    if current_game_id in action_type_stats:
        current_tick = data['gameState']['tick']
        print("Game Ended", current_game_id, "at tick", current_tick)
        # ACTION_TYPES is a dictionary of text to index, i want to print the text corresponding to the stats

        sorted_action_type_stats = sorted(action_type_stats[current_game_id].items(), key=lambda x: x[1], reverse=True)
        for action_type, index in ACTION_TYPES.items():
            print(f"{action_type}: {sorted_action_type_stats[current_game_id].get(index, 0)}")

    current_game_id = current_game_id + 1
    action_type_stats[current_game_id] = {}

    return {"status": 200}

@app.post("/receive")
async def receive_data(request: Request):
    """
    Process and filter city actions from game state
    """
    global current_game_id
    global action_type_stats

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
        x2, y2, action_type, extra_var = index_to_action_tuple(action[-1])
        # print("Action Type: ", action_type)
        # print("Extra Var: ", extra_var)

        unpacked_action = [action[0], action[1], x2, y2, action_type, extra_var]
        # print("Unpacked Action: ", unpacked_action)

        action_type_stats[current_game_id][action_type] = action_type_stats[current_game_id].get(action_type, 0) + 1

        return {
            "status": 200, 
            "action": unpacked_action
        }

    except Exception as e:
        print(f"Error: {e}")
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
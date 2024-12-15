import json
import torch
from utils import timing_decorator

# Constants for mapping strings to integers
TERRAIN_MAP = {
    'PLAIN': 0, 'SHALLOW_WATER': 1, 'DEEP_WATER': 2, 'MOUNTAIN': 3,
    'VILLAGE': 4, 'CITY': 5, 'FOREST': 6, 'FOG': 7
}

BUILDING_MAP = {
    'PORT': 0, 'MINE': 1, 'FORGE': 2, 'FARM': 3, 'WINDMILL': 4,
    'CUSTOMS_HOUSE': 5, 'LUMBER_HUT': 6, 'SAWMILL': 7, 'TEMPLE': 8,
    'WATER_TEMPLE': 9, 'FOREST_TEMPLE': 10, 'MOUNTAIN_TEMPLE': 11,
    'ALTAR_OF_PEACE': 12, 'EMPERORS_TOMB': 13, 'EYE_OF_GOD': 14,
    'GATE_OF_POWER': 15, 'GRAND_BAZAR': 16, 'PARK_OF_FORTUNE': 17,
    'TOWER_OF_WISDOM': 18
}

RESOURCE_MAP = {
    'FISH': 0, 'FRUIT': 1, 'ANIMAL': 2, 'WHALES': 3,
    'ORE': 5, 'CROPS': 6, 'RUINS': 7
}

UNIT_STATUS_MAP = {
    'FRESH': 0,
    'MOVED': 1,
    'ATTACKED': 2,
    'MOVED_AND_ATTACKED': 3,
    'PUSHED': 4,
    'FINISHED': 5
}

UNIT_TYPE_MAP = {
    'WARRIOR': 0, 'RIDER': 1, 'DEFENDER': 2, 'SWORDMAN': 3,
    'ARCHER': 4, 'CATAPULT': 5, 'KNIGHT': 6, 'MIND_BENDER': 7,
    'BOAT': 8, 'SHIP': 9, 'BATTLESHIP': 10, 'SUPERUNIT': 11
}

# @timing_decorator
# @profile
def game_state_to_vector(gs_list, device):
    """
    Vectorize a batch of game states into tensors.
    
    Args:
        gs_list: List of game states or single game state
        
    Returns:
        spatial_tensor: Shape (batch_size, board_size, board_size, 27)
        global_info: Shape (batch_size, 2)
    """
    # Handle single game state case
    if not isinstance(gs_list, list):
        gs_list = [gs_list]
    
    batch_size = len(gs_list)
    board_size = len(gs_list[0]['board']['terrains'])
    
    # Initialize batch tensors
    spatial_tensors = torch.zeros((batch_size, board_size, board_size, 27), dtype=torch.float32).to(device)
    global_infos = torch.zeros((batch_size, 2), dtype=torch.float32).to(device)
    
    # Process each game state
    for i, gs in enumerate(gs_list):
        spatial_tensor, global_info = _process_single_game_state(gs, device)
        spatial_tensors[i] = spatial_tensor
        global_infos[i] = global_info
        
    return spatial_tensors, global_infos

# @profile
def _process_single_game_state(gs, device):
    """
    Vectorize the game state into a numpy array. Below is the structure of the game state (and below that is the vectorization of the relevant features since not all features are relevant to the game for now).

    GAME STATE STRUCTURE:

    - board
        - terrain 
            - Board size where each cell is a terrain type that needs to be converted); E.g. [['FOG', ...]]
        - resources
            - Board size where each cell is a resource type that needs to be converted); E.g. [[None, 'FRUIT', ...]]
        - buildings
            - Board size where each cell is a building type that needs to be converted); E.g. [[None, 'FARM', ...]]
        - units
            - Board size where each cell is a unit ID, more information in "gameActors")); E.g. [[0, 2, ...]]
        - tribes
            - List of tribe information
            - E.g. [{'citiesID': [1],
                     'capitalID': 1,
                     'tribe': 'IMPERIUS',
                     'techTree': {'researched': [True, True, False, ...],
                                  'everythingResearched': False},
                     'stars': 3,
                     'winner': 'INCOMPLETE',
                     'score': 1903,
                     'obsGrid': [[False, True, ...]],
                     'connectedCities': [],
                     'monuments': {'PARK_OF_FORTUNE': 'UNAVAILABLE', ...},
                     'tribesMet': [0, 1],
                     'extraUnits': [],
                     'nKills': 1,
                     'nPacifistCount': 0,
                     'starsSent': 1,
                     'hasDeclaredWar': False,
                     'nWarsDeclared': 17,
                     'nStarsSent': 63,
                     'actorId': 0,
                     'tribeId': 0}]
        - capitalIDs
            - List of capital IDs (more information about it in "gameActors"); E.g. [1, 3]
        - tileCityId
            - Board size where each cell maps to the ID of the city that owns that tile
            - TODO: This is not confirmed, but if it's 0, I believe that means it's in the fog
            - -1 means no city owns it
            - E.g. [[0, 0, -1, 14, 14, 14, -1, -1, -1, -1, -1], ...]
        - gameActors
            - Object mapping id to actor information (could be city, unit, etc.)
            - E.g. {
                    '3': {'level': 2,
                          'population': 2,
                          'population_need': 3,
                          ...
                        },
                    '7': {'ATK': 2,
                          'DEF': 2,
                          'MOV': 1,
                          ...
                        },
                    }
        - size
            - Size of the board; E.g. 11
        - activeTribeID
            - ID of the tribe that is currently making a move; E.g. 1
        - actorIDcounter
            - Counter for the actor ID (incremented by 1 for each new actor); E.g. 14
        - tradeNetwork
            - Object mapping network tiles to bools; E.g. {'networkTiles': [[False,...], ...]
            - TODO: I'm not sure what this is, but it's in the game state
        - diplomacy
            - Allegiance status of each tribe to each other tribe; E.g. {'allegianceStatus': [[0, -37], [-37, 0]]}
        - isNative
            - Boolean indicating if the game state is native (not a copy); E.g. False
    - cityActions
        - Actions for each city; E.g. {'1': [{'cityId': 1,
                                              'targetPos': {'x': 2, 'y': 3},
                                              'actionType': 'DESTROY'},
                                              ...
                                             ]},
    - unitActions
        - Actions for each unit; E.g. {'9': [{'unitId': 9, 'actionType': 'DISBAND'},
                                             {'destination': {'x': 1, 'y': 3},
                                              'unitId': 9,
                                              'actionType': 'MOVE'},
                                              ...
                                             ]
                                      },
    - tribeActions
        - Actions for each tribe; E.g. [{'targetID': 1, 'tribeId': 0, 'actionType': 'DECLARE_WAR'},
                                {'numStars': 1,
                                 'targetID': 1,
                                 'tribeId': 0,
                                 'actionType': 'SEND_STARS'},
                                {'tribeId': 0, 'actionType': 'END_TURN'}]}}
    - ranking
        - Tribe rank information (some might be hidden due to partial information)
        - E.g. [{'result': 'INCOMPLETE',
                'id': 1,
                'score': 949,
                'numTechsResearched': 3,
                'numCities': 1,
                'production': 2,
                'numWars': 3,
                'numStars': 4}, ...]
    - gameMode
        - Mode of the game; E.g. 'CAPITALS'
    - tick
        - Current tick of the game; E.g. 13
    - gameIsOver
        - Boolean indicating if the game is over; E.g. False



    VECTORIZATION OF RELEVANT FEATURES (may need to normalize if training is unstable)

    Assume that the board size is n
    - board
        - terrain -> n x n x 1 (int ranging from 0 to 7 for terrain type)
        - resources -> n x n x 1 (int ranging from -1 to 7 for resource types including None)
        - buildings -> n x n x 1 (int ranging from -1 to 18 for building types including None)
        - units -> n x n x 13 for the 13 attributes of a unit
            - ATK -> int, no specific range, but typically between 0 to 5
            - DEF -> int, no specific range, but typically between 0 to 5
            - MOV -> int, no specific range, but typically between 0 to 3
            - RANGE -> int, no specific range, but typically between 1 to 3
            - maxHP -> int, no specific range, but typically between 1 to 40
            - currentHP -> int, no specific range, but typically between 1 to 40
            - kills -> int, no specific range, but typically between 0 to 5
            - isVeteran -> bool
            - cityPositionX -> int, x coordinate of the city the unit belongs to
            - cityPositionY -> int, y coordinate of the city the unit belongs to
            - status -> int, ranging from 0 to 5 for the status of the unit (fresh, moved, attacked, ...)
            - tribeId -> int, id of the tribe the unit belongs to
            - baseLandUnit -> int, -1 to 11 for the base land unit type if the unit is a water unit, otherwise -1
            - IF THERE IS NO UNIT, ALL VALUES ARE -1
            - TODO: If we choose this, then action space should probably not use ID (use position instead?) since it may be an extra confusing step to learn a mapping between units at random sqaures to their ID. Otherwise, I'd have to define like board size^2 * 4 * 14 for the state space, but that's an arbitrary limit
        - cities -> n x n x 10 for the 10 attributes of a city
            - level -> int, no specific range, but typically between 1 to 10
            - population -> int, no specific range, but typically between 0 to 10
            - populationNeed -> int, no specific range, but typically between 1 to 10
            - isCapital -> bool
            - production -> int, no specific range, but typically between 0 to 12
            - hasWalls -> bool
            - bound -> int, between 1 to 2 for the extension of the ccity
            - pointsWorth -> TODO: not included for now
            - tribeId -> int, id of the tribe the city belongs to
            - positionX -> int, x coordinate of the center of the city
            - positionY -> int, y coordinate of the center of the city
            - IF THERE IS NO CITY, ALL VALUES ARE -1
            - TODO: Similar to units, we could use position instead of ID
        - tribes
            - TODO: lots of information here, will do this later
        - activeTribeID -> int, id of the tribe that is currently making a move
        - tradeNetwork -> n x n x 1 (1 bool for each tile)
        - diplomacy -> 1 (float, ranges between -60 to 60, assumes only one other player)
    - ranking
        - TODO: lots of information here, will do this later (is this a duplicate of tribes in board?)

    For the final tensor shape, we can do:
    - CNN Approach for Actor network (keeps spatial information + uses less total parameters in network)
        -  Create a tensor of shape (n, n, 27)
            -  1 for terrain
            -  1 for resources
            -  1 for buildings
            -  13 for unit attributes
            -  10 for city attributes
            -  1 for trade network
        - We also add global information (diplomacy, activeTribeID) as a separate vector
        - Then combine features learned from both networks to a single vector and feed forward to the actor network
    - Regular NN where we flatten everything -> bit simpler implementation, but loses performance
    
    """
    board_size = len(gs['board']['terrains'])
    
    # Initialize arrays with appropriate shapes
    terrain_array = torch.tensor([[[TERRAIN_MAP.get(cell, 0)] for cell in row] 
                            for row in gs['board']['terrains']], 
                           dtype=torch.int32, 
                           device=device)
    resource_array = torch.tensor([[[RESOURCE_MAP.get(cell, -1) if cell is not None else -1] 
                              for cell in row]
                             for row in gs['board']['resources']], 
                            dtype=torch.int32, 
                            device=device)
    building_array = torch.tensor([[[BUILDING_MAP.get(cell, -1) if cell is not None else -1] 
                              for cell in row]
                             for row in gs['board']['buildings']], 
                            dtype=torch.int32, 
                            device=device)
    trade_network = (torch.tensor(gs['board']['tradeNetwork']['networkTiles'], 
                            dtype=torch.int32, 
                            device=device).reshape(board_size, board_size, 1) 
                 if 'tradeNetwork' in gs['board'] 
                 else torch.zeros((board_size, board_size, 1), 
                                dtype=torch.int32, 
                                device=device))

    # Create a dictionary mapping unit_id to its attributes
    game_actors = gs['board']['gameActors']
    unit_data = {
        int(uid): {
            'ATK': unit.get('ATK', -1),
            'DEF': unit.get('DEF', -1),
            'MOV': unit.get('MOV', -1),
            'RANGE': unit.get('RANGE', -1),
            'maxHP': unit.get('maxHP', -1),
            'currentHP': unit.get('currentHP', -1),
            'kills': unit.get('kills', -1),
            'isVeteran': 1 if unit.get('isVeteran', False) else 0,
            'city_pos_x': game_actors[str(unit['cityId'])]['position'].get('x', -1) if 'cityId' in unit and str(unit['cityId']) in game_actors else -1,
            'city_pos_y': game_actors[str(unit['cityId'])]['position'].get('y', -1) if 'cityId' in unit and str(unit['cityId']) in game_actors else -1,
            'status': UNIT_STATUS_MAP.get(unit.get('status', 'FRESH'), 0),
            'tribeId': unit.get('tribeId', -1),
            'baseLandUnit': UNIT_TYPE_MAP.get(unit.get('baseLandUnit'), -1) if unit.get('baseLandUnit') is not None else -1
        }
        for uid, unit in gs['board']['gameActors'].items()
        if isinstance(uid, str) and 'ATK' in unit
    }

    # Create the unit array in one go
    unit_array = torch.tensor([
        [
            list(unit_data.get(unit_id, {
                'ATK': -1, 'DEF': -1, 'MOV': -1, 'RANGE': -1,
                'maxHP': -1, 'currentHP': -1, 'kills': -1,
                'isVeteran': 0, 'city_pos_x': -1, 'city_pos_y': -1,
                'status': 0, 'tribeId': -1, 'baseLandUnit': -1
            }).values())
            for unit_id in row
        ]
        for row in gs['board']['units']
    ], dtype=torch.int32, device=device)

    # Create a dictionary mapping city_id to its attributes
    city_data = {
        int(cid): {
            'level': city.get('level', -1),
            'population': city.get('population', -1),
            'population_need': city.get('population_need', -1),
            'is_capital': 1 if city.get('isCapital', False) else 0,
            'production': city.get('production', -1),
            'has_walls': 1 if city.get('hasWalls', False) else 0,
            'bound': city.get('bound', -1),
            'tribe_id': city.get('tribeId', -1),
            'pos_x': city.get('position', {}).get('x', -1),
            'pos_y': city.get('position', {}).get('y', -1)
        }
        for cid, city in gs['board']['gameActors'].items()
        if isinstance(cid, str) and 'level' in city
    }

    # Create the city array in one go
    board_size = len(gs['board']['terrains'])
    tile_city_ids = gs['board']['tileCityId']
    
    city_array = torch.tensor([
        [
            list(city_data.get(tile_city_ids[y][x], {
                'level': -1, 'population': -1, 'population_need': -1,
                'is_capital': 0, 'production': -1, 'has_walls': 0,
                'bound': -1, 'tribe_id': -1, 'pos_x': -1, 'pos_y': -1
            }).values())
            for x in range(board_size)
        ]
        for y in range(board_size)
    ], dtype=torch.int32, device=device)

    # Combine all arrays into final tensor
    spatial_tensor = torch.cat([
        terrain_array,
        resource_array,
        building_array,
        unit_array,
        city_array,
        trade_network
    ], dim=2).float().to(device)

    # Get global information
    global_info = torch.tensor([
        gs['board']['activeTribeID'],
        gs['board']['diplomacy']['allegianceStatus'][0][1 - gs['board']['activeTribeID']]  # Assuming 2-player game
    ], dtype=torch.float32).to(device)

    return spatial_tensor, global_info


# Testing the vectorization
if __name__ == "__main__":
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load the sample game state
    with open('ppo_clip/sample_game_states/game_state_20241212_122822.json', 'r') as f:
        game_state = json.load(f)

    # Convert game state to vector
    spatial_tensor, global_info = game_state_to_vector(game_state, DEVICE)

    # Print some basic information about the outputs
    print(f"Spatial tensor shape: {spatial_tensor.shape}")  # Should be (11, 11, 27)
    print(f"Global info shape: {global_info.shape}")       # Should be (2,)

    # Let's check some specific values to verify correctness
    print("\nSample checks:")

    # Print the entire terrain
    print(f"Terrain: {spatial_tensor[:,:,0]}")

    # Print the entire resource
    print(f"Resource: {spatial_tensor[:,:,1]}")

    # Print the entire building
    print(f"Buildings: {spatial_tensor[:,:,2]}")

    # Print out some information about the units
    print(f"Units: {spatial_tensor[:,:,15]}")

    # Print some information about the city
    print(f"City: {spatial_tensor[:,:,25]}")

    # Print information about the trade network
    print(f"Trade Network: {spatial_tensor[:,:,26]}")

    # Check global information
    print(f"\nGlobal info:")
    print(f"Active Tribe ID: {global_info[0]}")  # Should be 1
    print(f"Diplomacy Status: {global_info[1]}")  # Should be 0

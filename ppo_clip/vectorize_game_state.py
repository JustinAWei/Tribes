import numpy as np

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

def game_state_to_vector(gs):
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
        - resources -> n x n x 1 (int ranging from 0 to 8 for resource types including None)
        - buildings -> n x n x 1 (int ranging from 0 to 19 for building types including None)
        - units -> n x n x 12 for the 12 attributes of a unit
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
            - IF THERE IS NO UNIT, ALL VALUES ARE -1
            - TODO: If we choose this, then action space should probably not use ID (use position instead?) since it may be an extra confusing step to learn a mapping between units at random sqaures to their ID. Otherwise, I'd have to define like board size^2 * 4 * 14 for the state space, but that's an arbitrary limit
        - cities -> n x n x 8 for the 8 attributes of a city
            - level -> int, no specific range, but typically between 1 to 10
            - population -> int, no specific range, but typically between 0 to 10
            - populationNeed -> int, no specific range, but typically between 1 to 10
            - isCapital -> bool
            - production -> int, no specific range, but typically between 0 to 12
            - hasWalls -> bool
            - bound -> int, between 1 to 2 for the extension of the ccity
            - pointsWorth -> TODO: not included for now
            - tribeId -> int, id of the tribe the city belongs to
            - IF THERE IS NO CITY, ALL VALUES ARE -1
            - TODO: Similar to units, we could use position instead of ID
        - tribes
            - TODO: lots of information here, will do this later
        - tileCityId -> n x n x 2
            - positionX -> int, x coordinate of the city that owns the tile
            - positionY -> int, y coordinate of the city that owns the tile
            - IF THERE IS NO CITY, ALL VALUES ARE -1
            - TODO: Not confirmed, but if it's 0, I believe that means it's in the fog, so values will be -2
        - activeTribeID -> int, id of the tribe that is currently making a move
        - tradeNetwork -> n x n x 1 (1 bool for each tile)
        - diplomacy -> 1 (float, ranges between -60 to 60, assumes only one other player)
    - ranking
        - TODO: lots of information here, will do this later (is this a duplicate of tribes in board?)

    For the final tensor shape, we can do:
    - CNN Approach for Actor network (keeps spatial information + uses less total parameters in network)
        -  Create a tensor of shape (n, n, 26)
            -  1 for terrain
            -  1 for resources
            -  1 for buildings
            -  12 for unit attributes
            -  8 for city attributes
            -  2 for tile ownership
            -  1 for trade network
        - We also add global information (diplomacy, activeTribeID) as a separate vector
        - Then combine features learned from both networks to a single vector and feed forward to the actor network
    - Regular NN where we flatten everything -> bit simpler implementation, but loses performance
    
    """
    board_size = len(gs['board']['terrains'])
    
    # Initialize arrays with appropriate shapes
    terrain_array = np.zeros((board_size, board_size, 1), dtype=np.int32)
    resource_array = np.zeros((board_size, board_size, 1), dtype=np.int32)
    building_array = np.zeros((board_size, board_size, 1), dtype=np.int32)
    unit_array = np.full((board_size, board_size, 12), -1, dtype=np.int32)  # -1 for no unit
    city_array = np.full((board_size, board_size, 8), -1, dtype=np.int32)   # -1 for no city
    tile_ownership = np.full((board_size, board_size, 2), -1, dtype=np.int32)
    trade_network = np.zeros((board_size, board_size, 1), dtype=np.int32)

    # Fill terrain array
    for y, row in enumerate(gs['board']['terrains']):
        for x, cell in enumerate(row):
            terrain_array[y, x, 0] = TERRAIN_MAP.get(cell, 0)

    # Fill resource array
    for y, row in enumerate(gs['board']['resources']):
        for x, cell in enumerate(row):
            if cell is not None:
                resource_array[y, x, 0] = RESOURCE_MAP.get(cell, 0)

    # Fill building array
    for y, row in enumerate(gs['board']['buildings']):
        for x, cell in enumerate(row):
            if cell is not None:
                building_array[y, x, 0] = BUILDING_MAP.get(cell, 0)

    # Fill unit array
    game_actors = gs['board']['gameActors']
    for y, row in enumerate(gs['board']['units']):
        for x, unit_id in enumerate(row):
            if unit_id is not None and str(unit_id) in game_actors:
                unit = game_actors[str(unit_id)]
                if 'ATK' in unit:  # Check if it's actually a unit
                    unit_array[y, x] = [
                        unit.get('ATK', 0),
                        unit.get('DEF', 0),
                        unit.get('MOV', 0),
                        unit.get('RANGE', 1),
                        unit.get('maxHP', 0),
                        unit.get('currentHP', 0),
                        unit.get('kills', 0),
                        1 if unit.get('isVeteran', False) else 0,
                        unit.get('cityPositionX', -1),
                        unit.get('cityPositionY', -1),
                        unit.get('status', 0),
                        unit.get('tribeId', -1)
                    ]

    # Fill city array and tile ownership
    for y, row in enumerate(gs['board']['tileCityId']):
        for x, city_id in enumerate(row):
            if city_id == 0:  # Fog of war
                tile_ownership[y, x] = [-2, -2]
            elif city_id > 0 and str(city_id) in game_actors:
                city = game_actors[str(city_id)]
                if 'level' in city:  # Check if it's actually a city
                    city_array[y, x] = [
                        city.get('level', 1),
                        city.get('population', 0),
                        city.get('populationNeed', 1),
                        1 if city.get('isCapital', False) else 0,
                        city.get('production', 0),
                        1 if city.get('hasWalls', False) else 0,
                        city.get('bound', 1),
                        city.get('tribeId', -1)
                    ]
                    # Store city position for tile ownership
                    city_pos = city.get('position', {})
                    tile_ownership[y, x] = [
                        city_pos.get('x', -1),
                        city_pos.get('y', -1)
                    ]

    # Fill trade network
    if 'tradeNetwork' in gs['board']:
        trade_network = np.array(gs['board']['tradeNetwork']['networkTiles']).reshape(board_size, board_size, 1)

    # Combine all arrays into final tensor
    spatial_tensor = np.concatenate([
        terrain_array,
        resource_array,
        building_array,
        unit_array,
        city_array,
        tile_ownership,
        trade_network
    ], axis=2)

    # Get global information
    global_info = np.array([
        gs['board']['activeTribeID'],
        gs['board']['diplomacy']['allegianceStatus'][0][1]  # Assuming 2-player game
    ])

    return spatial_tensor, global_info
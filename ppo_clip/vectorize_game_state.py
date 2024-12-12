import numpy as np

def game_state_to_vector(game_state):
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
        - buildings -> n x n x 1 (int ranging from 0 to 20 for building types including None)
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
    # # Initialize vector components (adjust these based on your game state structure)
    # vector = []
    # 
    # # Example features (modify these according to your actual game state):
    # # Player position
    # vector.extend([
    #     game_state.get('player', {}).get('x', 0) / 1000,  # Normalize position
    #     game_state.get('player', {}).get('y', 0) / 1000,
    # ])
    # 
    # # Player velocity/direction
    # vector.extend([
    #     game_state.get('player', {}).get('velocityX', 0) / 10,
    #     game_state.get('player', {}).get('velocityY', 0) / 10,
    # ])
    # 
    # # Other game state features...
    # 
    # return np.array(vector, dtype=np.float32)
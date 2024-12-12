import numpy as np

def game_state_to_vector(game_state):
    """
    Vectorize the game state into a numpy array. Below is the structure of the game state (and below that is the vectorization of the relevant features since not all features are relevant to the game for now).

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
            - This is not confirmed, but if it's 0, I believe that means it's in the fog
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
import torch
import joblib
import time
import pprint

MASK = 0

BOARD_LEN = 11
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_actor_x_y(actor_id, gs):
    # in board -> gameActors
    game_actors = gs.get('board', {}).get('gameActors', {})
    actor = game_actors.get(str(actor_id), {})
    position = actor.get('position', {})
    x = position.get('x', None)
    y = position.get('y', None)

    if x is None or y is None:
        pprint(gs)
        print(f"ACTOR {actor_id} DOESNT EXIST. RETURNING {MASK}, {MASK}")
        return MASK, MASK

    # print("actor_id", actor_id)
    # print(f"x: {x} | y: {y}")

    return x, y


# Reversed dictionary for action types
ACTION_TYPES = {
    "BUILD": 0,
    "LEVEL_UP": 5,
    "RESOURCE_GATHERING": 6,
    "SPAWN": 7,

    "BUILD_ROAD": 8,
    "END_TURN": 9,
    "RESEARCH_TECH": 10,


    "ATTACK": 13,
    "CAPTURE": 14,
    "DISBAND": 16,
    "EXAMINE": 17,
    "MAKE_VETERAN": 19,
    "MOVE": 20,
    "RECOVER": 21,

    "UPGRADE_BOAT": 23,
    "UPGRADE_SHIP": 24,
}

# When Action Type is LEVEL_UP
BONUS_TYPES = {
    "WORKSHOP": 0,
    "EXPLORER": 1,
    "CITY_WALL": 2,
    "RESOURCES": 3,
    "POP_GROWTH": 4,
    "BORDER_GROWTH": 5,
    "PARK": 6,
    "SUPERUNIT": 7
}

# When Action Type is SPAWN
UNIT_TYPES = {
    "WARRIOR": 0,
    "RIDER": 1, 
    "DEFENDER": 2,
    "SWORDMAN": 3,
    "ARCHER": 4,
    "CATAPULT": 5,
    "KNIGHT": 6,
    "MIND_BENDER": 7,
    "BOAT": 8,
    "SHIP": 9,
    "BATTLESHIP": 10,
    "SUPERUNIT": 11
}

# When Action Type is BUILD
BUILDING_TYPES = {
    "PORT": 0,
    "MINE": 1,
    "FORGE": 2,
    "FARM": 3,
    "WINDMILL": 4,
    "CUSTOMS_HOUSE": 5,
    "LUMBER_HUT": 6,
    "SAWMILL": 7,
    "TEMPLE": 8,
    "WATER_TEMPLE": 9,
    "FOREST_TEMPLE": 10,
    "MOUNTAIN_TEMPLE": 11,
    "ALTAR_OF_PEACE": 12,
    "EMPERORS_TOMB": 13,
    "EYE_OF_GOD": 14,
    "GATE_OF_POWER": 15,
    "GRAND_BAZAR": 16,
    "PARK_OF_FORTUNE": 17,
    "TOWER_OF_WISDOM": 18
}

# When Action Type is RESEARCH_TECH
TECH_TYPES = {
    "CLIMBING": 0,
    "FISHING": 1,
    "HUNTING": 2,
    "ORGANIZATION": 3,
    "RIDING": 4,
    "ARCHERY": 5,
    "FARMING": 6,
    "FORESTRY": 7,
    "FREE_SPIRIT": 8,
    "MEDITATION": 9,
    "MINING": 10,
    "ROADS": 11,
    "SAILING": 12,
    "SHIELDS": 13,
    "WHALING": 14,
    "AQUATISM": 15,
    "CHIVALRY": 16,
    "CONSTRUCTION": 17,
    "MATHEMATICS": 18,
    "NAVIGATION": 19,
    "SMITHERY": 20,
    "SPIRITUALISM": 21,
    "TRADE": 22,
    "PHILOSOPHY": 23
}

# I want to map (Source X, Source Y, Dest X, Dest Y, Action Type, [Tech, Bonus, Building, Unit]) to a single index
action_tuples = []

# Add all action types first
for action_type in ACTION_TYPES.keys():

    # which ones need an x,y?
    # any: build, resource, build road, attack, move, 
    if action_type == 'BUILD':
        for building_type in BUILDING_TYPES.values():
            for x in range(BOARD_LEN):
                for y in range(BOARD_LEN):
                    action_tuples.append((x, y, ACTION_TYPES[action_type], building_type))

    elif action_type == 'RESOURCE_GATHERING':
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                action_tuples.append((x, y, ACTION_TYPES[action_type], MASK))

    elif action_type == 'BUILD_ROAD':
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                action_tuples.append((x, y, ACTION_TYPES[action_type], MASK))

    elif action_type == 'MOVE':
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                action_tuples.append((x, y, ACTION_TYPES[action_type], MASK))

    elif action_type == 'ATTACK':
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                action_tuples.append((x, y, ACTION_TYPES[action_type], MASK))

    # self unit actions 
    # self: spawn, level_up, capture, recover, examine, make_veteran, upgrade_boat, upgrade_ship
    elif action_type == 'CAPTURE':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))
    elif action_type == 'RECOVER':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))
    elif action_type == 'EXAMINE':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))
    elif action_type == 'MAKE_VETERAN':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))
    elif action_type == 'UPGRADE_BOAT':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))
    elif action_type == 'UPGRADE_SHIP':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))
    elif action_type == 'DISBAND':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))

    # self tribe actions
    elif action_type == 'RESEARCH_TECH':
        for tech_type in TECH_TYPES.values():
            action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], tech_type))

    # self city actions
    elif action_type == 'LEVEL_UP':
        for bonus_type in BONUS_TYPES.values():
            action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], bonus_type))

    elif action_type == 'SPAWN':
        for unit_type in UNIT_TYPES.values():
            action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], unit_type))

    # end turn
    elif action_type == 'END_TURN':
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))

    else:
        print("not explicitly handled")
        print("action_type", action_type)
        action_tuples.append((MASK, MASK, ACTION_TYPES[action_type], MASK))


# be able to convert from (action_type, extra_var) to index
def action_tuple_to_index(x,y, action_type, extra_var):
    return action_tuples.index((x,y, action_type, extra_var))

def index_to_action_tuple(index):
    return action_tuples[index]

MAX_EXTRA_VARS = max(len(ACTION_TYPES.values()), len(BONUS_TYPES.values()), len(TECH_TYPES.values()), len(BUILDING_TYPES.values()), len(UNIT_TYPES.values()))

def game_over(gs):
    return gs['gameIsOver']

def reward_fn(gs, active_tribe_id):
    '''
    Ranking is a list of tribes scored by WIN, score, tech, cities, production, wars, stars

        [{'id': 1,
            'numCities': 1,
            'numStars': 0,
            'numTechsResearched': 24,
            'numWars': 0,
            'production': 7,
            'result': 'INCOMPLETE',
            'score': 8025},
            {'id': 0,
            'numCities': 1,
            'numStars': 0,
            'numTechsResearched': 18,
            'numWars': 0,
            'production': 7,
            'result': 'INCOMPLETE',
            'score': 4515}],
    '''

    reward = 0

    # check if win by capitals, meaning we captured all the capitals
    rankings = gs['ranking']

    total_tribes = len(rankings)
    tribes_with_no_cities = 0
    for ranking in rankings:
        if ranking['numCities'] == 0:
            tribes_with_no_cities += 1
    win_by_capitals = tribes_with_no_cities == total_tribes - 1

    if win_by_capitals:
        for tribe in gs['board']['tribes']:
            if tribe['actorId'] == active_tribe_id:
                reward = [1 if tribe['winner'] == "WIN" else -1]
                break

    else:
        for tribe in rankings:
            if tribe['id'] == active_tribe_id:
                total_possible_counted_cities = max(1, BOARD_LEN * BOARD_LEN / 4)
                city_reward_pct = min(tribe['numCities'] / total_possible_counted_cities, 1)

                total_possible_counted_production = 150
                production_reward_pct = min(tribe['production'] / total_possible_counted_production, 1)

                # This can only be max .1, because the real reward is 1 for winning
                reward = .05 * city_reward_pct + .05 * production_reward_pct
                break

    return reward


def filter_actions(gs):
    valid_actions = []

    # Allowed action types (both numbers and strings)
    # Building types, unit types, tech types, etc.
    allowed_action_types = list(ACTION_TYPES.keys()) + list(ACTION_TYPES.values())

    # this is just a list
    tribe_actions = gs.get('tribeActions', [])
    # print("tribe_actions")
    # print(tribe_actions)
    filtered_tribe_actions = [
        action for action in tribe_actions
        if action.get('actionType') in allowed_action_types
    ]
    # print("filtered_tribe_actions")
    # print(filtered_tribe_actions)
    for action in filtered_tribe_actions:
        if action.get('actionType') == 'RESEARCH_TECH':
            tech_type = TECH_TYPES[action.get('tech')]
            valid_actions.append([MASK, MASK, action_tuple_to_index(MASK, MASK, ACTION_TYPES[action.get('actionType')], tech_type)])
        elif action.get('actionType') == 'BUILD_ROAD':
            # get position
            x, y = action.get('position').get('x'), action.get('position').get('y')
            valid_actions.append([x, y,  action_tuple_to_index(MASK, MASK, ACTION_TYPES[action.get('actionType')], MASK)])
        else:
            valid_actions.append([MASK, MASK, action_tuple_to_index(MASK, MASK, ACTION_TYPES[action.get('actionType')], MASK)])

    # Process city actions
    # List to store filtered city actions
    filtered_city_actions = []
    city_actions = gs.get('cityActions', {})
    # print("city_actions")
    # print(city_actions)
    for city_id, actions in city_actions.items():
        # Filter actions for this city
        filtered_city_actions = [
            action for action in actions
            if action.get('actionType') in allowed_action_types
        ]
        # print('city_id', 'filtered_city_actions')
        # print(city_id, filtered_city_actions)
        for action in filtered_city_actions:
            x2, y2 = MASK, MASK
            if 'targetPos' in action:
                x2 = action['targetPos']['x']
                y2 = action['targetPos']['y']

            x1, y1 = get_actor_x_y(int(city_id), gs)

            if action.get('actionType') == 'BUILD':
                building_type = BUILDING_TYPES[action.get('buildingType')]
                valid_actions.append([x1, y1, action_tuple_to_index(x2, y2, ACTION_TYPES[action.get('actionType')], building_type)])
            elif action.get('actionType') == 'SPAWN':
                unit_type = UNIT_TYPES[action.get('unit_type')]
                # Assumption: Spawn is always to the same position as the city
                valid_actions.append([x1, y1, action_tuple_to_index(MASK, MASK, ACTION_TYPES[action.get('actionType')], unit_type)])
            elif action.get('actionType') == 'LEVEL_UP':
                bonus_type = BONUS_TYPES[action.get('bonus')]
                valid_actions.append([x1, y1, action_tuple_to_index(MASK, MASK, ACTION_TYPES[action.get('actionType')], bonus_type)])
            else:
                valid_actions.append([x1, y1, action_tuple_to_index(x2, y2, ACTION_TYPES[action.get('actionType')], MASK)])

    unit_actions = gs.get('unitActions', {})
    # print("unit_actions")
    # print(unit_actions)
    for unit_id, actions in unit_actions.items():
        # Filter actions for this city
        filtered_unit_actions = [
            action for action in actions
            if action.get('actionType') in allowed_action_types
        ]
        # print('unit_id', 'filtered_unit_actions')
        # print(unit_id, filtered_unit_actions)
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

            elif action.get('actionType') in ['CAPTURE', 'RECOVER', 'EXAMINE', 'MAKE_VETERAN', 'DISBAND', 'UPGRADE_BOAT', 'UPGRADE_SHIP']:
                # Assumption: These are always to the same position as the unit
                x2, y2 = MASK, MASK

            valid_actions.append([x1, y1, action_tuple_to_index(x2, y2, ACTION_TYPES[action.get('actionType')], MASK)])

    # ensure its a tensor
    return valid_actions

def serialize_trajectories(trajectories):
    # Convert each tensor in the trajectories to a numpy array
    numpy_trajectories = {
        key: [tensor.detach().cpu().numpy() for tensor in tensors]
        for key, tensors in trajectories.items()
    }
    
    # Serialize the numpy arrays using joblib
    serialized_data = joblib.dumps(numpy_trajectories, compress=('zlib', 3))
    
    return serialized_data
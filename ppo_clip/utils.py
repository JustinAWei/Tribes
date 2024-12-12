
def get_actor_x_y(actor_id, gs):
    # in board -> gameActors
    game_actors = gs.get('board', {}).get('gameActors', {})
    actor = game_actors.get(str(actor_id), {})
    if not actor:
        return MASK, MASK

    position = actor.get('position', {})
    x = position.get('x', 0)
    y = position.get('y', 0)

    # print("actor_id", actor_id)
    # print(f"x: {x} | y: {y}")

    return x, y

# Reversed dictionary for action types
ACTION_TYPES = {
    "BUILD": 0,
    "LEVEL_UP": 5,
    "RESOURCE_GATHERING": 6,
    "SPAWN": 7,

    "END_TURN": 9,
    "RESEARCH_TECH": 10,

    "ATTACK": 13,
    "CAPTURE": 14,
    "MOVE": 20
}

# Reversed dictionary for action categories
ACTION_CATEGORIES = {
    "TRIBE": 0,
    "CITY": 1,
    "UNIT": 2
}


# Unit type mapping
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

# Building type mapping
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
# Technology type mapping
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


MASK = 0
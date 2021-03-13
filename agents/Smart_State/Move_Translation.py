# Helper file to translate cardinal directions to actual moves in Everglades

move_translator_left = {
    1: 1,
    2: 1,
    3: 3,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 11
}

move_translator_right = {
    1: 1,
    2: 5,
    3: 6,
    4: 7,
    5: 8,
    6: 9,
    7: 10,
    8: 11,
    9: 9,
    10: 11,
    11: 11
}

move_translator_up = {
    1: 2,
    2: 2,
    3: 2,
    4: 3,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 8,
    10: 9,
    11: 8
}

move_translator_down = {
    1: 4,
    2: 3,
    3: 4,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 9,
    9: 10,
    10: 10,
    11: 10
}

move_translator_stay = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11
}

move_translator_total = {
    0: move_translator_left,
    1: move_translator_right,
    2: move_translator_up,
    3: move_translator_down,
    4: move_translator_stay,
}

# Helper function to determine where a swarm should move given
# its current location and the direction it should move to
#
# @param 
def get_move(node_location_0_indexed, direction):
    """
    Helper function to determine where a swarm should move given
    its current location and the direction it should move.

    @param node_location_0_indexed The location the swarm is currently
        at using a 0-indexed system
    @param direction The direction the swarm should move as an integer
        between 0 and 3, each representing a different cardinal direction
    @returns The node the swarm should move to using a 1-indexed
        system
    """
    return move_translator_total[int(direction)][int(node_location_0_indexed) + 1]
        
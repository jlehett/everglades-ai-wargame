def penalize_long_games(reward0, reward1, done):
    """
    A reward system that penalizes the agent the longer the game goes on.
    """
    if done:
        if reward0 > reward1:
            return 100.0
        else:
            return -0.1
    return -0.001

def basic_reward(reward0, reward1, done):
    """
    Basic reward system that only gives a reward of 1.0 when the agent wins.
    """
    if done:
        if reward0 > reward1:
            return 1.0
    return 0.0
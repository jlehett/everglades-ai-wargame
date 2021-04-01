class RewardShaping():
    """
    The Reward Shaper Class
    Enables the use of multiple static reward shaping functions as well as dynamic reward shaping functions
    """
    def __init__(self):
        """
        Do nothing (for now)
        """
        pass

    def penalize_long_games(self, rewardArray, done, turnNum):
        """
        A reward system that penalizes the agent the longer the game goes on.
        """
        if done:
            if rewardArray[0] > rewardArray[1]:
                return 100.0
            else:
                return -0.1
        return -0.001

    def basic_reward(self, rewardArray, done, turnNum):
        """
        Basic reward system that only gives a reward of 1.0 when the agent wins.
        """
        if done:
            if rewardArray[0] > rewardArray[1]:
                return 1.0
        return 0.0

    def reward_short_games(self, rewardArray, done, turnNum):
        """
        Similar to penalize_long_games, but instead provides larger rewards for
        finishing earlier.
        """
        if done:
            if rewardArray[0] > rewardArray[1]:
                return (150.0 - turnNum) / 150.0
            else:
                return -1.0
        return 0.0
def transition(reward_shaping_fn1, reward_shaping_fn2, fully_transitioned_episode_num,
               game_num, player_num, rewardArray, done, turnNum):
    """
    A reward system that transitions between 2 different reward shaping functions.
    @param {Function} reward_shaping_fn1 The starting reward shaping function
    @param {Function} reward_shaping_fn2 The ending reward shaping function
    @param {int} fully_transitioned_episode_num The episode number at which the system has fully
        transitioned to the second reward shaping function
    @param {int} game_num The current game number
    """
    ratio = min(1.0, game_num / fully_transitioned_episode_num)
    reward1 = reward_shaping_fn1(player_num, rewardArray, done, turnNum) * (1.0 - ratio)
    reward2 = reward_shaping_fn2(player_num, rewardArray, done, turnNum) * ratio
    return reward1 + reward2

def penalize_long_games(player_num, rewardArray, done, turnNum):
    """
    A reward system that penalizes the agent the longer the game goes on.
    """
    opposing_player_num = int(1 - player_num)
    if done:
        if rewardArray[player_num] > rewardArray[opposing_player_num]:
            return 100.0
        else:
            return -0.1
    return -0.001

def basic_reward(player_num, rewardArray, done, turnNum):
    """
    Basic reward system that only gives a reward of 1.0 when the agent wins.
    """
    opposing_player_num = int(1 - player_num)
    if done:
        if rewardArray[player_num] > rewardArray[opposing_player_num]:
            return 1.0
    return 0.0

def reward_short_games(player_num, rewardArray, done, turnNum):
    """
    Similar to penalize_long_games, but instead provides larger rewards for
    finishing earlier.
    """
    opposing_player_num = int(1 - player_num)
    if done:
        if rewardArray[player_num] > rewardArray[opposing_player_num]:
            return (150.0 - turnNum) / 150.0
        else:
            return -1.0
    return 0.0

def normalized_score(player_num, rewardArray, done, turnNum):
    """
    Utilize the normalized score for the reward (should already be in the reward
    array).
    """
    return rewardArray[player_num]
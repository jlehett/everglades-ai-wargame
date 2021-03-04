class RewardShaping():
    def __init__(self):
        self.loss_reward_decay = {-2: 0.64} #reward decay for losing
        self.reward_divider = 1000 #value to divide by for normalization
        self.final_score_agent = 0
        self.final_score_opponent = 0

    def get_reward(self, done,reward):
        won = False
        if not done: # Game not finished
            # Gets the final score before end of game turn
            self.final_score_agent = reward[0]
            self.final_score_opponent = reward[1]

            # Set reward for non-game ending turns
            reward[0] = 0
        elif reward[0] < reward[1]: # Agent lost
            # Calculate the normalized score
            reward_sub = (self.final_score_agent - self.final_score_opponent) / self.reward_divider
            found_less = False

            # Check if score is below a requirement and give that reward
            for required_reward in self.loss_reward_decay.keys():
                if reward_sub < required_reward:
                    reward[0] = self.loss_reward_decay.get(required_reward)
                    found_less = True
                    break

            # If not below any requirements give default reward
            if not found_less:
                reward[0] = 0.8
            
            reward[0] = 0 # Override to test on basic reward system (0 for loss 1 for win)
        else: # Agent won
            # Set agent win condition
            won = True
            
            # Reward for winning
            reward[0] = 1

        return won, reward, self.final_score_agent, self.final_score_opponent

    def update_rewards(self, i_episode):
        # Stop updating after 1000 episodes
        if i_episode >= 1000:
            pass

        # Default value to make sure we get the right requirement
        new_requirement = -100

        # Loop over keys and update new reward values for requirement and get most recent requirement (largest value)
        for required_reward in self.loss_reward_decay.keys():
            self.loss_reward_decay[required_reward] *= 0.8
            if required_reward > new_requirement:
                new_requirement = required_reward

        # Create new requirement and store
        new_requirement += 0.2
        self.loss_reward_decay[new_requirement] = 0.64
import numpy as np

class PrioritizedMemory(object):
    """
    Implementation of Prioritized Experience Replay
    """
    def __init__(self, capacity, prob_alpha=0.6):
        """
        Initialize the replay memory

        @param capacity The size of the replay memory
        @param prob_alpha The initial probabilities of the replay
        """
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """
        Stores a memory in the replay

        @param state The state of the game
        @param action The action taken by the agent
        @param reward The reward received by the agent
        @param next_state The next state of the game
        @param done Whether or not the game has ended
        """
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch of memories using priority

        @param batch_size The size of the memory batch
        @param beta The beta value for PER
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        """
        Updates the priority values of the memories

        @param batch_indices The indices of the memories to update
        @param batch_priorities The priority values for the batch indices
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """
        Gets the length of the replay memory
        """
        return len(self.buffer)
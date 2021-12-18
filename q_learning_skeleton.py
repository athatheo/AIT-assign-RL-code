import random
import numpy as np

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500

DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1

# Coding exercise 2 (decaying epsilon)
DECAY_ENABLED = True
EPSILON_MAX = 1
EPSILON_MIN = 0.05
EPSILON_DECAY = 1 - 1/NUM_EPISODES


class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"
        self.Q = np.zeros((num_states, num_actions))
        if DECAY_ENABLED:
            self.epsilon = EPSILON_MAX
        else:
            self.epsilon = EPSILON


    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        if DECAY_ENABLED:
            self.epsilon = EPSILON_DECAY*self.epsilon
            if self.epsilon < 0.01:
                self.epsilon = 0.01
        else:
            pass


    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        if not done:
            self.Q[state, action] = (1 - LEARNINGRATE)*self.Q[state, action] + LEARNINGRATE*(reward + DEFAULT_DISCOUNT*np.max(self.Q[next_state,:]))
        else:
            self.Q[state, action] = (1 - LEARNINGRATE)*self.Q[state, action] + LEARNINGRATE*reward

    def select_action(self, state): 
        """
        Returns an action, selected based on the current state
        """
        epsilon = random.random() # return probability between [0.0, 1.0)
        if epsilon < self.epsilon:
            # Exploration: If probability is smaller than EPSILON, pick one action at random
            return random.randint(0, 3)
        else:
            # Exploitation: If probability is larger than EPSILON, pick the action with the
            # highest q value
            return np.random.choice(np.argwhere(self.Q[state,:] == np.max(self.Q[state, :])).flatten().tolist())

    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")








        

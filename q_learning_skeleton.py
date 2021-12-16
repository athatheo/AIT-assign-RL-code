import random
import numpy as np

NUM_EPISODES = 100000
MAX_EPISODE_LENGTH = 1000


DEFAULT_DISCOUNT = 0.9
EPSILON_MAX = 1
LEARNINGRATE = 0.1
EPSILON_MIN = 0.05
EPSILON = 1
EPSILON_DECAY = 0.9999


class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"
        self.Q = np.zeros((num_states, num_actions))
        self.epsilon = EPSILON_MAX


    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        self.epsilon = EPSILON_DECAY*self.epsilon
        #if self.epsilon < 0.01:
            #self.epsilon = 0.01


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
        epsilon = random.random()
        if epsilon < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.random.choice(np.argwhere(self.Q[state,:] == np.max(self.Q[state, :])).flatten().tolist())



    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")








        

"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        # START EDITING HERE
        # You can add any other variables you need here
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        # choosing factor to accept probability to some extent beyond maximum
        self.exploit = 0.92 + np.random.random() / 20
        self.thres = ((self.num_arms - 1) / self.num_arms) * self.exploit
        # recording the means and the current choice
        self.means = np.ones(self.num_arms) * (self.num_arms - 1) / 2
        self.optimal = np.random.randint(self.num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # check if the current choice is within threshold
        if self.means[self.optimal] >= self.thres:
            return self.optimal
        # choose a new arm randomly
        self.optimal = np.random.randint(self.num_arms)
        return self.optimal
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # update the pulls and means and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward
        self.means[arm_index] = self.rewards[arm_index] / self.pulls[arm_index]
        # END EDITING HERE

"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
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

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need        

# function to calculate empirical means using rewards and pulls
def getEmpMean(arms, rewards, pulls):
    empMean = []
    for i in range(arms):
        if pulls[i] == 0:
            # arm has not yet been pulled
            empMean.append(1e5)
        else:
            # updated empirical mean
            empMean.append(rewards[i] / pulls[i])
    # convert to numpy array
    return np.array(empMean)

# function to calculate extra term in UCB
def getUCBUncert(arms, time, pulls):
    # numerator to find horizon/time factor
    num = math.sqrt(2 * math.log(time))
    uncert = []
    for i in range(arms):
        if pulls[i] == 0:
            # arm has not yet been pulled
            uncert.append(1e5)
        else:
            # updated uncertainty
            uncert.append(num / math.sqrt(pulls[i]))
    # convert to numpy array
    return np.array(uncert)

# function to calculate KL-divergence
def KL(p, q):
    # base case 1
    if p == 0:
        return math.log(1 / (1 - q))
    # base case 2
    if p == 1:
        return math.log(1 / q)
    # full expression
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

# function to binary search for maximum value
def getUCBKLUncert(time, c, p, pulls):
    if (pulls == 0):
        # arm has not yet been pulled
        return (1 + p) / 2
    # upper bound for the divergence
    bound = (math.log(time) + c * math.log(math.log(time))) / pulls
    l = p
    r = 1
    # searching for the largest allowed value
    while r - l > 1e-3:
        q = (l + r) / 2
        # find divergence
        kl = KL(p,q)
        if kl < bound:
            # within bound so move interval ahead
            l = q
        else:
            # out of bound so move interval behind
            r = q
    # found value
    return (l + r) / 2
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_pulls = 0
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.num_pulls += 1
        # get the UCB value for this arm at the given time (modeled by number of pulls)
        ucb = getEmpMean(self.num_arms,self.rewards,self.pulls) + getUCBUncert(self.num_arms,self.num_pulls,self.pulls)
        # return index of the largest/optimal
        return np.argmax(ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # update the pulls and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_pulls = 0
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        self.c = 3
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.num_pulls += 1
        # get the empirical means
        empMean = getEmpMean(self.num_arms,self.rewards,self.pulls)
        ucbkl = []
        for i in range(self.num_arms):
            # iterate over the arms and find the maximum value for ucb-kl for each
            ucbkl.append(getUCBKLUncert(self.num_pulls,self.c,empMean[i],self.pulls[i]))
        # return index of the largest/optimal
        return np.argmax(np.array(ucbkl))
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # update the pulls and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # get the values using beta distribution on each
        thmpsn = np.random.beta(self.rewards + 1,self.pulls - self.rewards + 1)
        # return index of the largest/optimal
        return np.argmax(thmpsn)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # update the pulls and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward
        # END EDITING HERE

# BernoulliArm and BernoulliBandit
# do not modify!

import numpy as np
import matplotlib.pyplot as plt

class BernoulliArm:
  def __init__(self, p):
    self.p = p

  def pull(self, num_pulls=None):
    return np.random.binomial(1, self.p, num_pulls)

class BernoulliBandit:
  def __init__(self, probs=[0.3, 0.5, 0.7], batch_size=1):
    self.__arms = [BernoulliArm(p) for p in probs]
    self.__batch_size = batch_size
    self.__max_p = max(probs)
    self.__regret = 0

  def pull(self, index):
    assert self.__batch_size == 1, "\
    'pull' can't be called for in batched setting, use 'batch_pull' instead"
    reward = self.__arms[index].pull()
    self.__regret += self.__max_p - reward
    return reward

  def batch_pull(self, indices, num_pulls):
    assert sum(num_pulls) == self.__batch_size, "\
    total number of pulls should match batch_size of  %d" % self.__batch_size
    rewards = {}
    for i, np in zip(indices, num_pulls):
      rewards[i] = self.__arms[i].pull(np)
      self.__regret += (self.__max_p * np - rewards[i].sum())
    return rewards

  def regret(self):
    return self.__regret

  def batch_size(self):
    return self.__batch_size

  def num_arms(self):
    return len(self.__arms)

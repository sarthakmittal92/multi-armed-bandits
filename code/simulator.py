# simulator
# do not modify! (except final few lines)

from bernoulli_bandit import *
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task2 import AlgorithmBatched
from task3 import AlgorithmManyArms
from multiprocessing import Pool
import time

def single_sim(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  np.random.seed(seed)
  np.random.shuffle(PROBS)
  bandit = BernoulliBandit(probs=PROBS)
  algo_inst = ALGO(num_arms=len(PROBS), horizon=HORIZON)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()

def single_batch_sim(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000, BATCH_SIZE=1):
  np.random.seed(seed)
  np.random.shuffle(PROBS)
  bandit = BernoulliBandit(probs=PROBS, batch_size=BATCH_SIZE)
  algo_inst = ALGO(num_arms=len(PROBS),
    horizon=HORIZON, batch_size=BATCH_SIZE)
  for t in range(HORIZON//BATCH_SIZE):
    indices, num_pulls = algo_inst.give_pull()
    rewards_dict = bandit.batch_pull(indices, num_pulls)
    algo_inst.get_reward(rewards_dict)
  return bandit.regret()

def simulate(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      regrets = pool.starmap(single_sim,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return regrets

  return np.mean(multiple_sims(num_sims))

def batch_simulate(algorithm, probs, horizon, batch_size, num_sims=50):
  """simulates algorithm of class AlgorithmBatched
  for BernoulliBandit bandit, with horizon=horizon
  """

  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      regrets = pool.starmap(single_batch_sim,
        [(i, algorithm, probs, horizon, batch_size) for i in range(num_sims)])
    return regrets

  return np.mean(multiple_sims(num_sims))

def task1(algorithm, probs, num_sims=50):
  """generates the plots and regrets for task1
  """
  horizons = [2**i for i in range(10, 19)]
  regrets = []
  for horizon in horizons:
    regrets.append(simulate(algorithm, probs, horizon, num_sims))

  print(regrets)
  plt.plot(horizons, regrets)
  plt.title("Regret vs Horizon")
  plt.savefig("task1-{}-{}.png".format(algorithm.__name__, time.strftime("%Y%m%d-%H%M%S")))
  plt.clf()

def task2(algorithm, probs, horizon=10000):
  """generates the plots and regrets for task2
  """
  batch_sizes = [10, 20, 50, 100, 200, 500, 1000]
  regrets = []
  for batch_size in batch_sizes:
    regrets.append(batch_simulate(
      algorithm, probs, horizon, batch_size))

  print(regrets)
  plt.plot(batch_sizes, regrets)
  plt.title("Regret vs Batch Size")
  plt.savefig("task2-{}.png".format(time.strftime("%Y%m%d-%H%M%S")))
  plt.clf()

def task3(algorithm):
  """generates the plots and regrets for task3
  """
  horizons = [1000, 5000, 10000, 15000, 20000, 30000]
  regrets = []
  for horizon in horizons:
    probs = [i/horizon for i in range(horizon)]
    regrets.append(simulate(algorithm, probs, horizon))

  print(regrets)
  plt.plot(horizons, regrets)
  plt.title("Regret vs Horizon=NUM_ARMS")
  plt.savefig("task3-{}.png".format(time.strftime("%Y%m%d-%H%M%S")))
  plt.clf()


if __name__ == '__main__':
  ### EDIT only the following code ###

  # Note - all the plots generated will be for the following bandit instance:
  # 20 arms with uniformly distributed means
  probs = [i/20 for i in range(20)]

#   task1(Eps_Greedy, probs)
#   task1(UCB, probs)
#   task1(KL_UCB, probs)
#   task1(Thompson_Sampling, probs)

#   task2(AlgorithmBatched, probs)

#   task3(AlgorithmManyArms)


import argparse, time
from simulator import simulate, batch_simulate
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task2 import AlgorithmBatched
from task3 import AlgorithmManyArms

FACTOR = 1.5

class Testcase:
    def __init__(self, task, probs, horizon, batch_size):
        self.task = task
        self.probs = probs
        self.horizon = horizon
        self.batch_size = batch_size
        self.ucb = 0
        self.kl_ucb = 0
        self.thompson = 0
        self.other = 0

def read_tc(path):
    tc = None
    with open(path, 'r') as f:
        lines = f.readlines()
        task = int(lines[0].strip())
        horizon = int(lines[1].strip())
        if task == 1:
            probs = [float(p) for p in lines[2].strip().split()]
            ucb, kl_ucb, thompson = [float(x) for x in lines[3].strip().split()]
            tc = Testcase(task, probs, horizon, 1)
            tc.ucb = ucb
            tc.kl_ucb = kl_ucb
            tc.thompson = thompson
        elif task == 2:
            probs = [float(p) for p in lines[2].strip().split()]
            batch_size = int(lines[3].strip())
            reference = float(lines[4].strip())
            tc = Testcase(task, probs, horizon, batch_size)
            tc.other = reference
        elif task == 3:
            probs = [i/horizon for i in range(horizon)]
            reference = float(lines[2].strip())
            tc = Testcase(task, probs, horizon, 1)
            tc.other = reference
            
    return tc

def grade_task1(tc_path, algo):
    algo = algo.lower()
    tc = read_tc(tc_path)
    regrets = {}
    scores = {}
    if algo == 'ucb' or algo == 'all':
        regrets['UCB'] = simulate(UCB, tc.probs, tc.horizon)
        scores['UCB'] = 1 if regrets['UCB'] <= tc.ucb * FACTOR else 0
    if algo == 'kl_ucb' or algo == 'all':
        regrets['KL-UCB'] = simulate(KL_UCB, tc.probs, tc.horizon, num_sims=20)
        scores['KL-UCB'] = 1 if regrets['KL-UCB'] <= tc.kl_ucb * FACTOR else 0
    if algo == 'thompson' or algo == 'all':
        regrets['Thompson Sampling'] = simulate(Thompson_Sampling, tc.probs, tc.horizon)
        scores['Thompson Sampling'] = 1 if regrets['Thompson Sampling'] <= tc.thompson * FACTOR else 0
    
    return scores, regrets

def grade_task2(tc_path):
    tc = read_tc(tc_path)
    regret = batch_simulate(AlgorithmBatched, tc.probs, tc.horizon, tc.batch_size)
    score = 1 if regret <= tc.other * FACTOR else 0
    return score, regret

def grade_task3(tc_path):
    tc = read_tc(tc_path)
    regret = simulate(AlgorithmManyArms, tc.probs, tc.horizon)
    score = 1 if regret <= tc.other * FACTOR else 0
    return score, regret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='The task to run. Valid values are: 1, 2, 3, all')
    parser.add_argument('--algo', type=str, required=False, help='The algo to run (for task 1 only). Valid values are: ucb, kl_ucb, thompson, all')
    args = parser.parse_args()
    pass_fail = ['FAILED', 'PASSED']

    start = time.time()
    if args.task == '1' or args.task == 'all':
        if args.task == 'all':
            args.algo = 'all'
        if args.algo is None:
            print('Please specify an algorithm for task 1')
            exit(1)
        if args.algo.lower() not in ['ucb', 'kl_ucb', 'thompson', 'all']:
            print('Invalid algorithm')
            exit(1)

        print("="*18+" Task 1 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            scores, regrets = grade_task1(f'testcases/task1-{i}.txt', args.algo)
            for algo, score in scores.items():
                print("{:18}: {}. Regret: {:.2f}".format(algo, pass_fail[score], regrets[algo]))
            print("")
    
    if args.task == '2' or args.task == 'all':
        print("="*18+" Task 2 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            score, regret = grade_task2(f'testcases/task2-{i}.txt')
            print("Batched Algorithm: {}. Regret: {:.2f}".format(pass_fail[score], regret))
            print("")
    
    if args.task == '3' or args.task == 'all':
        print("="*18+" Task 3 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            score, regret = grade_task3(f'testcases/task3-{i}.txt')
            print("Many Arms Algorithm: {}. Regret: {:.2f}".format(pass_fail[score], regret))
            print("")
    end = time.time()

    print("Time elapsed: {:.2f} seconds".format(end-start))

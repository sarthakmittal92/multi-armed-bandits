# multi-armed-bandits

Repository for the course project done as part of CS-747 (Foundations of Intelligent & Learning Agents) course at IIT Bombay in Autumn 2022.  
Webpage: https://sarthakmittal92.github.io/projects/aut22/multi-armed-bandits

## Overview
This assignment tests your understanding of the regret minimization algorithms discussed in class, and ability to extend them to different scenarios. There are three tasks, each worth 5 marks.

To begin, in Task 1, you will implement UCB, KL-UCB, and Thompson Sampling, more or less identical to the versions discussed in class.

Task 2 involves coming up with an algorithm for batched sampling. The idea is that at every decision making step, the algorithm must specify an entire batch of arms to pull (for example, if the batch size is 100, it could be split as perhaps 25 pulls for arm 1, 55 pulls for arm 2, and 20 pulls for arm 3). All these pulls are performed and the results returned to the algorithm in aggregate before its next batch of pulls.

In Task 3, you need to come up with an algorithm for the case when the horizon is equal to the number of arms, but it is given that the arm means are distributed uniformly (so if the horizon is 100, the arm means are a permutation of [0, 0.01, 0.02, ..., 0.99]).

The theory developed in class applies meaningfully only when the horizon is large; how would you deal with shorter horizons? The fact that the bandit instance comes from a restricted family could possibly help.

All the code you write for this assignment must be in Python 3.8.10. The only libraries you may use are Numpy v1.21.0 (to work with vectors and matrices) and Matplotlib (for generating plots). All of these come installed with the docker image that has been shared for the course, and are already imported in the files you need to complete.

## Code Structure
[This compressed directory](https://www.cse.iitb.ac.in/~shivaram/teaching/cs747-a2022/pa-1/code.tar.gz) has all the files required. bernoulli_bandit.py defines the BernoulliBandit which, strictly speaking, you do not need to worry about. It is, however, advised that you read through the code to understand how the pull and batch_pull functions work. We have also provided simulator.py to run simulations and generate plots, which you'll have to submit as described later. Finally, there's autograder.py which evaluates your algorithms for a fixed few bandit instances, and outputs the score you would have received if we were evaluating your algorithms on these instances. The only files you need to edit are task1.py, task2.py, and task3.py. Do not edit any other files. You are allowed to comment/uncomment the final few lines of simulator.py. It is strongly recommended that in any code that you write, which involves generating random numbers, you fix the seed for your generation process. The consequence is that your code will produce identical results each time (so long as we call it from an outer loop that also has its random seed fixed). This is a good practice while running experiments with programs.

For evaluation, we will use another set of bandit instances in the autograder, and use their scores as is for 80% of the evaluation. For a particular test instance, the pass/fail criterion of the autograder is determined based on your regret lying within 1.5 times of the regret of our reference implementation. If your code produces an error, it will directly receive a 0 score in the autograded tasks. It will also get 0 marks if for any subtask whatsoever, the autograder takes over 20 minutes to run the subtask. The remaining part of the evaluation will be done based on your report, which includes plots, and explanation of your algorithms. See the exact details below.

## Report
Your report needs to have all the plots that simulator.py generates. There are 5 plots in total (3 for task 1 and 1 each for tasks 2 and 3). You do not need to include the epsilon-greedy plot in your report. In addition, you need to explain your method for each task. For task 1, explain your code for the three algorithms, and for tasks 2 and 3 explain your approach to the problem and provide justification to the trend seen in the plots.

## Complete Problem
The full problem statement is accessible here: [CS747 Programming Assigmnent 1](https://hackmd.io/@sarthakmittal/ryQtJgOyo)
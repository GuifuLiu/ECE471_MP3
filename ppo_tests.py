from ppo import *
import numpy as np

# Test for visualization function in task 4.1

iteration_rewards = []
x = np.arange(1,200)
smoothed_rewards = np.log(x)
smoothed_slo_preservations = np.exp(x)
smoothed_cpu_utils = x**2

visualization(iteration_rewards, smoothed_rewards, smoothed_slo_preservations, smoothed_cpu_utils)
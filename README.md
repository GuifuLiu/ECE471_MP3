# ECE471_MP3

## Task 1.1

Task 1.1 was implemented in the test_env() function which followed the four outlined requirements for calling the step function on the environment and printing the result.

## Task 1.2

Task 1.2 was implemented in the generate_traces() function in main.py. The policy randomly chooses whether to increase or decrease in an action then randomly chooses whether to update the CPU shares or the number of containers. The function also prints the CPU utilization and SLO preservation of the random trajectory as specified.

## Task 2.3

Task 2.3 was completed in ppo.py and created the actor and critic neural networks for the Policy Proximity Optimization algorithm. In the initialization for both the actor and critic networks a pytorch neural network was created with an input layer, hidden layer, and output layer. ReLU was used as the activation function and Softmax was used as the output function.

In the forward() function for the neural networks the input of the function was simply passed into the neural network. In the train() function of the PPO class, 3 lines of code were added after loss was calculated to perform backwards propagation. The optimizer was also updated by performing another step after the backwards propagation.

## Task 3.1

Task 3.1 was implemented in the convert_state_action_to_reward_overprovisioning() function in util.py. The reward function simply determines the reward based on SLO preservation, which is the variable in the state resembling performance. In order to optimize this reward function the RL algorithm could naively allocate more resources than it actually requires.

## Task 3.2

Task 3.2 was implemented in the convert_state_action_to_reward_tightpacking() function in util.py. The reward function uses CPU utilzation as its reward criteria. In this case the agent would try to maximize this reward by packing as many containers on a node as possible.

## Task 3.3

Task 3.3 was implemented in convert_state_action_to_reward() in util.py. The reward function balances performance and resource utilization by adding SLO preservation and CPU utilization then subtracting the number of containers.

## Task 3.4

Task 3.4 is also implemented in convert_state_action_to_reward() in util.py. The undesirable action of scaling up then scaling down or vice versa is avoided by decreasing the reward when it occurs. The illegal actions of removing nonexistent CPU shares/containers and creating too many CPU shares/containers are avoided by giving the actions 0 reward.

## Task 4.1

Task 4.1 is implemented in visualization() in ppo.py. PyPlot is used to visualize the reward function plotted against the SLO Preservation and CPU Utilization for each iteration. The datetime library is used to timestamp a file name and the visualization is written to a .png file.

## Task 4.2

load_checkpoint() and save_checkpoint() in ppo.py allow for the saving and loading of the actor and critic neural nets after they have been trained. This was done by saving the pytorch objects to a .tar file.

## Task 4.3

Task 4.3 was implemented in the generate_traces_trained() function in main.py. The initial policy from task 1.2 was added to find the total reward for 2000 steps. The trained RL agent was then used to generate actions and calculate the total reward for 2000 steps by calling agent.calc_action().

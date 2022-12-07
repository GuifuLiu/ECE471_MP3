from util import *
import random
from serverless_env import SimEnvironment
from pg import pg
from ppo import PPO
from dqn import dqn
from util import convert_state_action_to_reward
from util import convert_state_action_to_reward_overprovisioning
from util import convert_state_action_to_reward_tightpacking
from util import print_state


def test_env(env, function_name):
    """
    [Task 1.1] TODO: Write your code here to:
    - Call the step function step() by providing a vertical scaling-up action (i.e., adding 128 cpu.shares)
    - Print the received current state (ignore the reward for now)
    - Call the step function step() by providing a horizontal scaling-out action (i.e., adding 1 container)
    - Print the received current state (ignore the reward for now)
    [Task 1.1] Hint: Check out serverless_env.py
    """

    # Perform one RL step by providing a vertical scaling-up action
    curr_state, curr_reward, done = env.step(function_name=function_name, action={'vertical': 128, 'horizontal': 0})

    # Print the received current state
    print_state(curr_state)

    # Perform one RL step by providing a horizontal scaling-out action
    curr_state, curr_reward, done = env.step(function_name=function_name, action={'vertical': 0, 'horizontal': 1})

    # Print the received current state
    print_state(curr_state)


def generate_traces(env, function_name):
    state = env.reset(function_name)

    num_steps = 10
    file = open("example-traces.csv", "w")
    file.write('avg_cpu_util,slo_preservation,total_cpu_shares,cpu_shares_others,num_containers,arrival_rate,' +
               'vertical_scaling,horizontal_action,reward\n')

    for i in range(num_steps):

        # Create random action policy for vertical or horizontal scaling
        decrease = -1 if random.choice([True, False]) else 1

        if random.choice([True, False]):
          random_policy = {'vertical': 128 * decrease, 'horizontal': 0}
        else:
          random_policy = {'vertical': 0, 'horizontal': 1 * decrease}

        # Performing the RL step
        next_state, reward, done = env.step(function_name=function_name, action=random_policy)

        # Printing utilization and SLO preservation
        print('CPU Utilization: %f, SLO Preservation: %f' % (state[0], state[1]))

        # print to file
        file.write(','.join([str(j) for j in state]) + ',' + 
                             str(random_policy['vertical']) + ',' + 
                             str(random_policy['horizontal']) + ',' + 
                             str(reward) + '\n')
        state = next_state

    file.close()
    print('Trajectory generated!')

#add and complete the following function for Task 4.3:
def generate_traces_trained(env, function_name, agent, arrival_rate):

    #your policy in 1.2
    env.reset(function_name)
    state = env.reset_arrival_rate(function_name,arrival_rate)

    num_steps = 2000

    file = open("traces_your_policy.csv", "w")
    file.write('avg_cpu_util,slo_preservation,total_cpu_shares,cpu_shares_others,num_containers,arrival_rate,' +
               'vertical_scaling,horizontal_action,reward\n')

    total_reward = 0 
    for i in range(num_steps):
        
        # [Task 4.3] TODO:here use the same code you use for Task 1.2
        # [Your Code]

        # Create random action policy for vertical or horizontal scaling
        decrease = -1 if random.choice([True, False]) else 1

        if random.choice([True, False]):
          random_policy = {'vertical': 128 * decrease, 'horizontal': 0}
        else:
          random_policy = {'vertical': 0, 'horizontal': 1 * decrease}

        # Performing the RL step
        next_state, reward, done = env.step(function_name=function_name, action=random_policy)


        total_reward += reward
        # print to file
        file.write(','.join([str(j) for j in state]) + ',' + str(vertical_action) + ',' + str(horizontal_action) +
                   ',' + str(reward) + '\n')
        state = next_state

    file.write('total_reward: ' + str(total_reward))

    print("your policy total reward: ", total_reward)
    file.close()


    #RL  policy  
    #env.reset(function_name)
    #state = env.reset_arrival_rate(function_name,arrival_rate)[:NUM_STATES]
    state = state[:NUM_STATES]

    file = open("traces_RL.csv", "w")
    file.write('avg_cpu_util,slo_preservation,total_cpu_shares,cpu_shares_others,num_containers,arrival_rate,' +
               'vertical_scaling,horizontal_action,reward\n')
    
    total_reward = 0 

    for i in range(num_steps):

        # [Task 4.3] TODO:here use the trained RL agent to generate the action for each step
        # hint: check calc_action in class PPO
        # [Your Code]

        action = agent.calc_action(state)
        # Performing the RL step
        next_state, reward, done = env.step(function_name=function_name, action=action)

        total_reward += reward

        # print to file
        file.write(','.join([str(j) for j in state]) + ',' + str(action_to_execute['vertical']) + ',' + str(action_to_execute['horizontal']) +
                   ',' + str(reward) + '\n')
        state = next_state
        
    file.write('total_reward: ' + str(total_reward))

    print("RL policy total reward: ", total_reward)

    file.close()
    print('Trajectory generated!')

def test_reward_function():
    action = {
        'vertical': 0,
        'horizontal': 1
    }

    last_action = {
        'vertical': 0,
        'horizontal': 1
    }

    # [cpu util, slo preservation, cpu.shares, cpu.shares (others), # of containers, arrival rate, latency]
    state_a = [0.5, 0.7, 0.2, 0.0, 0.2, 0.3, 1.3]
    state_b = [0.8, 0.7, 0.2, 0.0, 0.2, 0.3, 1.3]
    state_c = [0.5, 0.9, 0.2, 0.0, 0.2, 0.3, 1.3]
    arrival_rate = 3

    reward_a = convert_state_action_to_reward_overprovisioning(state_a, action, last_action, arrival_rate)
    reward_c = convert_state_action_to_reward_overprovisioning(state_c, action, last_action, arrival_rate)

    if reward_a < reward_c:
        print('Task 3.1 - Reward function test passed!')
    else:
        print('Task 3.1 - Reward function test failed!')

    reward_a = convert_state_action_to_reward_tightpacking(state_a, action, last_action, arrival_rate)
    reward_b = convert_state_action_to_reward_tightpacking(state_b, action, last_action, arrival_rate)

    if reward_a < reward_b:
        print('Task 3.2 - Reward function test passed!')
    else:
        print('Task 3.2 - Reward function test failed!')

    reward_a = convert_state_action_to_reward(state_a, action, last_action, arrival_rate)
    reward_b = convert_state_action_to_reward(state_b, action, last_action, arrival_rate)
    reward_c = convert_state_action_to_reward(state_c, action, last_action, arrival_rate)

    if reward_a < reward_b and reward_a < reward_c:
        print('Task 3.3 - Reward function test passed!')
    else:
        print('Task 3.3 - Reward function test failed!')


def main():
    """
    This is the main function for RL training and inference.
    Please complete the tasks below.
    """

    """
    Task 1.1 (Part I)
    - Create and initialize an RL environment
    - Reset the environment and print the initial state
    """
    # create and initialize the environment for rl training
    env = SimEnvironment()
    function_name = env.get_function_name()

    # Reset the environment and print the initial state
    initial_state = env.reset(function_name=function_name)
    print_state(initial_state)

    """
    Task 1.1 (Part II)
    - Call the step function step() by providing a vertical scaling-up action (i.e., adding 128 cpu.shares)
    - Print the received current state (ignore the reward for now)
    - Call the step function step() by providing a horizontal scaling-out action (i.e., adding 1 container)
    - Print the received current state (ignore the reward for now)
    """
    # test the initialized environment
    test_env(env, function_name)
    print('')

    """
    Task 1.2
    Create your own policy (could be a random action generator) and perform 10 RL steps (i.e., a trajectory)
    - Complete the function generate_traces(env, function_name) that we provided to you
    - Check out the CPU utilization and SLO preservation along the way
    """
    # print a sample trajectory
    generate_traces(env, function_name)
    print('>>>>>>>>>> End of Task 1 <<<<<<<<<<\n')

    """
    Task 2:
    Implement the RL algorithm PPO (one of the state-of-the-art actor-critic RL algorithms)
    - Complete the skeleton functions that we provided in the ppo.py
    - (Optional) Complete the skeleton functions that we provided in the pg/
    - (Optional) Complete the skeleton functions that we provided in the dqn/
    """
    # init an RL agent
    agent_type = 'PPO'
    agent = None
    if agent_type == 'PPO':
        agent = PPO(env, function_name)
    elif agent_type == 'PG':
        agent = pg.PG(env, function_name)
    elif agent_type == 'DQN':
        agent = dqn.DQN(env, function_name)
    print('RL agent initialized!')
    print('>>>>>>>>>> End of Task 2 <<<<<<<<<<\n')

    """
    Task 3:
    Design and implement the reward functions to achieve different policies
    - Complete the convert_state_action_to_reward*() functions that we provided in the util.py
    """
    test_reward_function()
    print('>>>>>>>>>> End of Task 3 <<<<<<<<<<\n')

    """
    Task 4:
    - Complete the save_checkpoint() and load_checkpoint() function (in ppo.py) for RL model checkpointing
    - Change use_checkpoint to True and agent_type to 'PPO'
    - Change the checkpoint_file path to be the location of the checkpoint file that you want to use
    - Complete the visualization() function for RL training visualization
    """
    # init from saved checkpoints
    use_checkpoint = False
    checkpoint_file = './checkpoints/ppo-ep0.pth.tar'
    if use_checkpoint:
        agent.load_checkpoint(checkpoint_file)
        print('>>>>>>>>>> End of Task 4 <<<<<<<<<<\n')

    # CP2 manually fixing arrival rate for training
    env.reset_arrival_rate(function_name,7)

    # start RL training
    agent.train()

    generate_traces_trained(env, function_name, agent, arrival_rate)


if __name__ == "__main__":
    main()

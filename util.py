import numpy as np


NUM_STATES = 6
NUM_GLOBAL_STATES = 4
NUM_GLOBAL_STATES_WITH_VARIANCE = NUM_GLOBAL_STATES * 2
NUM_MEAN_FIELD_STATES = NUM_STATES
NUM_MEAN_FIELD_STATES_WITH_ACTIONS = NUM_STATES + 2  # 2 for horizontal and vertical actions
NUM_ACTIONS = 5
NUM_TOTAL_ACTIONS = 10
VERTICAL_SCALING_STEP = 128
HORIZONTAL_SCALING_STEP = 1
MAX_NUM_CONTAINERS = 10.0
MAX_CPU_SHARES = 2048.0

ILLEGAL_PENALTY = -1
DATA_PATH = 'data.csv'


# print current state
def print_state(state_list):
    print('Avg CPU util: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]),
          'Num of containers:', state_list[4],
          'CPU shares:', state_list[2], 'CPU shares (others):', state_list[3],
          'Arrival rate:', state_list[4])


# print current action
def print_action(action_dict):
    if action_dict['vertical'] > 0:
        print('Action: Scaling-up by', VERTICAL_SCALING_STEP, 'cpu.shares')
    elif action_dict['vertical'] < 0:
        print('Action: Scaling-down by', VERTICAL_SCALING_STEP, 'cpu.shares')
    elif action_dict['horizontal'] > 0:
        print('Action: Scaling-out by', HORIZONTAL_SCALING_STEP, 'container')
    elif action_dict['horizontal'] < 0:
        print('Action: Scaling-in by', HORIZONTAL_SCALING_STEP, 'container')
    else:
        print('No action to perform')


# print (state, action, reward) for the current step
def print_step_info(step, state_list, action_dict, reward):
    state = 'State: [Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]) +\
            ' Num of containers: ' + str(state_list[4]) +\
            ' CPU shares: ' + str(state_list[2]) + ' CPU shares (others): ' + str(state_list[3]) +\
            ' Arrival rate: ' + str(state_list[5]) + ']'
    action = 'Action: N/A'
    if action_dict['vertical'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['vertical'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['horizontal'] > 0:
        action = 'Action: Scaling-out by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    elif action_dict['horizontal'] < 0:
        action = 'Action: Scaling-in by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    print('Step #' + str(step), '|', state, '|', action, '| Reward:', reward)


def print_step_info_with_function_name(step, state_list, action_dict, reward, function_name):
    state = 'State: [Avg CPU utilization: {:.7f} Avg SLO preservation: {:.7f}'.format(state_list[0], state_list[1]) +\
            ' Num of containers: ' + str(state_list[4]) +\
            ' CPU shares: ' + str(state_list[2]) + ' CPU shares (others): ' + str(state_list[3]) +\
            ' Arrival rate: ' + str(state_list[5]) + ']'
    action = 'Action: N/A'
    if action_dict['vertical'] > 0:
        action = 'Action: Scaling-up by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['vertical'] < 0:
        action = 'Action: Scaling-down by ' + str(VERTICAL_SCALING_STEP) + ' cpu.shares'
    elif action_dict['horizontal'] > 0:
        action = 'Action: Scaling-out by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    elif action_dict['horizontal'] < 0:
        action = 'Action: Scaling-in by ' + str(HORIZONTAL_SCALING_STEP) + ' container'
    print('[', function_name, '] - Step #' + str(step), '|', state, '|', action, '| Reward:', reward)


# calculate the reward based on the current states (after the execution of the current action)
# + cpu utilization percentage [0, 1]
# + slo preservation [0, 1]
# - number of containers (/ arrival rate)
# - penalty
def convert_state_action_to_reward(state, action, last_action, arrival_rate):
    # [Task 3.3] TODO: Implement a reward function that achieves a balance between performance and utilization
    # [Task 3.4] TODO: Add punishment to the reward which helps avoid undesired or illegal actions
    # [Your Code]
    
    cpu_util = state[0]
    slo_preservation = state[1]
    cpu_shares = state[2]
    num_containers = state[4]
    arrival_rate_change = state[5]
    reward = (cpu_util + slo_preservation) - num_containers

    # Undesirable action: scale up/down then scale down/up
    if last_action['vertical'] > 0 and action['vertical'] < 0:
        reward -= 0.25
    elif last_action['vertical'] < 0 and action['vertical'] > 0:
        reward -= 0.25

    # Illegal action: remove non-existant container
    if num_containers == 0 and action['horizontal'] == -1:
        reward = 0

    # Illegal action: create too many containers
    if MAX_NUM_CONTAINERS < num_containers + action['horizontal']:
        reward = 0

    # Illegal action: remove non-existant cpu shares
    if cpu_shares + action['vertical'] < 0:
        reward = 0

    # Illegal action: create too many cpu shares
    if MAX_CPU_SHARES < cpu_shares + action['vertical']:
        reward = 0

    # Ensuring the reward falls between 0 and 1
    return min(max(reward, 0), 1)


def convert_state_action_to_reward_overprovisioning(state, action, last_action, arrival_rate):
    # [Task 3.1] TODO: Implement a reward function that overprovision resources to achieve good performance
    # [Your Code]

    slo_preservation = state[1]
    reward = slo_preservation

    return reward


def convert_state_action_to_reward_tightpacking(state, action, last_action, arrival_rate):
    # [Task 3.2] TODO: Implement a reward function that underprovision resources to save utilization
    # [Your Code]

    cpu_util = state[0]
    reward = cpu_util

    return reward


# check the correctness of the action
def is_correct(action, num_containers, cpu_shares_per_container):
    # - scaling in when there's no containers
    # - scaling up/down when there's no containers
    # - scaling down leading to cpu.shares < 128
    if num_containers <= 0:
        if action['vertical'] != 0 or action['horizontal'] < 0:
            return False
    else:
        if action['vertical'] + cpu_shares_per_container < 128:
            return False

    # check for maximum number of containers or cpu.shares per container
    if num_containers + action['horizontal'] > MAX_NUM_CONTAINERS:
        return False
    if action['vertical'] + cpu_shares_per_container > MAX_CPU_SHARES:
        return False

    return True


# count the number of parameters in a model
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

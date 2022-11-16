import os

import psutil
import torch
from torch import nn
import matplotlib.pyplot as plt
from util import *

PLOT_FIG = True
SAVE_FIG = True
SAVE_TO_FILE = True

CHECKPOINT_DIR = './checkpoints/'

TOTAL_ITERATIONS = 500
EPISODES_PER_ITERATION = 5
EPISODE_LENGTH = 200

# hyperparameters
DISCOUNT = 0.99
HIDDEN_SIZE = 64
LR = 3e-4  # 5e-3 5e-6
SGD_EPOCHS = 5
MINI_BATCH_SIZE = 5
CLIP = 0.2
ENTROPY_COEFFICIENT = 0.01  # 0.001
CRITIC_LOSS_DISCOUNT = 0.05  # 0.03

FLAG_CONTINUOUS_ACTION = False

MAX_SAME_ITERATIONS = 2 * SGD_EPOCHS


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        # [Task 2.3] TODO: Write your code here to implement the type of layers that you will need in forward()
        # [Your Code]

        input_layer = nn.Linear(input_size, HIDDEN_SIZE)
        hidden_layer = nn.Linear(HIDDEN_SIZE, hidden_size)
        output_layer = nn.Linear(hidden_size, output_size)
        function = nn.ReLU()
        output_function = nn.Softmax()

        self.neural_net = nn.Sequential(input_layer, function, hidden_layer, function, output_layer, output_function)
        pass

    def forward(self, input_):
        output = None
        # [Task 2.3] TODO: Write your code here to implement the forward pass of the actor network
        # [Your Code]
        output = self.neural_net(torch.FloatTensor(input_))
        return output


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticNetwork, self).__init__()
        # [Task 2.3] TODO: Write your code here to implement the type of layers that you will need in forward()
        # [Your Code]
        input_layer = nn.Linear(input_size, HIDDEN_SIZE)
        hidden_layer = nn.Linear(HIDDEN_SIZE, hidden_size)
        output_layer = nn.Linear(hidden_size, output_size)
        function = nn.ReLU()

        self.neural_net = nn.Sequential(input_layer, function, hidden_layer, function, output_layer)
        pass

    def forward(self, input_):
        output = None
        # [Task 2.3] TODO: Write your code here to implement the forward pass of the critic network
        # [Your Code]
        output = self.neural_net(torch.FloatTensor(input_))
        return output


def calc_gae(rewards):
    returns = []
    for episode_rewards in reversed(rewards):
        discounted_return = 0.0
        # Caution: Episodes might have different lengths if stopped earlier
        for reward in reversed(episode_rewards):
            discounted_return = reward + discounted_return * DISCOUNT
            returns.insert(0, discounted_return)

    returns = torch.FloatTensor(returns)
    return returns


def visualization(iteration_rewards, smoothed_rewards):
    # [Task 4.1] TODO: Write your code here to visualize the reward progression (learning curve) of the RL agent
    # [Task 4.1] TODO: Save the figure to a local file
    # [Your Code]
    pass


class PPO:
    def __init__(self, env, function_name):
        self.env = env
        self.function_name = function_name

        self.state_size = NUM_STATES
        self.action_size = NUM_ACTIONS

        self.actor = ActorNetwork(self.state_size, HIDDEN_SIZE, self.action_size)
        self.critic = CriticNetwork(self.state_size, HIDDEN_SIZE, 1)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.cov = torch.diag(torch.ones(self.action_size, ) * 0.5)

        self.skip_update = False

        self.num_same_parameter_actor = 0
        self.num_same_parameter_critic = 0
        self.parameter_actor = None
        self.parameter_critic = None

    # skip update for the policy and critic network
    def disable_update(self):
        self.skip_update = True

    # enable update for the policy and critic network
    def enable_update(self):
        self.skip_update = False

    def calc_action(self, state):
        if FLAG_CONTINUOUS_ACTION:
            mean = self.actor(state)
            dist = torch.distributions.MultivariateNormal(mean, self.cov)
        else:
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def train(self):
        # create the checkpoint directory if not created
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        # for plots
        iteration_rewards = []
        smoothed_rewards = []

        # for explainability
        iteration_slo_preservations = []
        smoothed_slo_preservations = []
        iteration_cpu_utils = []
        smoothed_cpu_utils = []

        pid = os.getpid()
        python_process = psutil.Process(pid)
        for iteration in range(TOTAL_ITERATIONS):
            # get resource profiles
            memory_usage = python_process.memory_info()[0] / 2. ** 20  # memory use in MB
            cpu_util = python_process.cpu_percent(interval=None)
            print('Memory use:', memory_usage, 'CPU util:', cpu_util)

            states = []
            actions = []
            rewards = []
            log_probs = []
            all_cpu_utils = []
            all_slo_preservations = []

            for episode in range(EPISODES_PER_ITERATION):
                state = self.env.reset(self.function_name)[:NUM_STATES]
                episode_rewards = []

                for step in range(EPISODE_LENGTH):
                    action, log_prob = self.calc_action(state)

                    action_to_execute = {
                        'vertical': 0,
                        'horizontal': 0,
                    }

                    if action == 0:
                        # do nothing
                        pass
                    elif action == 1:
                        # scaling out
                        action_to_execute['horizontal'] = HORIZONTAL_SCALING_STEP
                    elif action == 2:
                        # scaling in
                        action_to_execute['horizontal'] = -HORIZONTAL_SCALING_STEP
                    elif action == 3:
                        # scaling up
                        action_to_execute['vertical'] = VERTICAL_SCALING_STEP
                    elif action == 4:
                        # scaling down
                        action_to_execute['vertical'] = -VERTICAL_SCALING_STEP

                    next_state, reward, done = self.env.step(self.function_name, action_to_execute)
                    next_state = next_state[:NUM_STATES]

                    states.append(state)
                    episode_rewards.append(reward)
                    log_probs.append(log_prob)
                    if FLAG_CONTINUOUS_ACTION:
                        actions.append(action)
                    else:
                        actions.append(action.item())

                    all_cpu_utils.append(next_state[0])
                    all_slo_preservations.append(next_state[1])

                    # verbose
                    if episode % 5 == 0 and iteration % 50 == 0:
                        print_step_info(step, state, action_to_execute, reward)

                    if done:
                        break
                    state = next_state

                # end of one episode
                rewards.append(episode_rewards)

            # end of one iteration
            iteration_rewards.append(np.mean([np.sum(episode_rewards) for episode_rewards in rewards]))
            smoothed_rewards.append(np.mean(iteration_rewards[-10:]))
            iteration_slo_preservations.append(np.mean(all_slo_preservations))
            smoothed_slo_preservations.append(np.mean(iteration_slo_preservations[-10:]))
            iteration_cpu_utils.append(np.mean(all_cpu_utils))
            smoothed_cpu_utils.append(np.mean(iteration_cpu_utils[-10:]))

            # states = torch.FloatTensor(states)
            states = torch.FloatTensor(np.array(states))
            if FLAG_CONTINUOUS_ACTION:
                actions = torch.FloatTensor(actions)
            else:
                actions = torch.IntTensor(actions)
            log_probs = torch.FloatTensor(log_probs)

            average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
            print('Iteration:', iteration, '- Average rewards across episodes:', np.round(average_rewards, decimals=3),
                  '| Moving average:', np.round(np.mean(iteration_rewards[-10:]), decimals=3))

            if self.skip_update:
                continue

            returns = calc_gae(rewards)

            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for epoch in range(SGD_EPOCHS):
                batch_size = states.size(0)  # whole batch of size 4000 (= 20 * 200)
                # use mini-batch instead of the whole batch
                for mini_batch in range(batch_size // MINI_BATCH_SIZE):
                    ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)

                    values = self.critic(states[ids]).squeeze()
                    if FLAG_CONTINUOUS_ACTION:
                        mean = self.actor(states[ids])
                        dist = torch.distributions.MultivariateNormal(mean, self.cov)
                    else:
                        action_probs = self.actor(states[ids])
                        dist = torch.distributions.Categorical(action_probs)

                    log_probs_new = dist.log_prob(actions[ids])
                    entropy = dist.entropy().mean()

                    ratios = (log_probs_new - log_probs[ids]).exp()

                    surrogate1 = ratios * advantage[ids]
                    surrogate2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage[ids]
                    actor_loss = - torch.min(surrogate1, surrogate2).mean()
                    critic_loss = (returns[ids] - values).pow(2).mean()

                    loss = actor_loss + CRITIC_LOSS_DISCOUNT * critic_loss - ENTROPY_COEFFICIENT * entropy

                    # [Task 2.3] TODO: Given loss, write you code here to update model parameters (backward propagation)
                    # [Your Code]
                    self.optimzier.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                if self.parameter_actor is None:
                    self.parameter_actor = []
                    for parameter in self.actor.parameters():
                        self.parameter_actor.append(parameter.clone())
                else:
                    # compare the model parameters
                    is_equal = True
                    for idx, parameter in enumerate(list(self.actor.parameters())):
                        if not torch.equal(parameter, self.parameter_actor[idx]):
                            is_equal = False
                            break
                    if is_equal:
                        self.num_same_parameter_actor += 1
                    else:
                        self.num_same_parameter_actor = 0
                        self.parameter_actor = []
                        for parameter in self.actor.parameters():
                            self.parameter_actor.append(parameter.clone())
                if self.parameter_critic is None:
                    self.parameter_critic = []
                    for parameter in self.critic.parameters():
                        self.parameter_critic.append(parameter.clone())
                else:
                    # compare the model parameters one by one
                    is_equal = True
                    for idx, parameter in enumerate(list(self.critic.parameters())):
                        if not torch.equal(parameter, self.parameter_critic[idx]):
                            is_equal = False
                            break
                    if is_equal:
                        self.num_same_parameter_critic += 1
                    else:
                        self.num_same_parameter_critic = 0
                        # self.parameter_critic = list(self.critic.parameters())
                        self.parameter_critic = []
                        for parameter in self.critic.parameters():
                            self.parameter_critic.append(parameter.clone())

            if self.num_same_parameter_critic > MAX_SAME_ITERATIONS and\
                    self.num_same_parameter_actor > MAX_SAME_ITERATIONS:
                break

            # save to checkpoint
            if iteration % 100 == 0:
                self.save_checkpoint(iteration)

        # plot
        if PLOT_FIG:
            visualization(iteration_rewards, smoothed_rewards)

        # write rewards to file
        if SAVE_TO_FILE:
            file = open("ppo_smoothed_rewards.txt", "w")
            for reward in smoothed_rewards:
                file.write(str(reward) + "\n")
            file.close()
            file = open("ppo_iteration_rewards.txt", "w")
            for reward in iteration_rewards:
                file.write(str(reward) + "\n")
            file.close()

            # write cpu_utils and slo_preservations to file
            file = open("ppo_cpu_utils_all.txt", "w")
            for cpu_util in iteration_cpu_utils:
                file.write(str(cpu_util) + "\n")
            file.close()
            file = open("ppo_cpu_utils_smoothed.txt", "w")
            for cpu_util in smoothed_cpu_utils:
                file.write(str(cpu_util) + "\n")
            file.close()
            file = open("ppo_slo_preservation_all.txt", "w")
            for ratio in iteration_slo_preservations:
                file.write(str(ratio) + "\n")
            file.close()
            file = open("ppo_slo_preservation_smoothed.txt", "w")
            for ratio in smoothed_slo_preservations:
                file.write(str(ratio) + "\n")
            file.close()

    # load all model parameters from a saved checkpoint
    def load_checkpoint(self, checkpoint_file_path):
        # [Task 4.2] TODO: Write your code here to load the RL model from a local checkpoint
        # [Task 4.2] Hint: Check out torch.load() function documentation
        # [Your Code]
        pass

    # save all model parameters to a checkpoint
    def save_checkpoint(self, episode_num):
        # [Task 4.2] TODO: Write your code here to save the RL model to a local checkpoint
        # [Task 4.2] Hint: Check out torch.save() function documentation
        # [Your Code]
        pass

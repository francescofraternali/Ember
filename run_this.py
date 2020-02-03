import gym
import gym_en_harv
import random
import pandas as pd
import numpy as np
import time
from gym.envs.registration import register
#from IPython.display import clear_output

#env_name = "FrozenLake-v0"
env_name = "gym_en_harv-v0"
#env_name = "FrozenLakeNoSlip-v0"
env = gym.make(env_name)

tot_episodes = 20000; tot_actions = 4
max_rew = 0

print("Observation space:", env.observation_space)
print("Action space:", tot_actions)


class QLearningTable:
    def __init__(self, actions, reward_decay=0.99, learning_rate=0.1):
        #super().__init__(env)
        #self.state_size = env.observation_space.n
        #print("State size:", self.state_size)
        self.gamma = reward_decay
        self.lr = learning_rate
        self.actions = actions
        self.epsilon = 0.01
        self.learning_rate = learning_rate
        self.delta = 0.000005
        self.build_model()

    def build_model(self):
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        #self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    def get_action(self, state):
        '''
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.eps else action_greedy
        '''

        self.check_state_exist(state)
        # action selection
        if np.random.uniform() < self.epsilon: # choose best action
            state = str(state)
            state_action = self.q_table.loc[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else: # choose random action
            action = np.random.choice(self.actions)
            #print ("Random Action Selected")
        return action

    def check_state_exist(self, state):
        #print(self.q_table.index)
        #time.sleep(5)
        state = str(state)
        if state not in self.q_table.index:
            # append new state to q table
            print ("New State: " + str(state))
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            #print(self.q_table)
            #time.sleep(1)

    def learn(self, s, a, r, s_):
        #self.check_state_exist(s)
        self.check_state_exist(s_)
        #print(s,s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

        self.epsilon = min(self.epsilon + self.delta, 1)

        return self.epsilon

agent = QLearningTable(list(range(tot_actions)))

for ep in range(tot_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        eps = agent.learn(str(state), action, reward, str(next_state))
        state = next_state
        total_reward += reward

        #time.sleep(2)
        if done and eps >= 1:
            env.render(ep, total_reward)
            exit()
        #print(agent.q_table)
        #time.sleep(0.05)
        #clear_output(wait=True)

    #if total_reward > max_rew:
    #    env.render(ep, total_reward)
    max_rew = max(max_rew, total_reward)

    print("Episode: {}, Total reward: {}, Max Reward {}, eps: {}".format(ep, total_reward, max_rew, eps))

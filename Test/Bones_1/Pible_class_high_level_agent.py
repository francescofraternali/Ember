import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym import spaces, logger
import datetime
import json
from Pible_parameters import *
import Pible_func
import random
import subprocess
from time import sleep
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

num_hours_input = 24

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):

        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32)

        self.Reward = []; self.Mode = [];  self.Time = []

        self.hour = np.array([23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 0])

    def reset(self):

        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        #self.hour = np.array([self.time.hour] * num_hours_input)
        self.hour = np.array([23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 0])

        return self.hour

    def step(self, action):

        self.mode = action

        reward = 0
        if self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17:
            if self.mode == 1:
                reward = 1
        else:
            if self.mode == 0:
                reward = 1

        self.Reward.append(reward); self.Time.append(self.time); self.Mode.append(self.mode)

        self.hour = np.roll(self.hour, 1)
        self.hour[0] = self.time.hour

        self.time = self.time + datetime.timedelta(minutes=60) #next_wake_up_time # in min

        done = self.time >= self.end

        return self.hour, reward, done, {}

    def render(self, tot_rew):
         plt.figure(1)
         ax1 = plt.subplot(211)
         plt.title(('Tot reward: {0}').format(round(tot_rew, 3)))
         plt.plot(self.Time, self.Mode, 'b-', label = 'Mode', markersize = 15)
         plt.ylabel('Mode\n[num]', fontsize=15)
         ax1.set_xticklabels([])
         plt.grid(True)

         ax2 = plt.subplot(212)
         plt.plot(self.Time, self.Reward, 'b.', label = 'Reward', markersize = 15)
         plt.ylabel('Reward\n[num]', fontsize=12)
         plt.xlabel('Time [hh:mm]', fontsize=15)
         xfmt = mdates.DateFormatter('%m/%d %H')
         ax2.xaxis.set_major_formatter(xfmt)
         plt.grid(True)
         plt.show()
         plt.close("all")

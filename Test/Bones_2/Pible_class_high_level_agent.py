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

num_hours_input = 24; num_week_input = 7

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):

        start_train = "10/7/19 00:00:00"
        end_train = "10/14/19 23:00:00"

        start_data_date = datetime.datetime.strptime(start_train, '%m/%d/%y %H:%M:%S')
        end_data_date = datetime.datetime.strptime(end_train, '%m/%d/%y %H:%M:%S')
        self.path_light_data = '/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Ember/Test/Fake_Input.txt'

        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.events_found_dict = []

        self.file_data = []
        with open(self.path_light_data, 'r') as f:
            for line in f:
                line_split = line.split("|")
                checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
                if start_data_date <= checker and checker <= end_data_date:
                    self.file_data.append(line)

        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32)
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            #spaces.Box(SC_volt_min, SC_volt_max, shape=(num_volt_input, ), dtype=np.float32),
            spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
            #spaces.Box(0, 10, shape=(1, ), dtype=np.float32),  #number of events

        ))

        self.Reward = []; self.Mode = [];  self.Time = []

        self.hour = np.array([23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 0])

    def reset(self):

        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.events_found_dict = []
        #self.hour = np.array([self.time.hour] * num_hours_input)
        self.hour = np.array([23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 0])

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        return self.hour, self.week_end

    def step(self, action):

        self.mode = action
        self.next_wake_up_time = 60
        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        if self.mode == 0:
            self.PIR_on_off = 0
        elif self.mode == 1:
            self.PIR_on_off = 1
        elif self.mode == 2:
            event_found = 0
            for time_checker in self.events_found_dict:
                if self.time.time() <= time_checker and time_checker < self.time_next.time():
                #if self.time.hour == time_checker:
                    event_found = 1
                    self.PIR_on_off = 1
                    break
            if event_found == 0:
                self.PIR_on_off = 0


        light, event_gt, self.events_found_dict = Pible_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.events_found_dict, 0, self.file_data)

        reward = Pible_func.reward_func_high_level(self.mode, event_gt, self.PIR_on_off)

        self.Reward.append(reward); self.Time.append(self.time); self.Mode.append(self.mode)

        self.hour = np.roll(self.hour, 1)
        self.hour[0] = self.time.hour
        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        self.time = self.time_next

        done = self.time >= self.end

        return (self.hour, self.week_end), reward, done, {}

    def render(self, tot_rew):
         plt.figure(1)
         ax1 = plt.subplot(211)
         plt.title(('Tot reward: {0}').format(round(tot_rew, 3)))
         plt.plot(self.Time, self.Mode, 'm-', label = 'Mode', markersize = 15)
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

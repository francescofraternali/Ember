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

num_hours_input = 20; num_minutes_input = 20; num_volt_input = 20; num_light_input = 20; num_week_input = 7;  num_events_input = 20

s_t_min_act = 0; s_t_max_act = 1; s_t_min_new = 1; s_t_max_new = 60

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.main_path = config["main_path"]
        self.path_light_data = self.main_path + "/" + "Fake_Input.txt"

        self.train = config["train/test"]
        if self.train == "train":
            start_data_date = datetime.datetime.strptime(config["start_train"], '%m/%d/%y %H:%M:%S')
            end_data_date = datetime.datetime.strptime(config["end_train"], '%m/%d/%y %H:%M:%S')
        else:
            start_data_date = datetime.datetime.strptime(config["start_test"], '%m/%d/%y %H:%M:%S')
            end_data_date = datetime.datetime.strptime(config["end_test"], '%m/%d/%y %H:%M:%S')

        self.file_data = []

        self.next_wake_up_time = 0

        self.light = 0
        #self.time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.light_count = 0

        self.action_space = spaces.Tuple((
            spaces.Discrete(3), # 0: Sleep; 1: Ground Truth Mode (GT); 2 Normal RL Mode (NormRL)
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(SC_volt_min, SC_volt_max, shape=(num_volt_input, ), dtype=np.float32),
            #spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
            spaces.Box(0, 1000, shape=(1, ), dtype=np.float32),  #number of events

        ))
        self.SC_Volt = []; self.Reward = []; self.Mode = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []; self.Len_Dict_Events = []
        self.tot_events = 0; self.events_detect = 0
        self.event_det_hist = []; self.event_miss_hist = []

        self.hour, self.minute, self.light_ar = Pible_func.build_inputs(self.time, num_hours_input, num_minutes_input, num_light_input, self.light)

    def reset(self):
        self.SC_volt = np.array([3.2] * num_volt_input)
        self.events_found_dict = []

        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)

        len_dict_event = np.array([len(self.events_found_dict)])
        #len_dict_event = np.array([0])

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        return self.hour, len_dict_event

    def step(self, action):
        assert action[0] in [0, 1, 2], action

        self.next_wake_up_time = 60
        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        self.mode = action[0]
        #self.mode = 1

        if self.mode == 0:
            self.PIR_on_off = 0
        elif self.mode == 1:
            self.PIR_on_off = 1
        elif self.mode == 2:
            event_found = 0
            for time_checker in self.events_found_dict:
                if self.time.hour == time_checker:
                    event_found = 1
                    self.PIR_on_off = 1
                    break
            if event_found == 0:
                self.PIR_on_off = 0

        #self.PIR_on_off = 1

        self.hour = np.roll(self.hour, 1)
        self.hour[0] = self.time.hour

        self.minute = np.roll(self.minute, 1)
        self.minute[0] = self.time.minute

        self.light = self.light_ar[0]
        self.light_ar = np.roll(self.light_ar, 1)

        SC_temp = self.SC_volt[0]
        #self.SC_volt = np.roll(self.SC_volt, 1)
        self.light, self.light_count, event_gt, self.events_found_dict = Pible_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.events_found_dict, self.light_count, self.light, self.file_data)

        #SC_temp = Pible_func.Energy(SC_temp, self.light, self.PIR_on_off, self.next_wake_up_time, event_gt)

        reward, event_det, event_miss = Pible_func.reward_func_high_level(self.PIR_on_off, self.mode, event_gt, SC_temp, self.next_wake_up_time)
        self.light_ar[0] = self.light
        #self.SC_volt[0] = SC_temp

        if reward > 0:
            self.events_detect += event_gt
        self.tot_events += event_gt

        done = self.time >= self.end

        len_dict_event = np.array([len(self.events_found_dict)])
        #len_dict_event = np.array([0])

        if self.train != "train":
            self.SC_Volt.append(SC_temp); self.Reward.append(reward); self.PIR_hist.append(event_gt); self.Time.append(self.time); self.Light.append(self.light); self.Mode.append(self.mode); self.PIR_OnOff.append(self.PIR_on_off); self.State_Trans.append(self.next_wake_up_time); self.event_det_hist.append(event_det); self.event_miss_hist.append(event_miss); self.Len_Dict_Events.append(len_dict_event)

        self.time = self.time_next #next_wake_up_time # in min

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        info = {}

        return (self.hour, len_dict_event), reward, done, info

    def render(self, episode, tot_rew, start, end, title):
        Pible_func.plot_hist(self.Time, self.Light, self.Mode, self.PIR_OnOff, self.State_Trans, self.Reward, self.Perf, self.SC_Volt, self.PIR_hist, self.event_det_hist, self.event_miss_hist, episode, tot_rew, self.events_detect, self.tot_events, self.Len_Dict_Events, title)

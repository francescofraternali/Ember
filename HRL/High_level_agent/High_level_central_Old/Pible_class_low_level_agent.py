import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym import spaces, logger
import datetime
import json
import os
from Pible_parameters import *
import Pible_func
import random

path_light = "/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Ember/FF66_2150_Middle_Event_RL_Adapted.txt"
light_divider = 2
title = "test"
curr_path = os.getcwd()

path_light_data = path_light;

sc_volt_start_test = "3.1"

num_hours_input = 20; num_minutes_input = 20;
num_volt_input = 20; num_light_input = 20; num_week_input = 7

s_t_min_act = 0; s_t_max_act = 1; s_t_min_new = 1; s_t_max_new = 60

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.train = config["train/test"]
        if self.train == "train":
            st = config["start_train"]
            end = config["end_train"]
            self.start_sc = config["sc_volt_start_train"]
        else:
            st = config["start_test"]
            end = config["end_test"]
            self.start_sc = config["sc_volt_start_test"]

        self.light_div = light_divider

        start_data_date = datetime.datetime.strptime(st, '%m/%d/%y %H:%M:%S')
        end_data_date = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S')
        diff_date = end_data_date - start_data_date
        self.diff_days = diff_date.days

        self.file_data = []
        self.path_light_data = path_light_data
        with open(self.path_light_data, 'r') as f:
            for line in f:
                line_split = line.split("|")
                checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
                if start_data_date <= checker and checker <= end_data_date:
                    self.file_data.append(line)

        self.next_wake_up_time = 0
        self.episode_count = 0
        self.file_data_orig = self.file_data
        self.light_len = len(self.file_data)
        line = self.file_data[0].split('|')
        self.light = int(int(line[8])/light_divider)
        self.light_count = 0
        self.death_days = 0
        self.death_min = 0
        self.days_repeat = 0
        #self.time = datetime.datetime.strptime('01/01/17 00:00:00', '%m/%d/%y %H:%M:%S')
        self.time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        self.time_begin = self.time
        self.end = self.time + datetime.timedelta(hours=24*1)

        self.action_space = spaces.Tuple((
            spaces.Discrete(2), # PIR On_Off
            spaces.Box(s_t_min_act, s_t_max_act, shape=(1, ), dtype=np.float32) # State Transition
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 59, shape=(num_minutes_input, ), dtype=np.float32),       # minutes
            spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_volt_input, ), dtype=np.float32),
            #spaces.Discrete(2)      #week/weekends
            spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
        ))
        self.SC_Volt = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []
        self.tot_events = 0
        self.events_detect = 0
        self.event_det_hist = []
        self.event_miss_hist = []

        self.hour, self.minute, self.light_ar = Pible_func.build_inputs(self.time, num_hours_input, num_minutes_input, num_light_input, self.light)

    def reset(self):

        if self.episode_count % 20 == 0:
            if  self.start_sc == "rand":
                self.SC_rand = random.uniform(SC_volt_die, 3.5)
            else:
                self.SC_rand = float(self.start_sc)

            input_volt = []
            for i in range(0, num_volt_input):
                input_volt.append(self.SC_rand)
            self.SC_volt = np.array(input_volt)
        self.episode_count += 1
        self.death_days = 0
        self.death_min = 0

        if self.light_len == self.light_count: # the file with light has been all read, reset light and time from beginiing of file
           self.light_count = 0
           self.days_repeat += 1
           self.time = self.time.replace(hour=self.time_begin.hour, minute=self.time_begin.minute, second=self.time_begin.second)
           self.end = self.time + datetime.timedelta(hours=24*1)

            #self.file_data = Pible_func.randomize_light_time(self.file_data_orig)

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        return (self.hour, self.light_ar, self.SC_volt, self.week_end)

    def step(self, action):
        assert action[0] in [0, 1], action
        #self.time = self.time + datetime.timedelta(0, 60*int(self.next_wake_up_time)) #next_wake_up_time # in min
        PIR_on_off = action[0]
        # adjust next wake up time from 0 to 1 to 1 to 60
        self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action[1]-s_t_max_act)+s_t_max_new)

        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        self.hour = np.roll(self.hour, 1)
        self.hour[0] = self.time.hour

        self.minute = np.roll(self.minute, 1)
        self.minute[0] = self.time.minute

        self.light = self.light_ar[0]
        self.light_ar = np.roll(self.light_ar, 1)

        SC_temp = self.SC_volt[0]
        self.SC_volt = np.roll(self.SC_volt, 1)

        #self.hour = np.array([self.time.hour])
        #self.minute = np.array([self.time.minute])

        #print(self.light_count, "light count")
        #sleep(1)
        self.light, self.light_count, event = Pible_func.light_event_func(self.time, self.time_next, 0, [], self.light_count, self.light, self.file_data, self.days_repeat, self.diff_days, self.light_div)
        SC_temp, en_prod, en_used = Pible_func.Energy(SC_temp, self.light, PIR_on_off, self.next_wake_up_time, event)
        reward, event_det, event_miss, self.death_days, self.death_min = Pible_func.reward_func_low_level(PIR_on_off, event, SC_temp, self.death_days, self.death_min, self.next_wake_up_time)
        self.light_ar[0] = self.light
        self.SC_volt[0] = SC_temp

        if reward > 0:
            self.events_detect += event
        self.tot_events += event

        done = self.time >= self.end

        if self.train != "train":
            self.SC_Volt.append(SC_temp); self.Reward.append(reward); self.PIR_hist.append(event); self.Time.append(self.time); self.Light.append(self.light); self.PIR_OnOff.append(action[0]); self.State_Trans.append(self.next_wake_up_time); self.event_det_hist.append(event_det); self.event_miss_hist.append(event_miss)

        if done: # one day is over
            self.end = self.time + datetime.timedelta(hours=24*1)
        else:
            self.time = self.time_next #next_wake_up_time # in min

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        #self.light_ar = np.array([self.light])

        info = {}
        info["Energy_used"] = en_used
        info["Energy_prod"] = en_prod
        info["Tot_events"] = self.tot_events
        info["events_detect"] = self.events_detect
        info["Death_days"] = self.death_days
        info["Death_min"] = self.death_min
        info["SC_volt"] = SC_temp
        return (self.hour, self.light_ar, self.SC_volt, self.week_end), reward, done, info

    def render(self, episode, tot_rew, start, end, title):
        if start != "": # If you want to print to a particular hours of the day
            start_pic_detail = datetime.datetime.strptime(start, '%m/%d/%y %H:%M:%S')
            end_pic_detail = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S')
            Time = []; Light = []; PIR_OnOff = []; State_Trans = []; Reward = []; Perf = []; SC_Volt = []; PIR_hist = []; PIR_det = []; PIR_miss = []
            for i in range(0, len(self.Time)):
                line = self.Time[i]
                if start_pic_detail <= line and line <= end_pic_detail:
                    Time.append(self.Time[i]); Light.append(self.Light[i]); PIR_OnOff.append(self.PIR_OnOff[i]); State_Trans.append(self.State_Trans[i]); Reward.append(self.Reward[i]); SC_Volt.append(self.SC_Volt[i]); PIR_hist.append(self.PIR_hist[i]); PIR_det.append(self.event_det_hist[i]); PIR_miss.append(self.event_miss_hist[i])
            Pible_func.plot_hist(Time, Light, PIR_OnOff, State_Trans, Reward, self.Perf, SC_Volt, PIR_hist, PIR_det, PIR_miss, episode, tot_rew, self.events_detect, self.tot_events, title)
        else:
            Pible_func.plot_hist(self.Time, self.Light, self.PIR_OnOff, self.State_Trans, self.Reward, self.Perf, self.SC_Volt, self.PIR_hist, self.event_det_hist, self.event_miss_hist, episode, tot_rew, self.events_detect, self.tot_events, title)

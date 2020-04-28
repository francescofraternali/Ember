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

num_hours_input = 1; num_minutes_input = 1; num_volt_input = 1; num_light_input = 1; num_week_input = 1;  num_events_input = 1

s_t_min_act = 0; s_t_max_act = 1; s_t_min_new = 1; s_t_max_new = 60

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        settings = config["settings"]
        path_light = settings[0]["path_light_data"]
        self.light_div = float(settings[0]["light_divider"])
        self.main_path = config["main_path"]
        self.path_light_data = self.main_path + "/" + path_light
        self.train = config["train/test"]
        '''
        self.train = config["train/test"]
        if self.train == "train":
            start_data_date = datetime.datetime.strptime(config["start_train"], '%m/%d/%y %H:%M:%S')
            end_data_date = datetime.datetime.strptime(config["end_train"], '%m/%d/%y %H:%M:%S')
            self.start_sc = config["sc_volt_start_train"]
        else:
            start_data_date = datetime.datetime.strptime(config["start_test"], '%m/%d/%y %H:%M:%S')
            end_data_date = datetime.datetime.strptime(config["end_test"], '%m/%d/%y %H:%M:%S')
            self.start_sc = config["sc_volt_start_test"]
        '''
        start_data_date = datetime.datetime.strptime("10/7/19 00:00:00", '%m/%d/%y %H:%M:%S')
        end_data_date = datetime.datetime.strptime("10/14/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.start_sc = '3.2'

        self.start = start_data_date
        self.end = end_data_date

        diff_date = end_data_date - start_data_date
        self.diff_days = diff_date.days

        self.file_data = []
        self.events_found_dict = []
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
        self.light = int(int(line[8])/self.light_div)
        self.light_count = 0
        self.death_days = 0
        self.death_min = 0
        self.days_repeat = 0
        #self.time = datetime.datetime.strptime('01/01/17 00:00:00', '%m/%d/%y %H:%M:%S')
        self.time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        self.time_begin = self.time
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.lock = 0

        self.action_space = spaces.Discrete(3)
        #self.action_space = spaces.Tuple((
        #    spaces.Discrete(3), # 0: Sleep; 1: Ground Truth Mode (GT); 2 Normal RL Mode (NormRL)
        #    #spaces.Box(s_t_min_act, s_t_max_act, shape=(1, ), dtype=np.float32) # State Transition
        #))
        #min = np.array([0, SC_volt_min, 0, 0], dtype=np.float32)
        #max = np.array([23, 5, 1, 10], dtype=np.float32)
        #self.observation_space = spaces.Box(-max, max, dtype=np.float32)
        #self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            #spaces.Box(SC_volt_min, SC_volt_max, shape=(num_volt_input, ), dtype=np.float32),
            #spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
            spaces.Box(0, 10, shape=(1, ), dtype=np.float32),  #number of events

        ))

        self.SC_Volt = []; self.Reward = []; self.Mode = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []; self.Len_Dict_Events = []
        self.tot_events = 0; self.events_detect = 0
        self.event_det_hist = []; self.event_miss_hist = []

        self.hour, self.minute, self.light_ar = Pible_func.build_inputs(self.time, num_hours_input, num_minutes_input, num_light_input, self.light)

    def reset(self):
        '''
        if self.episode_count % 4000 == 0:
            if  self.start_sc == "rand":
                self.SC_rand = random.uniform(SC_volt_die, 4)
            else:
                self.SC_rand = float(self.start_sc)

            self.SC_volt = np.array([self.SC_rand] * num_volt_input)
        '''

        self.SC_volt = np.array([3.2] * num_volt_input)
        self.episode_count += 1
        self.death_days = 0
        self.death_min = 0

        '''
        if self.light_len == self.light_count: # the file with light has been all read, reset light and time from beginiing of file
           self.light_count = 0
           self.days_repeat += 1
           #self.time = self.time.replace(hour=self.time_begin.hour, minute=self.time_begin.minute, second=self.time_begin.second)
        '''
        self.time = self.time_begin
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
           #self.file_data = Pible_func.randomize_light_time(self.file_data_orig)
        self.events_found_dict = []

        len_dict_event = np.array([len(self.events_found_dict)])

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        #self.SC_Volt = []; self.Reward = []; self.Mode = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []; self.Len_Dict_Events = []; self.event_det_hist = []; self.event_miss_hist = []

        #return np.array([self.time.hour, self.SC_volt, 1, len_dict_event])
        return (self.hour, len_dict_event)

    def step(self, action):
        #assert action[0] in [0, 1, 2], action
        #if action[0] == 2:
        #    action[0] = 1
        #action[0] = 2
        #print(self.time)
        #self.lock = 0
        #action[0] = 2
        # adjust next wake up time from 0 to 1 to 1 to 6
        #self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action[1]-s_t_max_act)+s_t_max_new)
        #self.time = self.time + datetime.timedelta(0, 60*int(self.next_wake_up_time)) #next_wake_up_time # in min
        self.next_wake_up_time = 60
        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min
        if self.SC_volt[0] <= SC_volt_die:
            action = 0

        self.mode = action

        if self.mode == 0:
            self.PIR_on_off = 0
        elif self.mode == 1:
            self.PIR_on_off = 1
        elif self.mode == 2:
            event_found = 0
            for time_checker in self.events_found_dict:
                if self.time.time() <= time_checker and time_checker < self.time_next.time():
                    event_found = 1
                    self.PIR_on_off = 1
                    break
            if event_found == 0:
                self.PIR_on_off = 0

        '''
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
        '''
            #low_level_agent(event_detected_list)

            #cmd = ['python', '/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Ember/HRL/High_level_central/low_level_agent.py']
            #subprocess.Popen(cmd).wait()
            #print("back to high level agent")
            #sleep(20)

        #self.lock = 1

        self.hour = np.roll(self.hour, 1)
        self.hour[0] = self.time.hour

        self.minute = np.roll(self.minute, 1)
        self.minute[0] = self.time.minute

        self.light = self.light_ar[0]
        self.light_ar = np.roll(self.light_ar, 1)

        SC_temp = self.SC_volt[0]
        self.SC_volt = np.roll(self.SC_volt, 1)

        self.light, event_gt, self.events_found_dict = Pible_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.events_found_dict, self.light, self.file_data)

        SC_temp, en_prod, en_used = Pible_func.Energy(SC_temp, self.light, self.PIR_on_off, self.next_wake_up_time, event_gt)

        reward, event_det, event_miss, self.death_days, self.death_min = Pible_func.reward_func_high_level(self.PIR_on_off, self.mode, event_gt, SC_temp, self.death_days, self.death_min, self.next_wake_up_time)
        self.light_ar[0] = self.light
        self.SC_volt[0] = SC_temp

        if reward > 0:
            self.events_detect += event_gt
        self.tot_events += event_gt

        done = self.time >= self.end

        if len(self.events_found_dict) > 2000:
            len_dict_event = np.array([2000])
        else:
            len_dict_event = np.array([len(self.events_found_dict)])

        if self.train != "train":
            self.SC_Volt.append(SC_temp); self.Reward.append(reward); self.PIR_hist.append(event_gt); self.Time.append(self.time); self.Light.append(self.light); self.Mode.append(self.mode); self.PIR_OnOff.append(self.PIR_on_off); self.State_Trans.append(self.next_wake_up_time); self.event_det_hist.append(event_det); self.event_miss_hist.append(event_miss); self.Len_Dict_Events.append(len_dict_event)

        if done: # one day is over
            #self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
            #print(self.episode_count)
            #if self.episode_count % 50 == 0:
            #    self.render(1, 0, self.start, self.end, "title_final")
            self.lock = 0
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

        return (self.hour, len_dict_event), reward, done, info

    def render(self, episode, tot_rew, start, end, title):
        Pible_func.plot_hist(self.Time, self.Light, self.Mode, self.PIR_OnOff, self.State_Trans, self.Reward, self.Perf, self.SC_Volt, self.PIR_hist, self.event_det_hist, self.event_miss_hist, episode, tot_rew, self.events_detect, self.tot_events, self.Len_Dict_Events, title)

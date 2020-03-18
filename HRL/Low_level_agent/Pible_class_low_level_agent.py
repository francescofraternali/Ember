import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym import spaces, logger
import datetime
from Pible_parameters import *
import Pible_func
import random
from time import sleep

num_hours_input = 24; #IMP leave this 24 or modify build_inputs
num_light_input = 24; num_sc_volt_input = 24; num_week_input = 7

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):

        settings = config["settings"]
        path_light = settings[0]["path_light_data"]
        self.light_div = float(settings[0]["light_divider"])
        self.path_light_data = config["main_path"] + "/" + path_light
        self.train = config["train/test"]

        if self.train == "train":
            start_data_date = datetime.datetime.strptime(config["start_train"], '%m/%d/%y %H:%M:%S')
            end_data_date = datetime.datetime.strptime(config["end_train"], '%m/%d/%y %H:%M:%S')
            self.start_sc = config["sc_volt_start_train"]
        else:
            start_data_date = datetime.datetime.strptime(config["start_test"], '%m/%d/%y %H:%M:%S')
            end_data_date = datetime.datetime.strptime(config["end_test"], '%m/%d/%y %H:%M:%S')
            self.start_sc = config["sc_volt_start_test"]

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
        line = self.file_data[0].split('|')
        self.light = int(int(line[8])/self.light_div)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_sc_volt_input, ), dtype=np.float32),
            spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
            #spaces.Box(0, 10, shape=(1, ), dtype=np.float32),  #number of events
        ))

        self.Reward = []; self.Mode = []; self.Time = []; self.Light = []; self.PIR_OnOff_hist = []; self.SC_Volt = []; self.State_Trans = []
        self.event_det_hist = []; self.event_miss_hist = []; self.Len_Dict_Events = []; self.tot_events_detect = 0; self.tot_events = 0;

        self.hour_array, self.light_array, self.SC_Volt_array = Pible_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_light_input, num_sc_volt_input)

    def reset(self):
        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.events_found_dict = []
        #self.hour = np.array([self.time.hour] * num_hours_input)
        self.hour_array, self.light_array, self.SC_Volt_array = Pible_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_light_input, num_sc_volt_input)

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        return self.hour_array, self.SC_Volt_array, self.week_end

    def step(self, action):

        self.mode = action
        self.next_wake_up_time = 60
        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        if self.mode == 0 or self.SC_Volt_array[0] < SC_volt_die:
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

        self.light, event_gt, self.events_found_dict = Pible_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.events_found_dict, self.light, self.light_div, self.file_data)

        SC_temp, en_prod, en_used = Pible_func.Energy(self.SC_Volt_array[0], self.light, self.PIR_on_off, self.next_wake_up_time, event_gt)

        reward, event_det, event_miss = Pible_func.reward_func_high_level(self.mode, event_gt, self.PIR_on_off, self.SC_Volt_array)

        len_dict_event = np.array([len(self.events_found_dict)])
        if self.train == 'test':
            self.Reward.append(reward); self.Time.append(self.time); self.Mode.append(self.mode); self.Light.append(self.light); self.PIR_OnOff_hist.append(self.PIR_on_off); self.SC_Volt.append(SC_temp);
            self.State_Trans.append(self.next_wake_up_time); self.event_det_hist.append(event_det); self.event_miss_hist.append(event_miss); self.Len_Dict_Events.append(len_dict_event)

        self.tot_events_detect += event_det
        self.tot_events += event_gt

        self.hour_array, self.light_array, self.SC_Volt_array = Pible_func.updates_arrays(self.hour_array, self.light_array, self.SC_Volt_array, self.time, self.light, SC_temp)

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        self.time = self.time_next

        done = self.time >= self.end

        return (self.hour_array, self.SC_Volt_array, self.week_end), reward, done, {}

    def render(self, tot_rew, title):
        Pible_func.plot_hist(self.Time, self.Light, self.Mode, self.PIR_OnOff_hist, self.State_Trans, self.Reward, self.SC_Volt, self.event_det_hist, self.event_miss_hist, tot_rew, self.tot_events_detect, self.tot_events, self.Len_Dict_Events, title)
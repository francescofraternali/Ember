import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym import spaces, logger
import datetime
from Pible_param_func import *
import Pible_param_func
import Ember_RL_func
import random
from time import sleep

num_hours_input = 24; num_minutes_input = 24 #IMP leave this 24 or modify build_inputs
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
        self.PIR_events_found_dict = []

        self.file_data = []
        with open(self.path_light_data, 'r') as f:
            for line in f:
                line_split = line.split("|")
                checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
                if start_data_date <= checker and checker <= end_data_date:
                    self.file_data.append(line)
        line = self.file_data[0].split('|')
        self.light = int(int(line[8])/self.light_div)

        self.action_space = spaces.Tuple((
            spaces.Discrete(2), # Num of Nodes
            spaces.Discrete(2), # Modes 0 Off and 1 Normal RL
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_sc_volt_input, ), dtype=np.float32),
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_sc_volt_input, ), dtype=np.float32),
            spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
            #spaces.Box(0, 10, shape=(1, ), dtype=np.float32),  #number of events
        ))

        self.Reward = []; self.Mode = []; self.Time = []; self.Light = []; self.PIR_OnOff_hist = []; self.SC_Volt_1 = []; self.SC_Volt_2 = []; self.State_Trans = [];
        self.event_det_hist = []; self.event_miss_hist = []; self.Len_Dict_Events = []; self.PIR_tot_events_detect = 0; self.PIR_tot_events = 0; self.Node_Select = []

        self.hour_array, null, self.light_array, self.SC_Volt_array = Ember_RL_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_minutes_input, num_light_input, num_sc_volt_input)
        self.SC_Volt_array_1 = self.SC_Volt_array; self.SC_Volt_array_2 = self.SC_Volt_array

    def reset(self):
        self.time = datetime.datetime.strptime("10/07/19 00:00:00", '%m/%d/%y %H:%M:%S')
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.PIR_events_found_dict = []
        #self.hour = np.array([self.time.hour] * num_hours_input)
        self.hour_array, null, self.light_array, self.SC_Volt_array = Ember_RL_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_minutes_input, num_light_input, num_sc_volt_input)
        self.SC_Volt_array_1 = self.SC_Volt_array; self.SC_Volt_array_2 = self.SC_Volt_array

        self.week_end = Ember_RL_func.calc_week(self.time, num_week_input)

        return self.hour_array, self.SC_Volt_array_1, self.SC_Volt_array_2, self.week_end

    def step(self, action):

        self.node_select = action[0]
        self.mode_orig = action[1]
        #self.node_select = 0
        #self.mode_orig = 1

        self.next_wake_up_time = 60
        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        if self.node_select == 0:
             #self.PIR_on_off_1 = 1; self.PIR_on_off_2 = 0; self.thpl_on_off_1 = 1; self.thpl_on_off_2 = 0; self.mode_1 = 0; self.mode_1 = 0
            self.mode_1 = self.mode_orig; self.mode_2 = 0
        elif self.node_select == 1:
            #self.PIR_on_off_1 = 0; self.PIR_on_off_2 = 1 self.thpl_on_off_1 = 0; self.thpl_on_off_2 = 1; self.mode_1 = 0
            self.mode_1 = 0; self.mode_2 = self.mode_orig

        PIR_temp = 0
        reward = 0
        PIR_event_det = 0
        for i in range(2):
            if i == 0:
                self.SC_Volt_array = self.SC_Volt_array_1; self.mode = self.mode_1
            elif i == 1:
                self.SC_Volt_array = self.SC_Volt_array_2; self.mode = self.mode_2

            if self.mode == 0 or self.SC_Volt_array[0] < SC_volt_die:
                self.PIR_on_off = 0
                self.thpl_on_off = 0
            elif self.mode == 1:
                self.PIR_on_off = 1
                self.thpl_on_off = 1

            '''
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
            '''

            #self.light, event_gt, self.events_found_dict = Pible_param_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.events_found_dict, self.light, self.light_div, self.file_data)

            self.light, PIR_event_gt, self.PIR_events_found_dict, thpl_event_gt = Pible_param_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.PIR_events_found_dict, self.light, self.light_div, self.file_data)

            PIR_event_det_temp, PIR_event_miss_temp, thpl_event_det, thpl_event_miss = Pible_param_func.event_det_miss(PIR_event_gt, thpl_event_gt, self.PIR_on_off, self.thpl_on_off, self.SC_Volt_array)

            SC_temp, en_prod, en_used = Pible_param_func.Energy(self.SC_Volt_array[0], self.light, self.PIR_on_off, self.thpl_on_off, self.next_wake_up_time, PIR_event_det_temp, thpl_event_det)

            reward_temp = Ember_RL_func.reward_func_meta_level(self.mode, PIR_event_det_temp, PIR_event_miss_temp, thpl_event_det, thpl_event_miss, self.PIR_on_off, self.thpl_on_off, self.SC_Volt_array)

            if i == 0:
                self.SC_Volt_array_1 = self.SC_Volt_array; SC_temp_1 = SC_temp
            elif i == 1:
                self.SC_Volt_array_2 = self.SC_Volt_array; SC_temp_2 = SC_temp

            PIR_temp += self.PIR_on_off
            reward += reward_temp
            PIR_event_det += PIR_event_det_temp

        PIR_event_miss = PIR_event_gt - PIR_event_det

        len_dict_event = np.array([len(self.PIR_events_found_dict)])
        if self.train == 'test':
            self.Reward.append(reward); self.Time.append(self.time); self.Mode.append(self.mode_orig); self.Light.append(self.light); self.PIR_OnOff_hist.append(PIR_temp); self.SC_Volt_1.append(SC_temp_1); self.SC_Volt_2.append(SC_temp_2);
            self.State_Trans.append(self.next_wake_up_time); self.event_det_hist.append(PIR_event_det); self.event_miss_hist.append(PIR_event_miss); self.Len_Dict_Events.append(len_dict_event); self.Node_Select.append(self.node_select)

        self.PIR_tot_events_detect += PIR_event_det
        self.PIR_tot_events += PIR_event_gt

        self.hour_array, null, self.light_array, self.SC_Volt_array_1, self.SC_Volt_array_2 = Ember_RL_func.updates_arrays(self.hour_array, [0], self.light_array, self.SC_Volt_array_1, self.SC_Volt_array_2, self.time, self.light, SC_temp_1, SC_temp_2)

        self.week_end = Ember_RL_func.calc_week(self.time, num_week_input)

        self.time = self.time_next

        done = self.time >= self.end

        info = {}
        info["energy_used"] = en_used
        info["energy_prod"] = en_prod
        info["PIR_tot_events"] = self.PIR_tot_events
        info["PIR_events_detect"] = self.PIR_tot_events_detect
        #info["death_days"] = self.death_days
        #info["death_min"] = self.death_min
        #info["SC_volt"] = SC_temp

        return (self.hour_array, self.SC_Volt_array_1, self.SC_Volt_array_2, self.week_end), reward, done, info

    def render(self, tot_rew, title):
        Ember_RL_func.plot_meta_RL(self.Time, self.Light, self.Mode, self.Node_Select, self.PIR_OnOff_hist, self.State_Trans, self.Reward, self.SC_Volt_1, self.SC_Volt_2, self.event_det_hist, self.event_miss_hist, tot_rew, self.PIR_tot_events_detect, self.PIR_tot_events, self.Len_Dict_Events, title)

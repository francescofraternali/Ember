import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym import spaces, logger
import datetime
from Pible_parameters import *
import Pible_func
import random
from time import sleep

num_hours_input = 24; num_minutes_input = 60
num_light_input = 24; num_sc_volt_input = 24; num_week_input = 7
s_t_min_act = 0; s_t_max_act = 1; s_t_min_new = 1; s_t_max_new = 60

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
        self.time_begin = self.time
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
        self.temp = int(line[2])
        self.hum = int(line[3])
        self.press = int(line[11])

        self.action_space = spaces.Tuple((
            spaces.Discrete(4), # PIR On_Off and Sens_On Off: 0 PIR and sens Off; 1 PIR On and senss Off; PIR Off ans Sens On; PIR and Sens On
            spaces.Box(s_t_min_act, s_t_max_act, shape=(1, ), dtype=np.float32) # State Transition
        ))
        #self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 59, shape=(num_minutes_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_sc_volt_input, ), dtype=np.float32),
            #spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
            #spaces.Box(0, 10, shape=(1, ), dtype=np.float32),  #number of events
        ))

        self.Reward = []; self.Mode = []; self.Time = []; self.Light = []; self.Sens_On_Off = []; self.SC_Volt = []; self.State_Trans = []; self.thpl_event_det_hist = []; self.thpl_event_miss_hist = []
        self.PIR_event_det_hist = []; self.PIR_event_miss_hist = []; self.Len_Dict_Events = []; self.PIR_tot_events_detect = 0; self.thpl_tot_events = 0; self.thpl_tot_events_detect = 0; self.PIR_tot_events = 0;  self.mode = 0

        self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Pible_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_minutes_input, num_light_input, num_sc_volt_input)

    def reset(self):
        self.time = self.time_begin
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.PIR_events_found_dict = []
        #self.hour = np.array([self.time.hour] * num_hours_input)
        self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Pible_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_minutes_input, num_light_input, num_sc_volt_input)

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        #return self.hour_array, self.minute_array, self.SC_Volt_array, self.week_end
        return self.hour_array, self.SC_Volt_array

    def step(self, action):
        #print("action is: ", action)
        sens_on_off = action[0]
        self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action[1]-s_t_max_act)+s_t_max_new)
        #self.next_wake_up_time = 60

        '''
        if self.time.hour == 8:
            sens_on_off = 3
        elif self.time.hour == 18:
            sens_on_off = 2
        elif self.time.hour == 10 or self.time.hour == 17:
            sens_on_off = 1
        else:
            sens_on_off = 0
        self.next_wake_up_time = 60
        '''
        '''# optimal solution
        if self.time.minute == 0 and (self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17 or self.time.hour == 18):
            sens_on_off = 1
            self.next_wake_up_time = 1
            if self.time.hour == 18:
                sens_on_off = 2
        #elif (self.time.minute == 1 or self.time.minute == 11 or self.time.minute == 21 or self.time.minute == 31 or self.time.minute == 41 or self.time.minute == 51) and (self.time.hour == 8 or self.time.hour == 18):
        elif (self.time.minute == 1) and (self.time.hour == 8):
            sens_on_off = 0
            self.next_wake_up_time = 9
        #elif (self.time.minute == 10 or self.time.minute == 20 or self.time.minute == 30 or self.time.minute == 40 or self.time.minute == 50) and (self.time.hour == 8 or self.time.hour == 18):
        elif (self.time.minute == 10) and (self.time.hour == 8):
            sens_on_off = 3
            self.next_wake_up_time = 1
        elif (self.time.minute == 11) and (self.time.hour == 8):
            sens_on_off = 0
            self.next_wake_up_time = 49
        elif self.time.minute == 0 and (self.time.hour == 18):
            sens_on_off = 3
            self.next_wake_up_time = 60
        #elif self.time.minute == 1 and (self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17):
        elif self.time.minute == 1 and (self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17 or self.time.hour == 18):
            sens_on_off = 0
            self.next_wake_up_time = 59
        else:
            sens_on_off = 0
            self.next_wake_up_time = 60
        '''# ending optimal solution


        if sens_on_off == 0:
            self.PIR_on_off = 0; self.thpl_on_off = 0
        elif sens_on_off == 1:
            self.PIR_on_off = 1; self.thpl_on_off = 0
        elif sens_on_off == 2:
            self.PIR_on_off = 0; self.thpl_on_off = 1
        elif sens_on_off == 3:
            self.PIR_on_off = 1; self.thpl_on_off = 1

        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        self.light, PIR_event_gt, PIR_events_found_dict, thpl_event_gt = Pible_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.PIR_events_found_dict, self.light, self.light_div, self.file_data)
        #print(self.time, PIR_event_gt, thpl_event_gt)
        PIR_event_det, PIR_event_miss, thpl_event_det, thpl_event_miss = Pible_func.event_det_miss(PIR_event_gt, thpl_event_gt, self.PIR_on_off, self.thpl_on_off, self.SC_Volt_array)

        SC_temp, en_prod, en_used = Pible_func.Energy(self.SC_Volt_array[0], self.light, self.PIR_on_off, self.thpl_on_off, self.next_wake_up_time, PIR_event_det, thpl_event_det)

        #reward = Pible_func.reward_func_low_level(self.mode, PIR_event_det, PIR_event_miss, thpl_event_det, thpl_event_miss, self.PIR_on_off, self.thpl_on_off, self.SC_Volt_array)

        reward = Pible_func.reward_new(en_prod, en_used, PIR_event_det, PIR_event_miss, thpl_event_det, thpl_event_miss, self.PIR_on_off, self.thpl_on_off, self.SC_Volt_array)

        len_dict_event = np.array([len(self.PIR_events_found_dict)])
        if self.train == 'test':
            self.Reward.append(reward); self.Time.append(self.time); self.Mode.append(self.mode); self.Light.append(self.light); self.Sens_On_Off.append(sens_on_off); self.SC_Volt.append(SC_temp);
            self.State_Trans.append(self.next_wake_up_time); self.PIR_event_det_hist.append(PIR_event_det); self.PIR_event_miss_hist.append(PIR_event_miss); self.Len_Dict_Events.append(len_dict_event)
            self.thpl_event_det_hist.append(thpl_event_det); self.thpl_event_miss_hist.append(thpl_event_miss);

        self.PIR_tot_events_detect += PIR_event_det
        self.PIR_tot_events += PIR_event_gt
        self.thpl_tot_events_detect += thpl_event_det
        self.thpl_tot_events += thpl_event_gt

        #print('action', action)
        #print("before", self.hour_array, self.minute_array, self.SC_Volt_array)

        #self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Pible_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_minutes_input, num_light_input, num_sc_volt_input)

        self.time = self.time_next

        self.week_end = Pible_func.calc_week(self.time, num_week_input)

        self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Pible_func.updates_arrays(self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array, self.time, self.light, SC_temp)

        #print("after", self.hour_array, self.minute_array, self.SC_Volt_array)

        done = self.time >= self.end

        info = {}
        info["energy_used"] = en_used
        info["energy_prod"] = en_prod
        info["PIR_tot_events"] = self.PIR_tot_events
        info["PIR_events_detect"] = self.PIR_tot_events_detect
        info["thpl_tot_events"] = self.thpl_tot_events
        info["thpl_events_detect"] = self.thpl_tot_events_detect
        #info["death_days"] = self.death_days
        #info["death_min"] = self.death_min
        info["SC_volt"] = SC_temp

        #return (self.hour_array, self.minute_array, self.SC_Volt_array, self.week_end), reward, done, info
        return (self.hour_array, self.SC_Volt_array), reward, done, info

    def render(self, tot_rew, title, energy_used):
        for i in range(len(self.thpl_event_miss_hist)):
            if self.thpl_event_miss_hist[i] == 0:
                self.thpl_event_miss_hist[i] = np.nan
            if self.PIR_event_miss_hist[i] == 0:
                self.PIR_event_miss_hist[i] = np.nan
        Pible_func.plot_hist_low_level(self.Time, self.Light, self.Mode, self.Sens_On_Off, self.State_Trans,
                                       self.Reward, self.SC_Volt, self.PIR_event_det_hist, self.PIR_event_miss_hist,
                                       self.thpl_event_det_hist, self.thpl_event_miss_hist, tot_rew, self.PIR_tot_events_detect,
                                       self.PIR_tot_events, self.Len_Dict_Events, title, energy_used)

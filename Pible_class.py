import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym import spaces, logger
import datetime
from Pible_param_func import *
import Pible_param_func
import Ember_RL_func
from time import sleep
import os
import pickle
import random

num_hours_input = 24; num_minutes_input = 60
num_light_input = 24; num_sc_volt_input = 7; num_week_input = 7
s_t_min_act = 0; s_t_max_act = 1; s_t_min_new = 1; s_t_max_new = 60

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):

        settings = config["settings"]
        self.config = config
        file_light = settings[0]["file_light"]
        self.light_div = float(settings[0]["light_divider"])
        self.PIR_or_thpl = settings[0]["PIR_or_thpl"]
        self.path_light_data = config["main_path"] + "/" + file_light
        self.train = config["train/test"]
        self.GT_mode = settings[0]["GT_mode"]
        self.diff_days = config["diff_days"]
        self.rem_miss = settings[0]["train_gt/rm_miss/inject"]
        #print(config["main_path"], self.path_light_data)

        if self.train == "train":
            #start_data_date = datetime.datetime.strptime(config["start_train"], '%m/%d/%y %H:%M:%S')
            self.start_data_date = config["start_train"]
            #self.end_data_date = datetime.datetime.strptime(config["end_train"], '%m/%d/%y %H:%M:%S')
            self.start_sc = float(config["sc_volt_start_train"])
            self.end_data_date = config["end_train"]
        elif self.train == "test":
            self.start_data_date = config["start_test"]
            self.end_data_date = config["end_test"]
            if isinstance(config["sc_volt_start_test"], np.ndarray):
                self.start_sc = config["sc_volt_start_test"]
            else:
                self.start_sc = float(config["sc_volt_start_test"])
        elif self.train == "real":
            Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], file_light, "")
            #sleep(10)
            #last_row = Ember_RL_func.last_valid_row(file_light)
            self.start_data_date = datetime.datetime.now()
            self.end_data_date = datetime.datetime.now() + datetime.timedelta(days=1)

        #self.time = datetime.datetime.strptime("04/15/20 00:00:00", '%m/%d/%y %H:%M:%S')
        self.time = self.start_data_date
        self.time_begin = self.time
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)
        self.data_pointer = 0
        self.PIR_events_found_dict = []

        #print("Starting looking for data in between: ", self.start_data_date, self.end_data_date)
        self.start_light_list, self.start_sc_volt_list, starter_data = Ember_RL_func.select_input_starter(self.path_light_data, self.start_data_date, num_light_input, num_sc_volt_input)

        if self.train == "test" and self.start_sc != '':
            self.start_sc_volt_list = Ember_RL_func.adjust_sc_voltage(self.start_sc_volt_list, self.start_sc)

        self.file_data = Ember_RL_func.select_input_data(self.path_light_data, self.start_data_date, self.end_data_date)
        if self.train == 'test':
            if self.file_data == []:
                print("\nInput data error or something else. Check!\n")
                exit()
        #for value in self.file_data:
        #    print(self.data_pointer, value)
        #    sleep(5)
        #self.data_pointer_orig = self.data_pointer
        #print(starter_data)
        #print(starter_data)
        self.last = starter_data[0]
        splitt = self.last.split('|')
        #self.start_sc = (float(splitt[5]) * SC_volt_max)/100
        #print("last line found: ", self.last)

        self.light = int(int(splitt[8])/self.light_div)
        #self.temp = float(line[2])
        #self.hum = float(line[3])
        #self.press = int(line[11])

        if self.PIR_or_thpl == '2':
            self.action_space = spaces.Tuple((
                spaces.Discrete(2), # PIR On_Off = 0 means PIR 0; PIR_onoff = 1 means PIR On
                spaces.Discrete(2), # Sens_On Off same as PIR
                #spaces.Box(s_t_min_act, s_t_max_act, shape=(1, ), dtype=np.float32) # State Transition
            ))
        else:
            self.action_space = spaces.Tuple((
                spaces.Discrete(2),
                #spaces.Box(s_t_min_act, s_t_max_act, shape=(1, ), dtype=np.float32) # State Transition
            ))
        #self.action_space = spaces.Discrete(2)
        if self.diff_days >= 7:
            self.observation_space = spaces.Tuple((
                spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
                #spaces.Box(0, 59, shape=(num_minutes_input, ), dtype=np.float32),       # hours
                spaces.Box(0, light_max, shape=(num_light_input, ), dtype=np.float32),       # light
                spaces.Box(SC_volt_min, SC_volt_max, shape=(num_sc_volt_input, ), dtype=np.float32),
                spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
                #spaces.Box(0, 10, shape=(1, ), dtype=np.float32),  #number of events
            ))
        else:
            self.observation_space = spaces.Tuple((
                spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
                #spaces.Box(0, 59, shape=(num_minutes_input, ), dtype=np.float32),       # hours
                spaces.Box(0, light_max, shape=(num_light_input, ), dtype=np.float32),       # light
                spaces.Box(SC_volt_min, SC_volt_max, shape=(num_sc_volt_input, ), dtype=np.float32),
                #spaces.Box(0, 1, shape=(num_week_input, ), dtype=np.float32),  #week/weekends
                #spaces.Box(0, 10, shape=(1, ), dtype=np.float32),  #number of events
            ))

        self.Reward = []; self.Mode = []; self.Time = []; self.Light = []; self.PIR_ON_OFF= []; self.THPL_ON_OFF = []; self.SC_Volt = []; self.State_Trans = []; self.thpl_event_det = []; self.thpl_event_miss = []
        self.PIR_event_det = []; self.PIR_event_miss = []; self.Len_Dict_Events = []; self.PIR_tot_events_detect = 0; self.thpl_tot_events = 0; self.thpl_tot_events_detect = 0; self.PIR_tot_events = 0;  self.mode = 0
        self.PIR_gt = []; self.THPL_gt = []

        self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Ember_RL_func.build_inputs(self.time, num_hours_input, num_minutes_input, self.start_light_list, self.start_sc_volt_list)

        #if self.GT_mode == '1' and self.train != 'train':
        self.GT_hour_start = int(config["GT_hour_start"])
        self.gt_hours = Ember_RL_func.gt_mode_hours((self.SC_Volt_array[0]/SC_volt_max)*100)

        # injecting problems
        self.hours_inj_prob = random.randint(0, 23)

        self.info = {"config": config}
        self.save_data()

    def reset(self):
        #print(self.time)
        if self.time >= self.end_data_date and self.train == "train":
            #print("reset time and light file accordingly")
            self.time = self.time_begin
            self.data_pointer = 0
            self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Ember_RL_func.build_inputs(self.time, num_hours_input, num_minutes_input, self.start_light_list, self.start_sc_volt_list)
            self.SC_Volt_array = Ember_RL_func.add_random_volt(self.SC_Volt_array)

            #self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Ember_RL_func.build_inputs(self.time, self.light, self.start_sc, num_hours_input, num_minutes_input, last_light_list, last_volt_list)
        self.week_end = Ember_RL_func.calc_week(self.time, num_week_input)
        self.end = self.time + datetime.timedelta(hours=24*episode_lenght)

        if self.diff_days >= 7:
            return self.hour_array, self.light_array, self.SC_Volt_array, self.week_end
        else:
            return self.hour_array, self.light_array, self.SC_Volt_array

    def step(self, action):
        #print("action ", action)
        #action = [1, 0.5]
        #if isinstance(action, list):
        if len(action) == 1:
            action_0 = action[0]
        elif len(action) == 2:
            action_0 = action[0]
            action_1 = action[1]
        #elif len(action) == 3:
        #    action_0 = action[0]
        #    action_1 = action[1]
        #    action_2 = action[2]

        if self.rem_miss == '2' and self.train == 'test': # injecting problems
            if self.time.hour >= self.hours_inj_prob and self.time.hour < (self.hours_inj_prob + 1):
                action_0 = 0
                if len(action) == 2:
                    action_1 = 0
                #print("injecting problems from to: ", self.hours_inj_prob, self.hours_inj_prob + 1)
                #sleep(10)

        PIR_gt = np.nan; THPL_gt = np.nan
        if self.GT_mode == '1' and self.train != 'train':
            #self.gt_hours = Ember_RL_func.gt_mode_hours((self.SC_Volt_array[0]/SC_volt_max)*100)
            print(self.time.hour, self.GT_hour_start, self.gt_hours, str(self.GT_hour_start + self.gt_hours))
            if self.time.hour >= self.GT_hour_start and self.time.hour < (self.GT_hour_start + self.gt_hours):
                #if len(action) > 0:
                action_0 = 1
                if len(action) == 2:
                    action_1 = 1

                if self.PIR_or_thpl == '2':
                    PIR_gt = 1; THPL_gt = 1
                elif self.PIR_or_thpl == '1':
                    THPL_gt = 1
                elif self.PIR_or_thpl == '0':
                    PIR_gt = 1
                print("GT action taken")

        if self.PIR_or_thpl == '2':
            self.PIR_on_off = action_0
            self.thpl_on_off = action_1
        elif self.PIR_or_thpl == '1':
            self.PIR_on_off = 0
            self.thpl_on_off = action_0
        elif self.PIR_or_thpl == '0':
            self.PIR_on_off = action_0
            self.thpl_on_off = 0

        #if len(action) == 2:
        #    self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action_1-s_t_max_act)+s_t_max_new)
        #elif len(action) == 3:
        #    self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action_2-s_t_max_act)+s_t_max_new)
        self.next_wake_up_time = 60

        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        if self.train == 'real': # now it changes because data are collected with time
            print("looking for data in between: ", self.time, self.time_next)
            #self.file_data = Ember_RL_func.select_input_data(self.path_light_data, self.start_data_date, self.end_data_date)
            self.file_data = Ember_RL_func.select_input_data(self.path_light_data, self.time, self.time_next)
            print("self.file_data: ")
            for data in self.file_data:
                print(data)

        #if len(self.file_data) > 0 and self.file_data[0] != self.last:
        #    self.file_data = [self.last] + self.file_data
        #    self.last = self.file_data[-1]

        #self.start_light_list, self.start_sc_volt_list, starter_data = Ember_RL_func.select_input_starter(self.path_light_data, start_data_date, num_light_input, num_sc_volt_input)
        #print(start_light_list, start_sc_volt_list)
        #self.data_pointer = 0
        #for line in self.file_data:
        #    print("file_data line: ", line)
        #splitt = self.last.split('|')
        #self.start_sc = (float(splitt[5]) * SC_volt_max)/100
        #print("last line found: ", self.last)

        #self.light, PIR_event_gt, PIR_events_found_dict, thpl_event_gt, self.data_pointer = light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.PIR_events_found_dict, self.light, self.light_div, self.file_data, self.data_pointer-1)
        self.light, PIR_event_gt, PIR_events_found_dict, thpl_event_gt, self.data_pointer = Pible_param_func.light_event_func_new(self.time, self.time_next, self.mode, self.PIR_on_off,
                                                                                                                                  self.PIR_events_found_dict, self.light, self.light_div,
                                                                                                                                  self.file_data, self.data_pointer)
        if self.PIR_or_thpl == '0':
            thpl_event_gt = 0
        elif self.PIR_or_thpl == '1':
            PIR_event_gt = 0

        PIR_event_det, PIR_event_miss, thpl_event_det, thpl_event_miss = Pible_param_func.event_det_miss(PIR_event_gt, thpl_event_gt, self.PIR_on_off, self.thpl_on_off, self.SC_Volt_array)

        if self.train == "test" and (PIR_event_miss > 0 or thpl_event_miss > 0) and (self.rem_miss == '1' or self.rem_miss == '2'):
            Ember_RL_func.remove_missed_data(self.time, self.time_next, self.path_light_data)

        SC_temp, en_prod, en_used = Pible_param_func.Energy(self.SC_Volt_array[0], self.light, self.PIR_or_thpl, self.PIR_on_off, self.thpl_on_off, self.next_wake_up_time, PIR_event_det, thpl_event_det)

        if self.train == 'real':
            if len(self.file_data) > 0:
                self.last = self.file_data[-1]
            splitt = self.last.split('|')
            SC_temp = (float(splitt[5]) * SC_volt_max)/100
            #print(SC_temp, self.last)
            print("new events PIR and THPL found: ", PIR_event_det, thpl_event_det)
            print("new events PIR and THPL miss: ", PIR_event_miss, thpl_event_miss)


        reward = self.get_reward_low_level(en_prod, en_used, PIR_event_det, PIR_event_miss, thpl_event_det, thpl_event_miss, self.PIR_on_off, self.thpl_on_off, self.SC_Volt_array)

        len_dict_event = np.array([len(self.PIR_events_found_dict)])
        if self.train == 'test' or self.train == 'real':
            self.Reward.append(reward); self.Time.append(self.time); self.Mode.append(self.mode); self.Light.append(self.light); self.PIR_ON_OFF.append(self.PIR_on_off); self.THPL_ON_OFF.append(self.thpl_on_off); self.SC_Volt.append(SC_temp);
            self.State_Trans.append(self.next_wake_up_time); self.PIR_event_det.append(PIR_event_det); self.PIR_event_miss.append(PIR_event_miss); self.Len_Dict_Events.append(len_dict_event)
            self.thpl_event_det.append(thpl_event_det); self.thpl_event_miss.append(thpl_event_miss); self.PIR_gt.append(PIR_gt); self.THPL_gt.append(THPL_gt)

        self.PIR_tot_events_detect += PIR_event_det
        self.PIR_tot_events += PIR_event_gt
        self.thpl_tot_events_detect += thpl_event_det
        self.thpl_tot_events += thpl_event_gt

        self.time = self.time_next

        self.week_end = Ember_RL_func.calc_week(self.time, num_week_input)

        self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Ember_RL_func.updates_arrays(self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array, self.time, self.light, SC_temp)

        #print("after", self.hour_array, self.minute_array, self.SC_Volt_array)
        done = self.time >= self.end
        #done = True

        self.info["energy_used"] = en_used
        self.info["energy_prod"] = en_prod
        self.info["PIR_tot_events"] = self.PIR_tot_events
        self.info["PIR_events_detect"] = self.PIR_tot_events_detect
        self.info["thpl_tot_events"] = self.thpl_tot_events
        self.info["thpl_events_detect"] = self.thpl_tot_events_detect
        #info["death_days"] = self.death_days
        #info["death_min"] = self.death_min
        self.info["SC_volt"] = self.SC_Volt_array
        self.info["state_transition"] = self.next_wake_up_time
        #self.info["GT_hours_start"] = 0 if (self.GT_hour_start + self.gt_hours) >= 24 else (self.GT_hour_start + self.gt_hours)
        self.info["GT_hours_start"] = 6 if (self.GT_hour_start + self.gt_hours) >= 21 else (self.GT_hour_start + self.gt_hours)
        self.info["hours_inj_prob"] = 0 if self.rem_miss != '2' else self.hours_inj_prob
        #print(self.PIR_tot_events, self.thpl_tot_events, self.PIR_tot_events_detect, self.thpl_tot_events_detect)
        #print(self.hour_array, self.light_array, self.SC_Volt_array, self.week_end, reward, done, info)
        #print(self.data_pointer)
        if self.train != "train" and done:
            self.save_data()

            #print(self.hour_array, self.light_array, self.SC_Volt_array, self.week_end, reward, done, info)
        #return (self.hour_array, self.minute_array, self.SC_Volt_array, self.week_end), reward, done, info
        #print("step ", self.hour_array, self.light_array, self.SC_Volt_array, self.week_end)
        #sleep(5)
        if self.diff_days >= 7:
            return (self.hour_array, self.light_array, self.SC_Volt_array, self.week_end), reward, done, self.info
        else:
            return (self.hour_array, self.light_array, self.SC_Volt_array), reward, done, self.info

    def render(self, tot_rew, title, energy_used, accuracy):
        data = self.prepare_data()
        Pible_param_func.plot_hist_low_level(data, tot_rew, title, energy_used, accuracy)

    def get_reward_low_level(self, en_prod, en_used, PIR_event_det, PIR_event_miss, thpl_event_det,
                   thpl_event_miss, PIR_on_off, thpl_on_off, SC_Volt_array):
        reward = 0.0
        reward += 0.01 * (PIR_event_det + thpl_event_det)
        #reward -= 0.01 * max((SC_Volt_array[1] - SC_Volt_array[0]), 0)

        #reward -= 0.01 * (PIR_event_miss + thpl_event_miss)
        #print(reward)
        #reward -= 0.1*en_used
        #if thpl_on_off == 1 and thpl_event_det == 0:
        #    reward -= 0.001
        #if PIR_on_off == 1 and PIR_event_det == 0:
        #    reward -= 0.001

        if SC_Volt_array[0] <= SC_volt_die:
            reward = -1 #-1

        return reward

    def save_data(self):
        curr_date = self.start_data_date.strftime('%m-%d-%y')
        end_date = self.end_data_date.strftime('%m-%d-%y')
        fold = os.path.basename(os.getcwd())
        final_data = self.prepare_data()
        with open("Save_Data/" + curr_date + '_' + end_date + '_' + self.train +'_' + fold + '.pkl', 'wb') as f:
            pickle.dump(final_data, f, protocol=2)

    def prepare_data(self):
        for i in range(len(self.thpl_event_miss)):
            if self.thpl_event_miss[i] == 0:
                self.thpl_event_miss[i] = np.nan
            if self.PIR_event_miss[i] == 0:
                self.PIR_event_miss[i] = np.nan

        data = {}
        data["Time"] = self.Time; data["Light"] = self.Light; data["Mode"] = self.Mode; data["PIR_ON_OFF"] = self.PIR_ON_OFF; data["THPL_ON_OFF"] = self.THPL_ON_OFF
        data["PIR_gt"] = self.PIR_gt; data["THPL_gt"] = self.THPL_gt; data["State_Trans"] = self.State_Trans; data["Reward"] = self.Reward
        data["SC_Volt"] = self.SC_Volt; data["PIR_event_det"] = self.PIR_event_det; data["PIR_event_miss"] = self.PIR_event_miss
        data["thpl_event_det"] = self.thpl_event_det; data["thpl_event_miss"] = self.thpl_event_miss
        data["PIR_tot_events_detect"] = self.PIR_tot_events_detect; data["info"] = self.info;
        data["Len_Dict_Events"] = self.Len_Dict_Events
        return data

import gym
import numpy as np
from gym.spaces import Discrete, Box
from gym import spaces, logger
import datetime
from Pible_parameters import SC_volt_min, SC_volt_max, episode_length, time_format
import Pible_func
import random
from time import sleep

num_hours_input = 24; num_minutes_input = 60; num_events_input = 10;
num_light_input = 24; num_sc_volt_input = 24; num_week_input = 7
s_t_min_act = 0; s_t_max_act = 1; s_t_min_new = 1; s_t_max_new = 60

MINS_LOOKAHEAD = 60;

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):

        settings = config["settings"]
        path_light = settings[0]["path_light_data"]
        self.light_div = float(settings[0]["light_divider"])
        self.path_light_data = config["main_path"] + "/" + path_light
        self.stage = config["train/test"]

        if self.stage == "train":
            start_data_date = datetime.datetime.strptime(config["start_train"], time_format)
            end_data_date = datetime.datetime.strptime(config["end_train"], time_format)
            self.start_sc = config["sc_volt_start_train"]
        else:
            start_data_date = datetime.datetime.strptime(config["start_test"], time_format)
            end_data_date = datetime.datetime.strptime(config["end_test"], time_format)
            self.start_sc = config["sc_volt_start_test"]

        self.SC_Volt = self.start_sc
        # self.time = datetime.datetime.strptime("10/07/19 00:00:00", time_format)

        self.time_begin = start_data_date
        self.time = self.time_begin
        self.end = self.time_begin + datetime.timedelta(hours=24*episode_length)
        self.events_found_dict = []
        self.file_data = []

        # parse data that are in the range [start_date_time, end_date_time]
        with open(self.path_light_data, 'r') as f:
            for line in f:
                line_split = line.split("|")
                checker = datetime.datetime.strptime(line_split[0], time_format)
                if start_data_date <= checker and checker <= end_data_date:
                    self.file_data.append(line)
        line = self.file_data[0].split('|')

        self.light_begin = int(int(line[8])/self.light_div)
        self.light = self.light_begin

        self.action_space = spaces.Discrete(2) # PIR On_Off, leave it just as 0 or 1 for now;

        self.observation_space = spaces.Tuple((
            spaces.Discrete(num_hours_input),                                       # hours
            spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_sc_volt_input, ), dtype=np.float32), # battery voltage
            spaces.Discrete(num_events_input)                                       # number of events
        ))

        self.Reward = []; self.Mode = []; self.Time = []; self.Light = []; self.PIR_OnOff_hist = []; self.SC_Volt_hist = []; self.State_Trans = []
        self.event_det_hist = []; self.event_miss_hist = []; self.Len_Dict_Events = []; self.tot_events_detect = 0; self.tot_events = 0; self.mode = 0

        self.event_belief = Pible_func.events_look_ahead(self.time, MINS_LOOKAHEAD, self.file_data) # belief for the number of events that would happen in the next interval

        # observation space [hour, light, battery_source, next hour of events];
        # Realistically, next hour of events cannot be obtained in advance, but here we are simplifying our assumption
        # and see if we can learn an optimal policy when it is given.

    def reset(self):
        self.time = self.time_begin
        self.end = self.time + datetime.timedelta(hours=24*episode_length)
        self.events_found_dict = []
        self.light = self.light_begin

        # TODO: reset source voltage;

        self.event_belief = Pible_func.events_look_ahead(self.time, MINS_LOOKAHEAD, self.file_data)
        return self.time.hour, self.light, self.SC_Volt, self.event_belief

    def step(self, action):
        #print("action is: ", action)
        self.PIR_on_off = action

        # self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action[1]-s_t_max_act)+s_t_max_new)
        self.next_wake_up_time = 60
        temp_polling_min = 60

        """ optimal solution for fake dataset;
        if self.time.minute == 0 and (self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17):
            self.PIR_on_off = 1
            self.next_wake_up_time = 1
        elif self.time.minute == 1 and (self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17):
            self.PIR_on_off = 0
            self.next_wake_up_time = 59
        else:
            self.PIR_on_off = 0
            self.next_wake_up_time = 60
        """

        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time in min
        self.light, event_gt, self.events_found_dict = Pible_func.light_event_func(self.time, self.time_next, self.mode, self.PIR_on_off, self.events_found_dict, self.light, self.light_div, self.file_data)

        # rewards and counts of event
        reward = Pible_func.reward_func_low_level_original(self.mode, event_gt, self.PIR_on_off, self.SC_Volt)
        event_det, event_miss = Pible_func.event_count_func(self.mode, event_gt, self.PIR_on_off, self.SC_Volt)

        # energy produced and consumed;
        SC_temp, en_prod, en_used = Pible_func.Energy_original(self.SC_Volt, self.light, self.PIR_on_off, temp_polling_min, self.next_wake_up_time, event_gt)

        len_dict_event = np.array([len(self.events_found_dict)])
        if self.stage == 'test':
            self.Reward.append(reward); self.Time.append(self.time); self.Mode.append(self.mode); self.Light.append(self.light); self.PIR_OnOff_hist.append(self.PIR_on_off); self.SC_Volt_hist.append(SC_temp);
            self.State_Trans.append(self.next_wake_up_time); self.event_det_hist.append(event_det); self.event_miss_hist.append(event_miss); self.Len_Dict_Events.append(len_dict_event)

        self.tot_events_detect += event_det
        self.tot_events += event_gt

        # self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array = Pible_func.updates_arrays(self.hour_array, self.minute_array, self.light_array, self.SC_Volt_array, self.time, self.light, SC_temp)
        # self.week_end = Pible_func.calc_week(self.time, num_week_input)

        self.time = self.time_next
        self.SC_Volt = SC_temp

        # Here we use the ground truth of events that happens in one hour time; Later we need to estimate this
        # From our event data.
        self.event_belief = Pible_func.events_look_ahead(self.time, MINS_LOOKAHEAD, self.file_data)
        done = self.time >= self.end

        info = {}
        info["energy_used"] = en_used
        info["energy_prod"] = en_prod
        info["tot_events"] = self.tot_events
        info["events_detect"] = self.tot_events_detect
        #info["death_days"] = self.death_days
        #info["death_min"] = self.death_min
        info["SC_volt"] = SC_temp

        return (self.time.hour, self.light, self.SC_Volt, self.event_belief), reward, done, info

    def render(self, tot_rew, title):
        Pible_func.plot_hist(self.Time, self.Light, self.Mode, self.PIR_OnOff_hist, self.State_Trans, self.Reward, self.SC_Volt_hist, self.event_det_hist, self.event_miss_hist, tot_rew, self.tot_events_detect, self.tot_events, self.Len_Dict_Events, title)

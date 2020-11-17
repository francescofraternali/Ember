from Pible_param_func import *
import numpy as np
import datetime
from time import sleep
import subprocess
import json
import threading
from subprocess import PIPE, Popen, call
import random
import multiprocessing
import os

def gt_mode_hours(sc_perc):
    if sc_perc >= 90:
        hours = 18
    elif sc_perc >= 80 and sc_perc < 90:
        hours = 12
    elif sc_perc >= 70 and sc_perc < 80:
        hours = 6
    elif sc_perc >= 60 and sc_perc < 70:
        hours = 3
    else:
        hours = 0

    return hours

def valid_line(line):
    if line != '\n':
        line_split = line.split("|")
        if len(line_split) > 9 and line_split[5] != '':
            error = 'list index out of range'
            if error not in line_split:
                if int(line_split[5]) != 0:
                    return True

    return False

def select_input_data(path_light_data, start_data_date, end_data_date):
    file_data = []

    for i, line in enumerate(list(open(path_light_data))):

        if valid_line(line):
            line_split = line.split("|")

            checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
            if start_data_date <= checker and checker < end_data_date:
                file_data.append(line)
            if checker >= end_data_date:
                break
    if len(file_data) == 0:
        pass

    return file_data

def select_input_starter(path_light_data, start_data_date, num_light_input, num_sc_volt_input):

    start_light_list = [0] * num_light_input
    start_volt_list = [SC_volt_die] * num_sc_volt_input
    count_light = 0; count_volt = 0
    starter_data = []

    for line in reversed(list(open(path_light_data))):
        if valid_line(line):
            line_split = line.split("|")
            checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
            if checker <= start_data_date:
                starter_data.append(line)
                if count_light < num_light_input:
                    light = int(line_split[8])
                    light = light_max if light > light_max else light
                    start_light_list[count_light] = light
                    count_light += 1

                if count_volt < num_sc_volt_input:
                    volt = (float(line_split[5]) * SC_volt_max) / 100
                    start_volt_list[count_volt] = round(volt, 2)
                    count_volt += 1


                if count_volt >= num_sc_volt_input and count_light >= num_light_input:
                    break

    return start_light_list, start_volt_list, starter_data

def build_inputs(time, num_hours_input, num_minutes_input, last_light_list, last_volt_list):
    list = []
    for i in range(0, num_hours_input):
        if i == time.hour:
            list.append(1)
        else:
            list.append(0)
    hour_array = np.array(list)

    list = []
    for i in range(0, num_minutes_input):
        if i == time.minute:
            list.append(1)
        else:
            list.append(0)
    minute_array = np.array(list)

    light_array = np.array(last_light_list)

    sc_array = np.array(last_volt_list)

    return hour_array, minute_array, light_array, sc_array

def remove_missed_data(t_now, t_next, path_light_data):
    file_data = []
    for i, line in enumerate(list(open(path_light_data))):
        if valid_line(line):
            line_split = line.split("|")
            checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')

            if checker >= t_now and checker <= t_next:

                pass
            else:
                file_data.append(line)

    with open(path_light_data, 'w') as f:
        for line in file_data:
            f.write('%s' % line)

def updates_arrays(hour_array, minute_array, light_array, SC_Volt_array, time, light, SC_temp):

    list = []
    for i in range(0, len(hour_array)):
        if i == time.hour:
            list.append(1)
        else:
            list.append(0)
    hour_array = np.array(list)

    list = []
    for i in range(0, len(minute_array)):
        if i == time.minute:
            list.append(1)
        else:
            list.append(0)
    minute_array = np.array(list)

    light_array = np.roll(light_array, 1)
    light_array[0] = light

    SC_Volt_array = np.roll(SC_Volt_array, 1)
    SC_Volt_array[0] = SC_temp

    return hour_array, minute_array, light_array, SC_Volt_array

def calc_week(time, num_week_input):
    input_week = []
    for i in range(0, num_week_input):
        if i == time.weekday():
            input_week.append(1)
        else:
            input_week.append(0)

    week_ar = np.array(input_week)
    return week_ar



def find_best_checkpoint(path):  # Find best checkpoint
    best_mean = - 10000
    for count, line in enumerate(open(path + "/result.json", 'r')):
        dict = json.loads(line)
        if round(dict['episode_reward_mean'], 3) >= best_mean and dict['training_iteration'] % 10 == 0: #and count > int(tot_iterations/2):
            best_mean = round(dict['episode_reward_mean'], 3)
            max = round(dict['episode_reward_max'], 3)
            min = round(dict['episode_reward_min'], 3)
            iteration = dict['training_iteration']

    print("Best checkpoint found:", iteration, ". Max Rew Episode: ", max , ". Mean Rew Episode: ", best_mean, ". Min Rew Episode", min)

    return int(iteration)

def rm_old_save_new_agent(parent_dir, save_agent_folder):
    for i in range(2):
        proc = subprocess.Popen("rm -r " + save_agent_folder + '/*', stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        sleep(0.5)
    # Save new Agent into Agents_Saved
    proc = subprocess.Popen("mv " + parent_dir + " " + save_agent_folder + '/', stdout=subprocess.PIPE, shell=True)
    sleep(0.5)
    # Remove file from original folder parent_dir
    for i in range(2):
        proc = subprocess.Popen("rm -r " + parent_dir, stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        sleep(0.5)
    sleep(0.5)

def cores_available(): # Find number of cores available in the running system
    print("Number of cores available: ", multiprocessing.cpu_count())
    print("Number of cores to use: ", multiprocessing.cpu_count() - 2)
    return int(multiprocessing.cpu_count()) - 2


def add_random_volt(SC_Volt_array):
    v_min = round(SC_Volt_array[0] - SC_volt_die - 0.1, 1)
    v_max = round(SC_volt_max - SC_Volt_array[0] - 0.1, 1)
    #print(-v_min, v_max)
    k = round(random.uniform(-v_min, v_max), 1)
    #print("before ", SC_Volt_array)
    SC_Volt_array = [round(x + k , 2) for x in SC_Volt_array]
    for i in range(len(SC_Volt_array)):
        SC_Volt_array[i] = SC_volt_max if SC_Volt_array[i] > SC_volt_max else SC_Volt_array[i]
        SC_Volt_array[i] = SC_volt_die if SC_Volt_array[i] < SC_volt_die else SC_Volt_array[i]
    #print(k, SC_Volt_array)

    return SC_Volt_array

def adjust_sc_voltage(old_list, start_sc):
    if isinstance(start_sc, np.ndarray):
        return start_sc

    new_list = []
    for i in range(0, len(old_list)):
        if i == 0:
            new_list.append(start_sc)
        else:
            new_list.append(start_sc + (old_list[i] - old_list[0]))

    for i in range(len(new_list)):
        new_list[i] = SC_volt_max if new_list[i] > SC_volt_max else new_list[i]
        new_list[i] = SC_volt_min if new_list[i] < SC_volt_min else new_list[i]

    return new_list

def calc_accuracy(info):
    accuracy = 0
    if (int(info["PIR_events_detect"]) + int(info["thpl_events_detect"])) != 0 or (int(info["PIR_tot_events"]) + int(info["thpl_tot_events"])) != 0 :
        accuracy = float((int(info["PIR_events_detect"]) + int(info["thpl_events_detect"]))/(int(info["PIR_tot_events"]) +int(info["thpl_tot_events"])))
        accuracy = accuracy * 100
    if (int(info["PIR_tot_events"]) + int(info["thpl_tot_events"])) == 0:
        accuracy = 100

    return round(accuracy, 1)

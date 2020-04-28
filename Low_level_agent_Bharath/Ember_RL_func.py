from Pible_param_func import *
import numpy as np
import datetime
from time import sleep
import subprocess
import json

#from training_pible import light_divider

def reward_low_level(en_prod, en_used, PIR_event_det, PIR_event_miss, thpl_event_det,
               thpl_event_miss, PIR_on_off, thpl_on_off, SC_Volt_array):
    reward = 0

    reward += 0.01 * (PIR_event_det + thpl_event_det)

    reward -= 0.01 * (PIR_event_miss + thpl_event_miss)

    reward -= 0.1*en_used

    #if thpl_on_off == 1 and thpl_event_det == 0:
    #    reward -= 0.001

    if SC_Volt_array[0] <= SC_volt_die:
        reward = -1 #-1

    return reward

def build_inputs(time, light, sc_volt, num_hours_input, num_minutes_input,  num_light_input, num_sc_volt_input):
    list = []
    for i in range(0, num_hours_input):
        value = time - datetime.timedelta(hours=1)
        time = time - datetime.timedelta(hours=1)
        list.append(value.hour)
    hour_array = np.array(list)

    list = []
    for i in range(0, num_minutes_input):
        value = time - datetime.timedelta(minutes=1)
        time = time - datetime.timedelta(minutes=1)
        list.append(value.minute)
    minute_array = np.array(list)

    light_array = np.array([0] * num_light_input)

    sc_array = np.array([sc_volt] * num_sc_volt_input)

    return hour_array, minute_array, light_array, sc_array

def updates_arrays(hour_array, minute_array, light_array, SC_Volt_array, time, light, SC_temp):
    hour_array = np.roll(hour_array, 1)
    hour_array[0] = time.hour

    minute_array = np.roll(minute_array, 1)
    minute_array[0] = time.minute

    list = []
    for i in range(0, 24):
        value = time - datetime.timedelta(hours=1)
        time = time - datetime.timedelta(hours=1)
        list.append(value.hour)
    hour_array = np.array(list)

    list = []
    for i in range(0, 60):
        value = time - datetime.timedelta(minutes=1)
        time = time - datetime.timedelta(minutes=1)
        list.append(value.minute)
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
    #print(week_ar)
    return week_ar


def find_agent_saved(path):
    Agnt = 'PPO'
    # Detect latest folder for trainer to resume
    latest = 0
    #proc = subprocess.Popen("ls " + path + "/ray_results/", stdout=subprocess.PIPE, shell=True)
    proc = subprocess.Popen("ls " + path, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode()
    spl = out.strip().split('\n')
    for i in spl:
        test = i.split('.')
        #print(test)
        if "json" not in test and len(test[0].split('_')) > 1:
            d = i.split('_')
            #print(d)
            date = d[2].split('-')
            hour = d[3].split('-')
            x = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(hour[0]), int(hour[1]))
            if latest == 0:
                folder = i; time = x; latest = 1
                folder_found = i
            else:
                if x >= time:
                    folder = i; time = x
                    # Checking for a better folder
                    #if d[3] == "lr=0.0001":
                    #if d[3] == "lr=1e-05":
                    if 1:
                        folder_found = i

    #print("folder: ", folder_found)
    folder = folder_found

    # detect checkpoint to resume
    proc = subprocess.Popen("ls " + path + "/" + folder + '/', stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    #print(out)
    out = out.decode()
    spl = out.strip().split('\n')
    max = 0
    for i in spl:
        tester = i.split('_')
        #print(tester, len(tester), tester[1].isdigit())
        if "checkpoint" in tester and len(tester)==2 and tester[1].isdigit():
            if int(tester[1]) > max:
                max = int(tester[1])
                iteration = i
    iteration = max
    print("\nFound folder: ", folder, "Last checkpoint found: ", iteration)


    # Find best checkpoint, If nor uncomment here and it will use the last checkpoint found
    max_mean = - 10000
    tot_iterations = iteration
    #print("tot iterations", tot_iterations)
    for count, line in enumerate(open(path + "/" + folder + "/result.json", 'r')):
        dict = json.loads(line)
        #print(count, int(tot_iterations/2))
        if round(dict['episode_reward_mean'], 3) >= max_mean and count > int(tot_iterations/2):
            max_mean = round(dict['episode_reward_mean'], 3)
            #iteration = count
            iteration = dict['training_iteration']
            #print("saving", iteration)
        #data = json.loads(text)
        #for p in data["episode_reward_mean"]:
        #    print(p)
    if iteration < 10:
        iteration = 10
    iter_str = str(iteration)
    iteration = (int(iter_str[:-1])* 10)
    print("Best checkpoint found:", iteration, ". Mean Reward Episode: ", round(max_mean, 3), ". Min Rew Episode", round(dict['episode_reward_min'], 3))


    return folder, iteration

from Pible_parameters import *
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
from time import sleep
import random
import subprocess
import json

#from training_pible import light_divider

def Energy(SC_volt, light, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event):
    #SC_volt_save = SC_volt

    if thpl_on_off == 1:
        temp_polling_min = 1
        num_sens = 3
    else:
        temp_polling_min = 15
        num_sens = 1

    volt_regulator = 2.55
    next_wake_up_time_sec = next_wake_up_time * 60 # in seconds
    temp_polling_sec = temp_polling_min * 60 # in seconds

    num_of_pollings = int(next_wake_up_time_sec/temp_polling_sec)
    time_sleep = next_wake_up_time_sec

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
        Energy_Used = 0
    else: # Node is alive
        # Energy used to sense sensors (i.e. light and temp)
        Energy_Used = (time_sens * volt_regulator * i_sens) * (num_sens * num_of_pollings)
        time_sleep -= (time_sens * num_of_pollings)

        # energy consumed to detect people
        Energy_Used += (time_PIR_detect * volt_regulator * i_PIR_detect) * PIR_event
        time_sleep -= (time_PIR_detect * PIR_event)

        # energy consumed by the node to send BLE data
        tot_events = PIR_event + thpl_event + 1 # the +1 is the heartbit event. Soat least there is a minimum of 1 data sent even if 0 events detected
        Energy_Used += (time_BLE_sens * volt_regulator * i_BLE_sens) * tot_events
        time_sleep -= (time_BLE_sens * tot_events)

        # Energy Consumed by the node in sleep mode
        i_sl = i_sleep_PIR if PIR_on_off == 1 else i_sleep
        Energy_Used += (time_sleep * volt_regulator * i_sl)

    Energy_Prod = next_wake_up_time_sec * p_solar_1_lux * light
    #print(Energy_Prod, Energy_Used, Energy_Rem, SC_volt, event)

    # Energy cannot be lower than 0
    Energy_Rem = max(Energy_Rem - Energy_Used + Energy_Prod, 0)

    SC_volt = np.sqrt((2*Energy_Rem)/SC_size)

    # Setting Boundaries for Voltage
    if SC_volt > SC_volt_max:
        SC_volt = np.array([SC_volt_max])
    if SC_volt < SC_volt_min:
        SC_volt = np.array([SC_volt_min])

    return SC_volt, Energy_Prod, Energy_Used


def light_event_func(t_now, next_wake_up_time, mode, PIR_on_off, PIR_events_found_dict, light_prev, light_div, file_data): # check how many events are on this laps of time
        #time_buff = []; light_buff = []; PIR_buff = []; temp_buff = []; hum_buff = []; press_buff = []
        light_buff = [];
        PIR_event_gt = 0; event_found = 0
        thpl_event_gt = 0
        #temp = temp_prev
        #hum = hum_prev
        #press = press_prev
        #for i in range(count, len(file_data)):
        #t_diff = next_wake_up_time - t_now
        #t_diff = int(t_diff.total_seconds()/60))
        hold = 0; no_event = 0
        for line in file_data:
            PIR = 0
            splitted = line.split("|")
            check_time = datetime.datetime.strptime(splitted[0], '%m/%d/%y %H:%M:%S')
            if hold == 0:
                old_time = check_time
            t_diff = int((check_time - old_time).total_seconds()/60)
            no_event = 1 if t_diff >= 60 else 0

            #print(t_now, check_time, next_wake_up_time, t_diff, thpl_event_gt)
            #sleep(5)
            #check_time = check_time_file + datetime.timedelta(days=days_repeat*diff_days)

            if t_now <= check_time and check_time < next_wake_up_time:
                light_buff.append(int(int(splitted[8])/light_div))
                PIR = int(splitted[6])
                PIR_event_gt += PIR

                if PIR == 0 and no_event == 0 and hold != 0:
                    thpl_event_gt += 1

            #if (mode == 1 or mode == 2) and PIR > 0 and PIR_on_off > 0:
            #    if check_time.time() not in PIR_events_found_dict:
            #        PIR_events_found_dict.append(check_time.time())
            if check_time >= next_wake_up_time:
                break

            old_time = check_time
            hold = 1

        #t_diff = int((next_wake_up_time - t_now).total_seconds()/60)
        #if thpl_event_gt == 1 and t_diff >= 60: # and (t_now.hour != 18) and (t_now.hour != 8):
        #    thpl_event_gt -= 1
        if len(light_buff) == 0:
            light_buff = [light_prev]

        light = sum(light_buff)/len(light_buff)
        #if len(time_buff) == 0:
        #    time_buff = [t_now]; light_buff = [light_prev]; PIR_buff = [0]; temp_buff = [temp_prev]; hum_buff = [hum_prev]; press_buff = [press_prev]

        #print(t_now, check_time, next_wake_up_time, t_diff, thpl_event_gt)

        #return time_buff, light_buff, PIR_buff, PIR_event_gt, PIR_events_found_dict, temp_buff, hum_buff, press_buff
        return light, PIR_event_gt, PIR_events_found_dict, thpl_event_gt

def reward_func_high_level(mode, event, PIR_on_off, SC_Volt_array):
    reward = 0; detect = 0
    miss = np.nan

    if SC_Volt_array[0] <= SC_volt_die:
        reward = -1 #-1
    elif PIR_on_off == 1 and event != 0:
        detect = event
        if mode == 2:
            reward = 0.01*event
        if mode == 1:
            reward = 0.001*event
    elif PIR_on_off == 0 and event != 0:
        miss = event
        detect = np.nan
    elif mode == 0 and event == 0:
        reward = 0.001

    return reward, detect, miss

def reward_func_low_level(mode, PIR_event_det, PIR_event_miss, thpl_event_det,
                          thpl_event_miss, PIR_on_off, thpl_on_off, SC_Volt_array):
    reward = 0

    #reward += 0.01 * (PIR_event_det + thpl_event_det)
    reward += 0.01 * (PIR_event_det)

    reward -= 0.01 * (PIR_event_miss) # reward -= 0.01 * (PIR_event_miss + thpl_event_miss)

    if PIR_on_off == 1 and PIR_event_det == 0:
        reward -= 0.001

    #if thpl_on_off == 1 and thpl_event_det == 0:
    #    reward -= 0.001

    if SC_Volt_array[0] <= SC_volt_die:
        reward = -1 #-1

    return reward

def reward_new(en_prod, en_used, PIR_event_det, PIR_event_miss, thpl_event_det,
               thpl_event_miss, PIR_on_off, thpl_on_off, SC_Volt_array):
    reward = 0

    reward += 0.01 * (PIR_event_det + thpl_event_det)
    #reward += 0.01 * (PIR_event_det)

    #reward -= 0.01 * (PIR_event_miss)
    reward -= 0.01 * (PIR_event_miss + thpl_event_miss)

    #reward -= (0.0517/en_used)*0.001
    reward -= 0.1*en_used

    #if thpl_on_off == 1 and thpl_event_det == 0:
    #    reward -= 0.001

    if SC_Volt_array[0] <= SC_volt_die:
        reward = -1 #-1

    return reward

def event_det_miss(PIR_event, thpl_event, PIR_on_off, thpl_on_off, SC_Volt_array):
    PIR_detect = 0; thpl_detect = 0
    PIR_miss = 0; thpl_miss = 0

    if PIR_on_off == 1:
        PIR_detect = PIR_event
    else:
        PIR_miss = PIR_event

    if thpl_on_off == 1:
        thpl_detect = thpl_event
    else:
        thpl_miss = thpl_event

    if SC_Volt_array[0] <= SC_volt_die:
        PIR_detect = 0; thpl_detect = 0;
        PIR_miss = PIR_event; thpl_miss = thpl_event

    return PIR_detect, PIR_miss, thpl_detect, thpl_miss

def build_inputs(time, light, sc_volt, num_hours_input, num_minutes_input,  num_light_input, num_sc_volt_input):
    #hour_array = np.array([23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 0])
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

    light_array = np.array([light] * num_light_input)

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

def plot_hist_low_level(Time, Light, Mode, Sens_OnOff, State_Trans, Reward,
                        SC_Volt, PIR_det, PIR_miss, thpl_det, thpl_miss, tot_rew,
                        event_detect, tot_events, Dict_Events, title_final, energy_used):

    plt.figure(1)
    ax1 = plt.subplot(711)
    plt.title(('{2}, Tot Rew: {0}, Energy Used: {1}').format(round(tot_rew, 5), round(energy_used, 5), title_final))
    #plt.title(title_final, fontsize = 17)
    plt.plot(Time, Light, 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    #ax1.set_xticklabels([])
    plt.grid(True)

    ax2 = plt.subplot(712)
    #plt.plot(Time, PIR, 'k.', label = 'PIR detection', markersize = 15)
    #plt.plot(Time, PIR_miss, 'r.', Time, PIR_det, 'k.', label = 'PIR detection', markersize = 15, )
    plt.plot(Time, PIR_miss, 'r.', label = 'Missed', markersize = 15)
    plt.plot(Time, PIR_det, 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center left", prop={'size': 9})
    #ax2.set_xticklabels([])
    plt.grid(True)

    ax3 = plt.subplot(713)
    plt.plot(Time, SC_Volt, 'm.', label = 'SC Voltage', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(2.3, 3.7)
    #ax3.set_xticklabels([])
    plt.grid(True)

    ax4 = plt.subplot(714)
    plt.plot(Time, Sens_OnOff, 'y.', label = 'Action', markersize = 15)
    plt.ylabel('Action\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    #ax4.set_xticklabels([])
    plt.grid(True)

    ax5 = plt.subplot(715)
    plt.plot(Time, State_Trans, 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[min]', fontsize=15)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    #ax5.set_xticklabels([])
    plt.grid(True)

    ax6 = plt.subplot(716)
    plt.plot(Time, thpl_miss, 'r.', label = 'Missed', markersize = 15)
    plt.plot(Time, thpl_det, 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('THPL\nevents\n[num]', fontsize=15)
    #ax6.set_xticklabels([])
    plt.grid(True)

    ax7 = plt.subplot(717)
    plt.plot(Time, Reward, 'b.', label = 'Reward', markersize = 15)
    plt.ylabel('Reward\n[num]', fontsize=12)
    plt.xlabel('Time [hh:mm]', fontsize=15)
    #ax8.tick_params(axis='both', which='major', labelsize=12)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    xfmt = mdates.DateFormatter('%m/%d %H')
    ax7.xaxis.set_major_formatter(xfmt)
    plt.grid(True)
    #plt.savefig('/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Ember/HRL/High_level_central/Save_Data/Graph_' + str(Time[0].date()) + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close("all")


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

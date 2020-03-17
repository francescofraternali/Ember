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

def Energy(SC_volt, light, action, next_wake_up_time, event):
    #SC_volt_save = SC_volt
    next_wake_up_time_sec = next_wake_up_time * 60 # in seconds

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
        Energy_Used = 0
    else: # Node is alive
        Energy_Used = (SC_volt * i_sens) * time_sens # Energy used to sense sensors (i.e. light)
        time_sleep = next_wake_up_time_sec - time_sens

        if action == 1: # Node was able to detect events using the PIR and hence he will consume energy
            i_sl = i_sleep_PIR
            Energy_Used += (time_PIR_detect * SC_volt * i_PIR_detect) * event # energy consumed to detect people
            time_sleep = time_sleep - (time_PIR_detect * event)
            Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) * event # energy consumed by the node to send data
            time_sleep -= time_BLE_sens * event
        else:
            i_sl = i_sleep

        if event == 0: # Every time it wakes up there is at least one BLE communication, even with events = 0
            Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) # energy consumed by the node to send one data
            time_sleep -= time_BLE_sens

        Energy_Used += (time_sleep * SC_volt * i_sl) # Energy Consumed by the node in sleep mode

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

    #SC_volt = SC_volt_save
    #SC_volt = np.round(SC_volt, 4)

    return SC_volt, Energy_Prod, Energy_Used

'''
def light_event_func(t_now, next_wake_up_time, mode, PIR_on_off, events_found_dict, count, light_prev, file_data, days_repeat, diff_days, light_div): # check how many events are on this laps of time
    event_gt = 0
    event_found = 0
    light = 0
    PIR = 0
    count = 0
    #print("t_nowwwwww", t_now.hour)
    if (8 <= t_now.hour and t_now.hour < 9) or (10 <= t_now.hour and t_now.hour < 11) or (17 <= t_now.hour and t_now.hour < 18):
        light = 500
        PIR = 1
        event_gt += PIR
    if (mode == 1 or mode == 2) and PIR > 0 and PIR_on_off > 0:
        if t_now.hour not in events_found_dict:
            events_found_dict.append(t_now.hour)

    return light, count, event_gt, events_found_dict

'''
def light_event_func(t_now, next_wake_up_time, mode, PIR_on_off, events_found_dict, light_prev, file_data): # check how many events are on this laps of time
        event_gt = 0
        event_found = 0
        light = light_prev
        #for i in range(count, len(file_data)):
        for line in file_data:
            PIR = 0
            splitted = line.split("|")
            check_time = datetime.datetime.strptime(splitted[0], '%m/%d/%y %H:%M:%S')
            #check_time = check_time_file + datetime.timedelta(days=days_repeat*diff_days)
            if t_now <= check_time and check_time < next_wake_up_time:
                light = int(int(splitted[8])/2)
                PIR = int(splitted[6])
                event_gt += PIR
            if (mode == 1 or mode == 2) and PIR > 0 and PIR_on_off > 0:
                if check_time.time() not in events_found_dict:
                    events_found_dict.append(check_time.time())
            if check_time >= next_wake_up_time:
                break

        return light, event_gt, events_found_dict

def reward_func_high_level(PIR_on_Off, mode, event, SC_volt, death_days, death_min, next_wake_up):
    reward = 0; detect = 0
    miss = np.nan
    if PIR_on_Off == 1 and event != 0:
        detect = event
        if mode == 2:
            reward = 0.01*event
        if mode == 1:
            reward = 0.001*event
    elif PIR_on_Off == 0 and event != 0:
        miss = event
        detect = np.nan

    '''
    if SC_volt <= SC_volt_die:
        reward = -1 #-1
        death_days +=1
        death_min += next_wake_up
        if death_days >= 1:
            death_days = 1
    '''

    return reward, detect, miss, death_days, death_min

def reward_func_low_level(action, event, SC_volt, death_days, death_min, next_wake_up):
    reward = 0; detect = 0
    miss = np.nan
    if action == 1 and event != 0:
        reward = 0.01*event
        detect = event
    elif action == 0 and event != 0:
        reward = -0.01*(event)
        miss = event
    elif action == 1 and event == 0:
        reward = -0.001

    if SC_volt <= SC_volt_die:
        reward = -1 #-1
        death_days +=1
        death_min += next_wake_up
        if death_days >= 1:
            death_days = 1

    return reward, detect, miss, death_days, death_min

def randomize_light_time(input_data_raw):
    input_data = []
    rand_time = random.randrange(-15, 15, 1)
    rand_light = random.randrange(-30, 30, 1)
    #rand_time = 0
    #rand_light = 0
    for i in range(0,len(input_data_raw)):
        line = input_data_raw[i].split("|")
        curr_time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        curr_time = curr_time + datetime.timedelta(minutes=rand_time)
        curr_time_new = curr_time.strftime('%m/%d/%y %H:%M:%S')
        light = int(line[8])
        new_light = int(light + ((light/100) * rand_light))
        #if i == 1:
        #    print(rand_light, i, light, new_light)
        #    sleep(5)
        if new_light < 0:
            new_light = 0
        line[0] = curr_time_new
        line[8] = str(new_light)
        new_line = '|'.join(line)
        input_data.append(new_line)

    return input_data

def build_inputs(time, num_hours_input, num_minutes_input, num_light_input, light):
    hour = np.array([time.hour] * num_hours_input)

    minute = np.array([time.minute] * num_minutes_input)

    light_ar = np.array([light] * num_light_input)

    return hour, minute, light_ar

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

def plot_hist(Time, Light, Mode, PIR_OnOff, State_Trans, Reward, Perf, SC_Volt, PIR, PIR_det, PIR_miss, episode, tot_rew, event_detect, tot_events, Dict_Events, title_final):
    #print("Total reward: ", tot_rew)
    #print("SC_volt_init [V]:", SC_Volt[0], "SC_volt_final[V]: ", SC_Volt[-1])
    #perc_init = (SC_Volt[0]/SC_volt_max)*100
    #perc_final = (SC_Volt[-1]/SC_volt_max)*100
    #print("SC_difference [%]: ", perc_init - perc_final)
    #Start Plotting
    plt.figure(1)
    ax1 = plt.subplot(811)
    #plt.title(('Transmitting every {0} sec, PIR {1} ({2} events). Tot reward: {3}').format('60', using_PIR, PIR_events, tot_rew))
    plt.title(title_final, fontsize = 17)
    plt.plot(Time, Light, 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax1.set_xticklabels([])
    plt.grid(True)
    ax2 = plt.subplot(812)
    #plt.plot(Time, PIR, 'k.', label = 'PIR detection', markersize = 15)
    #plt.plot(Time, PIR_miss, 'r.', Time, PIR_det, 'k.', label = 'PIR detection', markersize = 15, )
    plt.plot(Time, PIR_miss, 'r.', label = 'Missed', markersize = 15)
    plt.plot(Time, PIR_det, 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center left", prop={'size': 9})
    ax2.set_xticklabels([])
    plt.grid(True)
    ax3 = plt.subplot(813)
    plt.plot(Time, SC_Volt, 'm.', label = 'SC Voltage', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(2.3, 3.7)
    ax3.set_xticklabels([])
    plt.grid(True)
    ax4 = plt.subplot(814)
    plt.plot(Time, PIR_OnOff, 'y.', label = 'PIR_OnOff', markersize = 15)
    plt.ylabel('PIR\nAction\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    ax4.set_xticklabels([])
    plt.grid(True)

    ax5 = plt.subplot(815)
    plt.plot(Time, State_Trans, 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[min]', fontsize=15)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax5.set_xticklabels([])
    plt.grid(True)

    ax6 = plt.subplot(816)
    plt.plot(Time, Mode, 'c.', label = 'Mode', markersize = 15)
    plt.ylabel('Mode\n[num]', fontsize=15)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(-0.1, 2.1)
    ax6.set_xticklabels([])
    plt.grid(True)

    ax7 = plt.subplot(817)
    plt.plot(Time, Dict_Events, '.', label = 'Mode', markersize = 15, color='gray')
    plt.ylabel('Events\nFounds\n[num]', fontsize=12)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0, 2.1)
    ax7.set_xticklabels([])
    ax7.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)

    ax8 = plt.subplot(818)
    plt.plot(Time, Reward, 'b.', label = 'Reward', markersize = 15)
    plt.ylabel('Reward\n[num]', fontsize=12)
    plt.xlabel('Time [hh:mm]', fontsize=15)
    #ax8.tick_params(axis='both', which='major', labelsize=12)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    xfmt = mdates.DateFormatter('%m/%d %H')
    ax8.xaxis.set_major_formatter(xfmt)
    plt.grid(True)
    plt.savefig('/mnt/c/Users/Francesco/Dropbox/EH/RL/RL_MY/Ember/HRL/High_level_central/Save_Data/Graph_' + str(Time[0].date()) + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close("all")

def write_results(tot_rew, start_train, end_train, start_test, end_test, percent, diff_days, energy_prod_tot_avg, energy_used_tot_avg, events_detect, Tot_events, death_days, death_min, volt_diff):
    import json
    dict = {"tot_rew": tot_rew, "start_train_date": start_train, "end_train_date": end_train, "start_test_date": start_test, "end_test_date": end_test, "percent": percent, "num_days": diff_days, "energy_prod_tot_avg": energy_prod_tot_avg, "energy_used_tot_avg": energy_used_tot_avg, "events_detect": events_detect, "Tot_events": Tot_events, "Death_days": str(death_days), "Death_min": str(death_min), "Volt_diff": str(volt_diff)}
    json = json.dumps(dict)
    with open("results.json","a") as f:
        f.write(json)
        f.write('\n')

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
        if dict['episode_reward_mean'] >= max_mean and count > int(tot_iterations/2):
            max_mean = dict['episode_reward_mean']
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

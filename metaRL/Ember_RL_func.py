from Pible_param_func import *
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
from time import sleep
import subprocess
import json

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

def reward_func_meta_level(mode, PIR_event_det, PIR_event_miss, thpl_event_det, thpl_event_miss, PIR_on_off, thpl_on_off, SC_Volt_array):
    reward = 0

    reward += 0.01 * (PIR_event_det + thpl_event_det)

    reward -= 0.01 * (PIR_event_miss + thpl_event_miss)

    if PIR_on_off == 1 and PIR_event_det == 0:
        reward -= 0.001

    if thpl_on_off == 1 and thpl_event_det == 0:
        reward -= 0.001

    if SC_Volt_array[0] <= SC_volt_die:
        reward = -1 #-1

    return reward

def reward_func_low_level(mode, PIR_event_det, PIR_event_miss, thpl_event_det, thpl_event_miss, PIR_on_off, thpl_on_off, SC_Volt_array):
    reward = 0

    if PIR_event_det is np.nan:
        PIR_event_det = 0
    if PIR_event_miss is np.nan:
        PIR_event_miss = 0
    if thpl_event_det is np.nan:
        thpl_event_det = 0
    if thpl_event_miss is np.nan:
        thpl_event_miss = 0

    reward += 0.01 * (PIR_event_det + thpl_event_det)

    reward -= 0.01 * (PIR_event_miss + thpl_event_miss)

    if PIR_on_off == 1 and PIR_event_det == 0:
        reward -= 0.001

    if thpl_on_off == 1 and thpl_event_det == 0:
        reward -= 0.001

    if SC_Volt_array[0] <= SC_volt_die:
        reward = -1 #-1

    return reward

def build_inputs(time, light, sc_volt, num_hours_input, num_minutes_input, num_light_input, num_sc_volt_input):
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

def updates_arrays(hour_array, minute_array, light_array, SC_Volt_array_1, SC_Volt_array_2, time, light, SC_temp_1, SC_temp_2):
    hour_array = np.roll(hour_array, 1)
    hour_array[0] = time.hour

    minute_array = np.roll(minute_array, 1)
    minute_array[0] = time.minute

    light_array = np.roll(light_array, 1)
    light_array[0] = light

    SC_Volt_array_1 = np.roll(SC_Volt_array_1, 1)
    SC_Volt_array_1[0] = SC_temp_1

    SC_Volt_array_2 = np.roll(SC_Volt_array_2, 1)
    SC_Volt_array_2[0] = SC_temp_2

    return hour_array, minute_array, light_array, SC_Volt_array_1, SC_Volt_array_2

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

def plot_hist_low_level(Time, Light, Mode, PIR_OnOff, State_Trans, Reward, SC_Volt, PIR_det, PIR_miss, thpl_det, thpl_miss, tot_rew, event_detect, tot_events, Dict_Events, title_final):

    plt.figure(1)
    ax1 = plt.subplot(711)
    plt.title(('Tot reward: {0}').format(round(tot_rew, 3)))
    #plt.title(title_final, fontsize = 17)
    plt.plot(Time, Light, 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax1.set_xticklabels([])
    plt.grid(True)

    ax2 = plt.subplot(712)
    #plt.plot(Time, PIR, 'k.', label = 'PIR detection', markersize = 15)
    #plt.plot(Time, PIR_miss, 'r.', Time, PIR_det, 'k.', label = 'PIR detection', markersize = 15, )
    plt.plot(Time, PIR_miss, 'r.', label = 'Missed', markersize = 15)
    plt.plot(Time, PIR_det, 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center left", prop={'size': 9})
    ax2.set_xticklabels([])
    plt.grid(True)

    ax3 = plt.subplot(713)
    plt.plot(Time, SC_Volt, 'm.', label = 'SC Voltage', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(2.3, 3.7)
    ax3.set_xticklabels([])
    plt.grid(True)

    ax4 = plt.subplot(714)
    plt.plot(Time, PIR_OnOff, 'y.', label = 'Action', markersize = 15)
    plt.ylabel('Action\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    ax4.set_xticklabels([])
    plt.grid(True)

    ax5 = plt.subplot(715)
    plt.plot(Time, State_Trans, 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[min]', fontsize=15)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax5.set_xticklabels([])
    plt.grid(True)

    ax6 = plt.subplot(716)
    plt.plot(Time, thpl_miss, 'r.', label = 'Missed', markersize = 15)
    plt.plot(Time, thpl_det, 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('THPL\nevents\n[num]', fontsize=15)
    ax6.set_xticklabels([])
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

def plot_meta_RL(Time, Light, Mode, Node_Select, PIR_OnOff, State_Trans, Reward, SC_Volt_1, SC_Volt_2, PIR_det, PIR_miss, tot_rew, event_detect, tot_events, Dict_Events, title_final):

    plt.figure(1)
    ax1 = plt.subplot(811)
    plt.title(('Tot reward: {0}').format(round(tot_rew, 3)))
    #plt.title(title_final, fontsize = 17)
    plt.plot(Time, Light, 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax1.set_xticklabels([])
    plt.grid(True)

    ax2 = plt.subplot(812)
    plt.plot(Time, PIR_miss, 'r.', label = 'Missed', markersize = 15)
    plt.plot(Time, PIR_det, 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center left", prop={'size': 9})
    ax2.set_xticklabels([])
    plt.grid(True)

    ax3 = plt.subplot(813)
    plt.plot(Time, SC_Volt_1, 'm.', label = 'SC Voltage 1', markersize = 10)
    plt.plot(Time, SC_Volt_2, 'b.', label = 'SC Voltage 2', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(2.3, 3.7)
    ax3.set_xticklabels([])
    plt.grid(True)

    ax4 = plt.subplot(814)
    plt.plot(Time, Node_Select, 'b.', label = 'Node Select', markersize = 15)
    plt.ylabel('Node\n\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    ax4.set_xticklabels([])
    plt.grid(True)

    ax5 = plt.subplot(815)
    plt.plot(Time, PIR_OnOff, 'y.', label = 'PIR_OnOff', markersize = 15)
    plt.ylabel('PIR\nAction\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    ax5.set_xticklabels([])
    plt.grid(True)

    ax6 = plt.subplot(816)
    plt.plot(Time, State_Trans, 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[min]', fontsize=15)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax6.set_xticklabels([])
    plt.grid(True)

    ax7 = plt.subplot(817)
    plt.plot(Time, Mode, 'c.', label = 'Mode', markersize = 15)
    plt.ylabel('Mode\n[num]', fontsize=15)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    plt.ylim(-0.1, 2.1)
    ax7.set_xticklabels([])
    plt.grid(True)

    '''
    ax7 = plt.subplot(817)
    plt.plot(Time, Dict_Events, '.', label = 'Mode', markersize = 15, color='gray')
    plt.ylabel('Events\nFounds\n[num]', fontsize=12)
    #plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0, 2.1)
    ax7.set_xticklabels([])
    ax7.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    '''

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

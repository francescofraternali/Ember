import datetime
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from time import sleep
import math

# Parameters to change
SC_volt_die = 3.0 # Voltage at which the simulator consider the node death
SC_begin = 4.0 # Super Capacitor initial voltage level. Put a number between 5.4 (i.e. max) and 2.3 (i.e. min)
using_Accelerometer = False # Activate only if using Accelerometer

# RL Parameters
episode_lenght = 1 # in days
light_max = 1000

# DO NOT MODIFY! POWER CONSUMPTION PARAMETERS! Change them only if you change components.
SC_volt_min = 2.3; SC_volt_max = 5.5; SC_size = 1.5; SC_volt_die = 3.0

# Board and Components Consumption
#i_sleep = 0.0000032;
#i_sens =  0.0003; time_sens = 0.2
i_sens =  0.0004; time_sens = 0.2 # It was i_sens =  0.00067; time_sens = 0.13
i_PIR_detect = 0.000102; time_PIR_detect = 2.5
i_accel_sens = 0.0026; accel_sens_time = 0.27
#i_sleep_PIR = i_sleep + 0.000001 # original
#i_sleep_PIR = i_sleep + 0.0001
#i_sleep_PIR = i_sleep + 0.000001

# if using_Accelerometer:
#    i_sleep += 0.000008

# Communication (wake up and transmission) and Sensing Consumption
i_wake_up_advertise = 0.000065; time_wake_up_advertise = 11
i_BLE_comm = 0.0003; time_BLE_comm = 5; # it was i_BLE_comm = 0.00025;
i_BLE_sens = ((i_wake_up_advertise * time_wake_up_advertise) + (i_BLE_comm * time_BLE_comm))/(time_wake_up_advertise + time_BLE_comm)
#time_BLE_sens = time_wake_up_advertise + time_BLE_comm

#i_BLE_sens = 0.000210; time_BLE_sens = 6.5

# Solar Panel Production
v_solar_200_lux = 1.5; i_solar_200_lux = 0.0000333
p_solar_1_lux = (v_solar_200_lux * i_solar_200_lux) / 200.0


def Energy(SC_volt, light, PIR_or_thpl, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event):
    #print("in: ", SC_volt)

    if int(PIR_or_thpl) > 0:
        i_sleep = 0.0000042; # it was i_sleep = 0.00000445
        time_BLE_comm = 12; num_sens = 3
        time_BLE_sens = time_wake_up_advertise + time_BLE_comm
    else:
        time_BLE_comm = 5; num_sens = 1
        time_BLE_sens = time_wake_up_advertise + time_BLE_comm
        i_sleep = 0.00000345;

    i_sleep_PIR = i_sleep + 0.000002 # it was 0.0000028

    if thpl_on_off == 1:
        temp_polling_min = 1
    else:
        temp_polling_min = 60

    i_BLE_sens = ((i_wake_up_advertise * time_wake_up_advertise) + (i_BLE_comm * (time_BLE_comm)))/(time_wake_up_advertise + time_BLE_comm)

    volt_regulator = 3.0
    next_wake_up_time_sec = next_wake_up_time * 60 # in seconds
    temp_polling_sec = temp_polling_min * 60 # in seconds

    num_of_pollings = int(next_wake_up_time_sec/temp_polling_sec)
    time_sleep = next_wake_up_time_sec

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
    #if False:
        Energy_Used = 0
    else: # Node is alive
        # Energy used to sense sensors (e.g. light, temp)
        Energy_Used = (time_sens * volt_regulator * i_sens) * (num_sens * num_of_pollings)
        time_sleep -= (time_sens * num_of_pollings)

        # energy consumed to detect people with PIR
        Energy_Used += (time_PIR_detect * volt_regulator * i_PIR_detect) * PIR_event
        time_sleep -= (time_PIR_detect * PIR_event)

        # energy consumed by the node to send BLE data
        tot_events = PIR_event + thpl_event + 1 # the +1 is the heartbit event. Soat least there is a minimum of 1 data sent even if 0 events detected
        Energy_Used += (time_BLE_sens * volt_regulator * i_BLE_sens) * tot_events
        time_sleep -= (time_BLE_sens * tot_events)

        # Energy Consumed by the node in sleep mode
        i_sl = i_sleep_PIR if PIR_on_off == 1 else i_sleep
        # increase energy leakeage y the supercapacitor
        if SC_volt >= 5.15:
            i_sl += 0.0000075
        #i_sl += (SC_volt - SC_volt_die) * 0.000001

        Energy_Used += (time_sleep * volt_regulator * i_sl)

    Energy_Prod = next_wake_up_time_sec * p_solar_1_lux * light

    #Energy_Prod = next_wake_up_time_sec * p_solar_1_lux * light
    #if light > 50:
    #    print(light, next_wake_up_time_sec)
    #    print(Energy_Prod, Energy_Used)
    #    print(Energy_Prod/next_wake_up_time_sec, Energy_Used/next_wake_up_time_sec)
    #    exit()
    #print(Energy_Prod, Energy_Used, Energy_Rem, SC_volt, event)

    # Energy cannot be lower than 0
    #Energy_Rem = max(Energy_Rem - Energy_Used + Energy_Prod, 0)
    #Energy_Rem = Energy_Rem - Energy_Used + Energy_Prod
    #Energy_Rem += Energy_Prod
    #Energy_Rem -= 0.15
    i_diff_used = Energy_Used/(volt_regulator * next_wake_up_time_sec)
    i_diff_prod = Energy_Prod/(SC_volt * next_wake_up_time_sec)
    #i_diff_prod = (i_solar_200_lux/200) * light
    #SC_volt_used = np.sqrt((0.15/SC_size)*2)
    #SC_volt -= SC_volt_used
    #SC_volt = np.sqrt((Energy_Rem/SC_size)*2)
    #SC_volt = np.sqrt((Energy_Rem/SC_size)*2)
    SC_volt -= ((next_wake_up_time_sec*i_diff_used)/SC_size)
    SC_volt += ((next_wake_up_time_sec*i_diff_prod)/SC_size)

    # Setting Boundaries for Voltage

    if SC_volt > SC_volt_max:
        SC_volt = np.array([SC_volt_max])
    if SC_volt < SC_volt_min:
        SC_volt = np.array([SC_volt_min])
    #print("ou: ", SC_volt)

    return SC_volt, Energy_Prod, Energy_Used

def Energy_test(SC_volt):

    #E_1 = SC_volt * SC_volt * 0.5 * 1.5

    #E_2 = E_1 - 0.156693
    #Bt(seconds) = [C(Vcapmax - Vcapmin)/Imax]
    SC_volt = SC_volt - ((3600*0.0000047)/SC_size)
    #print(((3600*0.0000047)/SC_size), SC_volt)
    #exit()
    #DS = np.sqrt((E_2/1.5)*2) - np.sqrt((E_1/1.5)*2)
    #print(DS)
    #exit()
    #SC_volt_rem = np.sqrt((0.156693/1.5)*2)
    #SC_volt = SC_volt - 0.019

    #SC_volt += DS

    return SC_volt, 0, 0


def light_event_func_new(t_now, next_wake_up_time, mode, PIR_on_off, PIR_events_found_dict, light_prev, light_div, file_data, data_pointer): # check how many events are on this laps of time
        #time_buff = []; light_buff = []; PIR_buff = []; temp_buff = []; hum_buff = []; press_buff = []
        light_buff = [];
        PIR_event_gt = 0; event_found = 0
        thpl_event_gt = 0
        hold = 0; no_event = 0

        #with open(path_light_data, 'r') as f:
        #    file_data = f.readlines()
        #print(data_pointer)
        for i in range(data_pointer, len(file_data)):
            PIR = 0
            line_split = file_data[i].split("|")
            #if file_data[i] != '\n':
            #    try:
            #        test = 0; test += float(line_split[2]); test += float(line_split[4]); test += float(line_split[10])
            #    except:
            #        continue
            #    if test > 0: # Data read is valid
            check_time = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
            if hold == 0:
                old_time = check_time
            t_diff = int((check_time - old_time).total_seconds()/60)
            no_event = 1 if t_diff >= 60 else 0

            #print(t_now, next_wake_up_time, check_time, data_pointer, t_diff, no_event)

            if t_now <= check_time and check_time < next_wake_up_time:
                light_t = int(line_split[8])
                light_t = int(int(light_t)/light_div)
                light_t = light_max if light_t > light_max else light_t
                light_buff.append(light_t)
                PIR = int(line_split[6])
                PIR_event_gt += PIR

                if PIR == 0 and no_event == 0 and hold != 0:
                    thpl_event_gt += 1

            #if (mode == 1 or mode == 2) and PIR > 0 and PIR_on_off > 0:
            #    if check_time.time() not in PIR_events_found_dict:
            #        PIR_events_found_dict.append(check_time.time())
            if check_time >= next_wake_up_time:
                data_pointer = i
                break

            old_time = check_time
            hold = 1

        if len(light_buff) == 0:
            light_buff = [light_prev]

        light = sum(light_buff)/len(light_buff)

        #print(t_now, next_wake_up_time, thpl_event_gt)
        #print("exit", thpl_event_gt, data_pointer)
        #sleep(5)

        return light, PIR_event_gt, PIR_events_found_dict, thpl_event_gt, data_pointer

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


def plot_hist_low_level(data, tot_rew, title, energy_used, accuracy):

    tot_subplot = 4; sub_p = 1

    plt.figure(1)
    ax1 = plt.subplot(511)
    plt.title(('{2}. Tot Rew: {0}, Energy Used: {1}, Acc: {3}%').format(round(tot_rew, 5), round(energy_used, 5), title, accuracy))
    #plt.title(title_final, fontsize = 17)
    plt.plot(data["Time"], data["Light"], 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax1.set_xticklabels([])
    plt.grid(True)

    ax3 = plt.subplot(512)
    plt.plot(data["Time"], data["SC_Volt"], 'm.', label = 'SC Voltage', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(2.3, 3.7)
    plt.ylim(2.7, 5.6)
    ax3.set_xticklabels([])
    plt.grid(True)

    ax2 = plt.subplot(513)
    #plt.plot(Time, PIR, 'k.', label = 'PIR detection', markersize = 15)
    #plt.plot(Time, PIR_miss, 'r.', Time, PIR_det, 'k.', label = 'PIR detection', markersize = 15, )
    plt.plot(data["Time"], data["PIR_event_miss"], 'r.', label = 'Missed', markersize = 15)
    plt.plot(data["Time"], data["PIR_event_det"], 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="upper left", prop={'size': 9})
    ax2.set_xticklabels([])
    plt.grid(True)

    ax4 = plt.subplot(514)
    plt.plot(data["Time"], data["PIR_ON_OFF"], 'y.', label = 'RL Action', markersize = 15)
    plt.plot(data["Time"], data["PIR_gt"], 'b.', label = 'SS Action', markersize = 15)
    plt.ylabel('PIR\nOn_Off\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center", prop={'size': 9})
    plt.ylim(0)
    ax4.set_xticklabels([])
    plt.grid(True)

    ax5 = plt.subplot(515)
    plt.plot(data["Time"], data["State_Trans"], 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[min]', fontsize=15)
    plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    #ax5.set_xticklabels([])
    plt.grid(True)

    xfmt = mdates.DateFormatter('%m/%d %H')
    ax5.xaxis.set_major_formatter(xfmt)
    plt.show()


    # Start second graph

    plt.figure(1)
    ax1 = plt.subplot(511)
    plt.title(('{2}. Tot Rew: {0}, Energy Used: {1}, Acc: {3}%').format(round(tot_rew, 5), round(energy_used, 5), title, accuracy))
    #plt.title(title_final, fontsize = 17)
    plt.plot(data["Time"], data["Light"], 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax1.set_xticklabels([])
    plt.grid(True)

    ax3 = plt.subplot(512)
    plt.plot(data["Time"], data["SC_Volt"], 'm.', label = 'SC Voltage', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(2.3, 3.7)
    plt.ylim(2.7, 5.6)
    ax3.set_xticklabels([])
    plt.grid(True)

    ax6 = plt.subplot(513)
    plt.plot(data["Time"], data["thpl_event_miss"], 'r.', label = 'Missed', markersize = 15)
    plt.plot(data["Time"], data["thpl_event_det"], 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('THPL\nEvents\n[num]', fontsize=15)
    plt.legend(loc="upper left", prop={'size': 9})
    ax6.set_xticklabels([])
    plt.grid(True)

    ax4 = plt.subplot(514); sub_p += 1
    plt.plot(data["Time"], data["THPL_ON_OFF"], 'y.', label = 'RL Action', markersize = 15)
    plt.plot(data["Time"], data["THPL_gt"], 'b.', label = 'SS Action', markersize = 15)
    plt.ylabel('THPL\nOn_Off\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center", prop={'size': 9})
    plt.ylim(0)
    ax4.set_xticklabels([])
    plt.grid(True)

    ax5 = plt.subplot(515); sub_p += 1
    plt.plot(data["Time"], data["State_Trans"], 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[min]', fontsize=15)
    plt.xlabel('Time [h:m]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    #ax5.set_xticklabels([])
    plt.grid(True)

    xfmt = mdates.DateFormatter('%m/%d %H')
    ax5.xaxis.set_major_formatter(xfmt)
    plt.show()
    plt.close("all")

    '''
    ax7 = plt.subplot(514)
    plt.plot(Time, Reward, 'b.', label = 'Reward', markersize = 15)
    plt.ylabel('Reward\n[num]', fontsize=12)
    plt.xlabel('Time [hh:mm]', fontsize=15)
    #ax8.tick_params(axis='both', which='major', labelsize=12)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)

    ax6 = plt.subplot(tot_subplot, 1, 6)
    plt.plot(Time, thpl_miss, 'r.', label = 'Missed', markersize = 15)
    plt.plot(Time, thpl_det, 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('THPL\nevents\n[num]', fontsize=15)
    plt.legend(loc="center left", prop={'size': 9})
    ax6.set_xticklabels([])
    plt.grid(True)

    ax7 = plt.subplot(tot_subplot, 1, 7)
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
    '''

def plot_all_PIR_THPL(data, tot_rew, title, energy_used, accuracy):

    tot_subplot = 5; sub_p = 1

    plt.figure(1)
    ax1 = plt.subplot(711)
    plt.title(('{2}. Tot Rew: {0}, Energy Used: {1}, Acc: {3}%').format(round(tot_rew, 5), round(energy_used, 5), title, accuracy))
    #plt.title(title_final, fontsize = 17)
    plt.plot(data["Time"], data["Light"], 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax1.set_xticklabels([])
    plt.grid(True)

    ax3 = plt.subplot(712)
    plt.plot(data["Time"], data["SC_Volt"], 'm.', label = 'SC Voltage', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(2.3, 3.7)
    plt.ylim(2.7, 5.6)
    ax3.set_xticklabels([])
    plt.grid(True)

    ax2 = plt.subplot(713)
    #plt.plot(Time, PIR, 'k.', label = 'PIR detection', markersize = 15)
    #plt.plot(Time, PIR_miss, 'r.', Time, PIR_det, 'k.', label = 'PIR detection', markersize = 15, )
    plt.plot(data["Time"], data["PIR_event_miss"], 'r.', label = 'Missed', markersize = 15)
    plt.plot(data["Time"], data["PIR_event_det"], 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="upper left", prop={'size': 9})
    ax2.set_xticklabels([])
    plt.grid(True)

    ax4 = plt.subplot(714)
    plt.plot(data["Time"], data["PIR_ON_OFF"], 'y.', label = 'RL Action', markersize = 15)
    plt.plot(data["Time"], data["PIR_gt"], 'b.', label = 'SS Action', markersize = 15)
    plt.ylabel('PIR\nOn_Off\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center", prop={'size': 9})
    plt.ylim(0)
    ax4.set_xticklabels([])
    plt.grid(True)

    ax5 = plt.subplot(715)
    plt.plot(data["Time"], data["thpl_event_miss"], 'r.', label = 'Missed', markersize = 15)
    plt.plot(data["Time"], data["thpl_event_det"], 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('THPL\nEvents\n[num]', fontsize=15)
    plt.legend(loc="upper left", prop={'size': 9})
    ax5.set_xticklabels([])
    plt.grid(True)

    ax6 = plt.subplot(716); sub_p += 1
    plt.plot(data["Time"], data["THPL_ON_OFF"], 'y.', label = 'RL Action', markersize = 15)
    plt.plot(data["Time"], data["THPL_gt"], 'b.', label = 'SS Action', markersize = 15)
    plt.ylabel('THPL\nOn_Off\n[num]', fontsize=15)
    #plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc="center", prop={'size': 9})
    plt.ylim(0)
    ax6.set_xticklabels([])
    plt.grid(True)

    ax7 = plt.subplot(717)
    plt.plot(data["Time"], data["State_Trans"], 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[min]', fontsize=15)
    plt.xlabel('Time [hours]', fontsize=15)
    #plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    #ax5.set_xticklabels([])
    plt.grid(True)

    #xfmt = mdates.DateFormatter('%m/%d %H')
    xfmt = mdates.DateFormatter('%H')
    ax7.xaxis.set_major_formatter(xfmt)
    plt.show()
    plt.close("all")

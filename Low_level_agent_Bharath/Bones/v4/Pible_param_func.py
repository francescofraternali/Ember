import datetime
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Parameters to change
SC_volt_die = 3.0 # Voltage at which the simulator consider the node death
SC_begin = 4.0 # Super Capacitor initial voltage level. Put a number between 5.4 (i.e. max) and 2.3 (i.e. min)
using_PIR = True # PIR active and used to detect people
PIR_events = 100 # Number of PIR events detected during a day. This could happen also when light is not on
using_Accelerometer = False # Activate only if using Accelerometer

# RL Parameters
episode_lenght = 1 # in days

# DO NOT MODIFY! POWER CONSUMPTION PARAMETERS! Change them only if you change components.
SC_volt_min = 2.3; SC_volt_max = 5.5; SC_size = 1.5; SC_volt_die = 3.0

# Board and Components Consumption
i_sleep = 0.0000032;
i_sens =  0.000100; time_sens = 0.2
i_PIR_detect = 0.000102; time_PIR_detect = 2.5
i_accel_sens = 0.0026; accel_sens_time = 0.27
#i_sleep_PIR = i_sleep + 0.000001 # original
#i_sleep_PIR = i_sleep + 0.0001
i_sleep_PIR = i_sleep + 0.000001

'''
if using_PIR == True:
#    i_sleep += 0.000001
    if PIR_events != 0:
        PIR_events_time = (24*60*60)/PIR_events  # PIR events every "PIR_events_time" seconds. Averaged in a day
    else:
        PIR_events_time = 0
else:
    PIR_events = 0
'''

# if using_Accelerometer:
#    i_sleep += 0.000008

# Communication (wake up and transmission) and Sensing Consumption
i_wake_up_advertise = 0.00006; time_wake_up_advertise = 11
i_BLE_comm = 0.00025; time_BLE_comm = 4
i_BLE_sens = ((i_wake_up_advertise * time_wake_up_advertise) + (i_BLE_comm * time_BLE_comm))/(time_wake_up_advertise + time_BLE_comm)
time_BLE_sens = time_wake_up_advertise + time_BLE_comm

#i_BLE_sens = 0.000210; time_BLE_sens = 6.5

# Solar Panel Production
v_solar_200_lux = 1.5; i_solar_200_lux = 0.000031
p_solar_1_lux = (v_solar_200_lux * i_solar_200_lux) / 200.0


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


def light_event_func(t_now, next_wake_up_time, mode, PIR_on_off, PIR_events_found_dict, light_prev, light_div, file_data, data_pointer): # check how many events are on this laps of time
        #time_buff = []; light_buff = []; PIR_buff = []; temp_buff = []; hum_buff = []; press_buff = []
        light_buff = [];
        PIR_event_gt = 0; event_found = 0
        thpl_event_gt = 0

        hold = 0; no_event = 0
        for i in range(data_pointer, len(file_data)):
            PIR = 0
            splitted = file_data[i].split("|")
            check_time = datetime.datetime.strptime(splitted[0], '%m/%d/%y %H:%M:%S')
            if hold == 0:
                old_time = check_time
            t_diff = int((check_time - old_time).total_seconds()/60)
            no_event = 1 if t_diff >= 60 else 0

            #print(t_now, next_wake_up_time, check_time)

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
                data_pointer = i
                break

            old_time = check_time
            hold = 1

        if len(light_buff) == 0:
            light_buff = [light_prev]

        light = sum(light_buff)/len(light_buff)

        #print(t_now, check_time, next_wake_up_time, t_diff, thpl_event_gt)
        #print("exit")

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


def plot_hist_low_level(Time, Light, Mode, Sens_OnOff, State_Trans, Reward,
                        SC_Volt, PIR_det, PIR_miss, thpl_det, thpl_miss, tot_rew,
                        event_detect, tot_events, Dict_Events, title_final, energy_used, accuracy):
    plt.figure(1)
    ax1 = plt.subplot(711)
    plt.title(('{2}. Tot Rew: {0}, Energy Used: {1}, Acc: {3}%').format(round(tot_rew, 5), round(energy_used, 5), title_final, accuracy))
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
    plt.plot(Time, Sens_OnOff, 'y.', label = 'Action', markersize = 15)
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
    plt.legend(loc="center left", prop={'size': 9})
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

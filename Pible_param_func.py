import datetime
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from time import sleep
import math

# Parameters to change
SC_volt_die = 3.0 # Voltage at which the simulator consider the node death
SC_begin = 4.0 # Super Capacitor initial voltage level. Put a number between 5.4 (i.e. max) and 2.3 (i.e. min)

# RL Parameters
episode_lenght = 1 # in days
light_max = 1000

# DO NOT MODIFY! POWER CONSUMPTION PARAMETERS! Change them only if you change components.
SC_volt_min = 2.3; SC_volt_max = 5.5; SC_size = 1.5; SC_volt_die = 3.0

# Board and Components Consumption
i_sens =  0.0004; time_sens = 0.2
i_PIR_detect = 0.000102; time_PIR_detect = 2.5
i_accel_sens = 0.0026; accel_sens_time = 0.27

# Communication (wake up and transmission) and Sensing Consumption
i_wake_up_advertise = 0.000065; time_wake_up_advertise = 11
i_BLE_comm = 0.0003; time_BLE_comm = 5;
i_BLE_sens = ((i_wake_up_advertise * time_wake_up_advertise) + (i_BLE_comm * time_BLE_comm))/(time_wake_up_advertise + time_BLE_comm)

# Solar Panel Production
v_solar_200_lux = 1.5; i_solar_200_lux = 0.0000333
p_solar_1_lux = (v_solar_200_lux * i_solar_200_lux) / 200.0


def Energy(SC_volt, light, PIR_or_thpl, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event):

    if int(PIR_or_thpl) > 0:
        i_sleep = 0.0000042;
        time_BLE_comm = 12; num_sens = 3
        time_BLE_sens = time_wake_up_advertise + time_BLE_comm
    else:
        time_BLE_comm = 5; num_sens = 1
        time_BLE_sens = time_wake_up_advertise + time_BLE_comm
        i_sleep = 0.00000345;

    i_sleep_PIR = i_sleep + 0.000002

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

        Energy_Used += (time_sleep * volt_regulator * i_sl)

    Energy_Prod = next_wake_up_time_sec * p_solar_1_lux * light


    i_diff_used = Energy_Used/(volt_regulator * next_wake_up_time_sec)
    i_diff_prod = Energy_Prod/(SC_volt * next_wake_up_time_sec)

    SC_volt -= ((next_wake_up_time_sec*i_diff_used)/SC_size)
    SC_volt += ((next_wake_up_time_sec*i_diff_prod)/SC_size)

    # Setting Boundaries for Voltage
    if SC_volt > SC_volt_max:
        SC_volt = np.array([SC_volt_max])
    if SC_volt < SC_volt_min:
        SC_volt = np.array([SC_volt_min])

    return SC_volt, Energy_Prod, Energy_Used

def light_event_func_new(t_now, next_wake_up_time, mode, PIR_on_off, PIR_events_found_dict, light_prev, light_div, file_data, data_pointer): # check how many events are on this laps of time

        light_buff = [];
        PIR_event_gt = 0; event_found = 0
        thpl_event_gt = 0
        hold = 0; no_event = 0

        for i in range(data_pointer, len(file_data)):
            PIR = 0
            line_split = file_data[i].split("|")

            check_time = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
            if hold == 0:
                old_time = check_time
            t_diff = int((check_time - old_time).total_seconds()/60)
            no_event = 1 if t_diff >= 60 else 0

            if t_now <= check_time and check_time < next_wake_up_time:
                light_t = int(line_split[8])
                light_t = int(int(light_t)/light_div)
                light_t = light_max if light_t > light_max else light_t
                light_buff.append(light_t)
                PIR = int(line_split[6])
                PIR_event_gt += PIR

                if PIR == 0 and no_event == 0 and hold != 0:
                    thpl_event_gt += 1

            if check_time >= next_wake_up_time:
                data_pointer = i
                break

            old_time = check_time
            hold = 1

        if len(light_buff) == 0:
            light_buff = [light_prev]

        light = sum(light_buff)/len(light_buff)


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


def plot(data, tot_rew, title, energy_used, accuracy):

    plt.figure(1)
    ax1 = plt.subplot(411)
    plt.title(('{1}. Tot Reward: {0}, Accuracy: {2}%').format(round(tot_rew, 5), title, accuracy))
    plt.plot(data["Time"], data["Light"], 'b-', label = 'Light', markersize = 15)
    plt.ylabel('Light\n[lux]', fontsize=15)
    ax1.set_xticklabels([])
    plt.grid(True)

    ax3 = plt.subplot(412)
    plt.plot(data["Time"], data["SC_Volt"], 'm.', label = 'SC Voltage', markersize = 10)
    plt.ylabel('SC [V]\nVolt', fontsize=15)
    plt.ylim(2.7, 5.6)
    ax3.set_xticklabels([])
    plt.grid(True)

    ax2 = plt.subplot(413)
    plt.plot(data["Time"], data["PIR_event_miss"], 'r.', label = 'Missed', markersize = 15)
    plt.plot(data["Time"], data["PIR_event_det"], 'k.', label = 'Detected', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=15)
    plt.legend(loc="upper left", prop={'size': 9})
    ax2.set_xticklabels([])
    plt.grid(True)

    ax4 = plt.subplot(414)
    plt.plot(data["Time"], data["PIR_ON_OFF"], 'y.', label = 'RL Action', markersize = 15)
    plt.plot(data["Time"], data["PIR_gt"], 'b.', label = 'SS Action', markersize = 15)
    plt.ylabel('PIR\nOn_Off\n[num]', fontsize=15)
    plt.legend(loc="center", prop={'size': 9})
    plt.ylim(0)
    plt.xlabel('Time [month/day Hour]', fontsize=15)
    ax4.set_xticklabels([])
    plt.grid(True)

    xfmt = mdates.DateFormatter('%m/%d %H')
    ax4.xaxis.set_major_formatter(xfmt)
    plt.show()

    plt.close("all")

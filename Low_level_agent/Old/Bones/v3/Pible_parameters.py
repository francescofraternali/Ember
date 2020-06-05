
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


'''
# Parameters to change
state_trans = 900 # State transition in seconds. I.e. the system takes a new action every "state_trans" seconds. It also corresponds to the communication time
sens_time = 60 # Sensing time in seconds.
transm_thres_light = 100 # value in lux
SC_begin = 4.0 # Super Capacitor initial voltage level. Put a number between 5.4 (i.e. max) and 2.3 (i.e. min)
SC_volt_die = 3.0 # Voltage at which the simulator consider the node death
using_temp = False
using_light = True
using_PIR = True # PIR active and used to detect people
PIR_events = 100 # Number of PIR events detected during a day. This could happen also when light is not on
using_Accelerometer = False # Activate only if using Accelerometer
'''

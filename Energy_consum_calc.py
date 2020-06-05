import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('../Ember_Code/')
import Pible_param_func
from Ember_RL_func import valid_line
from Pible_param_func import *

def energy_consumption():
    total_min_experiments = 60*1 # A day
    SC_volt = 3.2
    avg_light_per_experim = 0

    PIR_on_off = 1; PIR_event = 0
    thpl_on_off = 1; thpl_event = 0

    next_wake_up_time = 60 # min
    energy_prod_tot = 0
    energy_used_tot = 0

    while total_min_experiments > 0:
        SC_volt, energy_prod, energy_used = Pible_param_func.Energy(SC_volt, avg_light_per_experim, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event)
        total_min_experiments -= next_wake_up_time
        PIR_event =-1
        thpl_event =-1
        PIR_event = 0 if PIR_event < 0 else PIR_event
        thpl_event = 0 if thpl_event < 0 else thpl_event
        #print(energy_prod)
        energy_prod_tot += energy_prod
        energy_used_tot += energy_used

    #print("energy prod", energy_prod_tot)
    print("energy used", energy_used_tot)
    print("curr used", energy_used_tot/(SC_volt*next_wake_up_time*60))
    print("rew normalized", -0.1 * energy_used_tot)

    energy_prod_tot = 0
    energy_used_tot = 0

    PIR_on_off = 1; PIR_event = 3; next_wake_up_time = 3
    SC_volt, energy_prod, energy_used = Pible_param_func.Energy(SC_volt, avg_light_per_experim, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event)
    energy_used_tot += energy_used

    PIR_on_off = 0; PIR_event = 0; next_wake_up_time = 57
    SC_volt, energy_prod, energy_used = Pible_param_func.Energy(SC_volt, avg_light_per_experim, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event)
    energy_used_tot += energy_used

    print("energy used", energy_used_tot)

def extract(path):
    Time_hist = []; Volt_hist = [];
    with open(path, 'r') as f:
        for line in f:
            if valid_line(line):
                line_split = line.split("|")
                time = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
                SC_Pure = int(line_split[5])
                #print(time)
                if SC_Pure == 0 and Light == 0 and PIR == 0:
                    pass
                else:
                    SC = SC_Pure
                    Time_hist.append(time); Volt_hist.append(SC)

    offset = Time_hist[0]
    for i, element in enumerate(Time_hist):
        difference = Time_hist[i] - offset
        Time_hist[i] = difference.total_seconds()/(60*60)

    return Time_hist, Volt_hist


def simulate_compare_discharge(compare, SC_volt, PIR_on_off, thpl_on_off, title):
    avg_light_per_experim = 0; next_wake_up_time = 60; PIR_event = 0; thpl_event = 0; time = 0; tot_energy_used = 0; PIR_or_thpl = 1;
    Time = [time]; SC_volt_list = [(SC_volt/SC_volt_max)*100]
    #Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size
    while True:
        SC_volt, energy_prod, energy_used = Pible_param_func.Energy(SC_volt, avg_light_per_experim, PIR_or_thpl, PIR_on_off, thpl_on_off, next_wake_up_time, PIR_event, thpl_event)
        time += 1
        Time.append(time)
        tot_energy_used += energy_used
        SC_volt_list.append((SC_volt/SC_volt_max)*100)
        if SC_volt <= SC_volt_die:
        #if Energy_Rem <= SC_volt_die* SC_volt_die * 0.5 * 1.5:
            break

    time, volt = extract(compare)

    print("Tot energy used: ", tot_energy_used)
    print("Tot energy available: ", (SC_volt_max * SC_volt_max * SC_size * 0.5) - (SC_volt_die * SC_volt_die * SC_size * 0.5))
    #plt.title(('{2}. Tot Rew: {0}, Energy Used: {1}, Acc: {3}%').format(round(tot_rew, 5), round(energy_used, 5), title_final, accuracy))
    plt.title(title, fontsize = 17)
    plt.plot(Time, SC_volt_list, 'b-', label = 'SC_perc Sim', markersize = 15)
    plt.plot(time, volt, 'r-', label = 'SC_perc Real', markersize = 15)
    plt.ylabel('SC \n[%]', fontsize=15)
    plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    plt.grid(True)
    plt.show()


    return 0



# Main
#energy_consumption()
simulate_compare_discharge("home_test_EH_FF17_discharge_all_PIR_sens_on.txt", (100/100)*SC_volt_max, PIR_on_off=1, thpl_on_off=1, title="All On")
simulate_compare_discharge("2142_Corridor_Batt_FF34_Discharge.txt", (90/100)*SC_volt_max, PIR_on_off=0, thpl_on_off=1, title="PIR off Sens On")
simulate_compare_discharge("home_test_EH_FF87_discharge_only_sens_on.txt", (100/100)*SC_volt_max, PIR_on_off=0, thpl_on_off=1, title="PIR off Sens On")
simulate_compare_discharge("home_test_EH_FF61.txt", (100/100)*SC_volt_max, PIR_on_off=1, thpl_on_off=0, title="PIR On Sens Off and not available")
simulate_compare_discharge("2140_Door_Batt_FF89_Discharge.txt",  (65/100)*SC_volt_max, PIR_on_off=0, thpl_on_off=0, title="All Off")
simulate_compare_discharge("2140_Door_Batt_FF95_Discharge.txt",  (74/100)*SC_volt_max, PIR_on_off=0, thpl_on_off=0, title="All Off")

#simulate_compare_discharge("home_test_EH_FF51.txt")

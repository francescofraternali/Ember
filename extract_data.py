import os
import pickle
from time import sleep
import sys
#sys.path.append("../")
import Pible_param_func
import Ember_RL_func

files = os.listdir("Save_Data")
#print(files)
tot_acc = 0; energy_used_tot = 0; tot_thpl_on_min = 0; tot_pir_on_min = 0
acc = []; sc_volt = []; inj_prob_hours = []
num_files = 0; tot_rew = 0
title = "Middle of an Office, THPL."
for file in files:
    with open("Save_Data/" + file, "rb") as f:
        file_spl = file.split('_')
        if 'test' in file_spl:
            print(file)
            data = pickle.load(f)
            num_files += 1
            accuracy = Ember_RL_func.calc_accuracy(data["info"])
            acc.append(accuracy)
            tot_acc += accuracy
            #print(data["Reward"])
            tot_rew += sum(data["Reward"])
            sc_volt.append(data["info"]["SC_volt"])
            #inj_prob_hours.append(data["info"]["hours_inj_prob"])
            energy_used_tot += float(data["info"]["energy_used"])
            THPL = data["THPL_ON_OFF"]; State_Trans = data["State_Trans"]
            PIR = data["PIR_ON_OFF"]

            for i, el in enumerate(THPL):
                if el == 1:
                    tot_thpl_on_min += int(State_Trans[i])

            for i, el in enumerate(PIR):
                if el == 1:
                    tot_pir_on_min += int(State_Trans[i])

            #Pible_param_func.plot_hist_low_level(data, 0, title, float(data["info"]["energy_used"]), accuracy)
            Pible_param_func.plot_all_PIR_THPL(data, 0, title, float(data["info"]["energy_used"]), accuracy)

print("Accuracy: ", acc)
print("Tot Rew: ", tot_rew)
print("SC Volt: ", sc_volt)
#print("Inj Prob Hours: ", inj_prob_hours)
print("accuracy: ", round(tot_acc/num_files, 1))
print("SC start: ", 4, ". SC end: ", sc_volt[-1], ". SC volt diff: ", 4 - sc_volt[-1])
print("Tot hours PIR on per day: ", tot_pir_on_min/(num_files*60))
print("Tot hours THPL on per day: ", tot_thpl_on_min/(num_files*60))

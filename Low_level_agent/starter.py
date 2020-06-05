import os
import subprocess
#sys.path.append('../Ember_Code/')
path = os.getcwd()
take_from = "../Ember_Code/"
proc = subprocess.Popen("cp " + take_from + "Main_low_level_agent.py .", shell=True)
(out, err) = proc.communicate()
proc = subprocess.Popen("cp " + take_from + "Pible_class_low_level_agent.py .", shell=True)
(out, err) = proc.communicate()
proc = subprocess.Popen("cp " + take_from + "Ember_RL_func.py .", shell=True)
(out, err) = proc.communicate()
proc = subprocess.Popen("cp " + take_from + "Pible_param_func.py .", shell=True)
(out, err) = proc.communicate()
print("Copy good")

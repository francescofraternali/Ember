from Pible_param_func import *
import numpy as np
import datetime
from time import sleep
import subprocess
import json
import threading
from subprocess import PIPE, Popen, call
import random

#from training_pible import light_divider
check_max = 5


def valid_line(line):
    #print(line)
    if line != '\n':
        line_split = line.split("|")
        error = 'list index out of range'
        if error not in line_split:
            return True

    return False


def select_input_data(path_light_data, start_data_date, end_data_date):
    file_data = []
    #for line in reversed(list(open(path_light_data))):
    for i, line in enumerate(list(open(path_light_data))):
        #print("line", line)
        if valid_line(line):
            line_split = line.split("|")
            #try:
            #    test = 0; test += float(line_split[2]); test += float(line_split[4]); test += float(line_split[10])
            #except:
            #    continue
            #if test > 0:
            checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
            if start_data_date <= checker and checker < end_data_date:
                file_data.append(line)
                #break
            if checker >= end_data_date:
                #data_pointer = i
                break
    if len(file_data) == 0:
        #print("No new rows found")
        pass
    #data_pointer -= len(file_data)

    return file_data

def select_input_starter(path_light_data, start_data_date, num_light_input, num_sc_volt_input):

    start_light_list = [0] * num_light_input
    start_volt_list = [SC_volt_die] * num_sc_volt_input
    count_light = 0; count_volt = 0
    starter_data = []

    for line in reversed(list(open(path_light_data))):
        #for line in list(open(path_light_data)):
        #print("line", line)
        if valid_line(line):
            line_split = line.split("|")
            checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
            #print(checker, start_data_date)
            if checker <= start_data_date:
                #print(count_light, count_volt)
                starter_data.append(line)
                if count_light < num_light_input:
                    light = int(line_split[8])
                    light = light_max if light > light_max else light
                    start_light_list[count_light] = light
                    count_light += 1

                if count_volt < num_sc_volt_input:
                    volt = (float(line_split[5]) * SC_volt_max) / 100
                    start_volt_list[count_volt] = round(volt, 2)
                    count_volt += 1

                #print(start_light_list, start_volt_list)

                if count_volt >= num_sc_volt_input and count_light >= num_light_input:
                    break

    return start_light_list, start_volt_list, starter_data

def build_inputs(time, sc_volt, num_hours_input, num_minutes_input, last_light_list, last_volt_list):
    list = []
    for i in range(0, num_hours_input):
        #value = time - datetime.timedelta(hours=1)
        list.append(time.hour)
        time = time - datetime.timedelta(hours=1)
        #list.append(value.hour)
    hour_array = np.array(list)

    list = []
    for i in range(0, num_minutes_input):
        #value = time - datetime.timedelta(minutes=1)
        list.append(time.minute)
        time = time - datetime.timedelta(minutes=1)
        #list.append(value.minute)
    minute_array = np.array(list)

    light_array = np.array(last_light_list)

    sc_array = np.array(last_volt_list)

    return hour_array, minute_array, light_array, sc_array

def updates_arrays(hour_array, minute_array, light_array, SC_Volt_array, time, light, SC_temp):
    hour_array = np.roll(hour_array, 1)
    hour_array[0] = time.hour

    minute_array = np.roll(minute_array, 1)
    minute_array[0] = time.minute

    list = []
    for i in range(0, 24):
        #value = time - datetime.timedelta(hours=1)
        list.append(time.hour)
        time = time - datetime.timedelta(hours=1)
        #list.append(value.hour)
    hour_array = np.array(list)

    list = []
    for i in range(0, 60):
        list.append(time.minute)
        #value = time - datetime.timedelta(minutes=1)
        time = time - datetime.timedelta(minutes=1)
        #list.append(value.minute)
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

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.stdout = None
        self.stderr = None

    def run(self, timeout):
        def target():
            #print('Thread started')
            self.process = Popen(self.cmd,  stdout=PIPE, stderr=PIPE, shell=True)
            self.stdout, self.stderr = self.process.communicate()
            #print('Thread finished')

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        try:
            if thread.is_alive():
                #print('Terminating process')
                self.process.terminate()
                thread.join()
        except:
            print("something wrong in the process. Maybe it was already terminated?")

        return self.stdout, self.stderr, self.process.returncode


def checker(message, timeout):
    global check_max
    command = Command(message)
    for i in range(0, check_max):
        #try:
        (out, err, check) = command.run(timeout=60)

        if check == -15:
            print("Base Station not answering. Trial num: " + str(i))
            sleep(900)
        else:
            break

    if check == -15:
        #check_max += 10
        message = "Base station not aswering, something wrong, resetting base station..."
        print(message)

    else:
        check_max = 5

    return out, check

def sync_input_data(pwd, bs_name, File_name, destination):
    #proc = subprocess.Popen("sshpass -p " + pwd +" scp -r -o StrictHostKeyChecking=no "+bs_name+":/home/pi/BLE_GIT/Data/"+File_name+" .", stdout=subprocess.PIPE, shell=True)
    #command = "sshpass -p " + pwd +" scp -r -o StrictHostKeyChecking=no "+bs_name+":/home/pi/BLE_GIT/Data/" + File_name + " Temp_"+ File_name
    command = "sshpass -p {0} scp -r -o StrictHostKeyChecking=no {1}:/home/pi/Base_Station_20/Data/{2} {3}Temp_{2}".format(pwd, bs_name, File_name, destination)
    out, check = checker(command, 60)

    if check == 0: #Everything went right
        with open(destination + File_name, 'ab') as outfile:
            with open(destination + "Temp_" + File_name, 'rb') as infile:
                outfile.write(infile.read())

        #print("Removing file ...")
        sleep(0.5)
        #command ="sshpass -p " + val + " ssh " + key +" rm /home/pi/BLE_GIT/Data/"+ file
        #command = "sshpass -p " + pwd +" ssh " + bs_name + " rm /home/pi/BLE_GIT/Data/" + File_name
        command = "sshpass -p {0} ssh {1} rm /home/pi/Base_Station_20/Data/{2}".format(pwd, bs_name, File_name)
        out, check = checker(command, 60)
        #command = "rm Temp_" + File_name
        command = "rm {1}Temp_{0}".format(File_name, destination)
        out, check = checker(command, 60)
        print("Merge OK")
    elif check == 1:
        print("No new file to merge. Check Check Check.")


def sync_action(file, action): # Now Let's update the action to the action file
    act_1, act_2 = action_decode(action)
    with open(file, 'r') as f:
        dic = json.load(f)

    act_3 = dic["Action_3"]
    dic["Action_1"] = act_1
    dic["Action_2"] = act_2
    dic["Action_3"] = act_3

    with open(file, 'w') as f:
        json.dump(dic, f)

def action_decode(action): # Action_1 = PIR_onoff + State_transition; Action_2 = Sensing sensitivity
    if len(action) < 3:
        PIR = int(action[0])
        thpl = int(action[1])
        if PIR == 0 and thpl == 0: # everything off
            Action_1 = '3C'; Action_2 = '01'
        elif PIR == 1 and thpl == 0:
            Action_1 = 'BC'; Action_2 = '01'
        elif PIR == 0 and thpl == 1:
            Action_1 = '3C'; Action_2 = '0B'
        elif PIR == 1 and thpl == 1:
            Action_1 = 'BC'; Action_2 = '0B'

    return Action_1, Action_2

def sync_ID_file_to_BS(pwd, bs_name, file_local_address, destination):

    call("sshpass -p " + pwd + " scp -r -o StrictHostKeyChecking=no " + file_local_address + " " + bs_name + ":" + destination, shell=True)

    '''
        if run == 0:
            send_email("Something wrong while synching actions. Check!")
    '''

def add_random_volt(SC_Volt_array):
    v_min = round(SC_Volt_array[0] - SC_volt_die - 0.1, 1)
    v_max = round(SC_volt_max - SC_Volt_array[0] - 0.1, 1)
    #print(-v_min, v_max)
    k = round(random.uniform(-v_min, v_max), 1)
    #print("before ", SC_Volt_array)
    SC_Volt_array = [round(x + k , 2) for x in SC_Volt_array]
    for i in range(len(SC_Volt_array)):
        SC_Volt_array[i] = SC_volt_max if SC_Volt_array[i] > SC_volt_max else SC_Volt_array[i]
        SC_Volt_array[i] = SC_volt_die if SC_Volt_array[i] < SC_volt_die else SC_Volt_array[i]
    #print(k, SC_Volt_array)

    return SC_Volt_array

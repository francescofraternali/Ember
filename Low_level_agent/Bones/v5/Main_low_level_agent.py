""" Hierarchical RL for Pible
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import numpy as np
import gym
from ray.tune.logger import pretty_print
import ray
from ray import tune
from ray.tune import grid_search
import json
from Pible_param_func import *
from Pible_class_low_level_agent import SimplePible
import Ember_RL_func
import datetime
from time import sleep
import os
import subprocess
import glob
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
import multiprocessing
import getpass

def test_and_print_results(folder, iteration, start_date, end_date, title, curr_path, sc_volt_test, train_test_real):
    train_test_real = 'test' if train_test_real == 'train' else train_test_real
    path = glob.glob(save_agent_folder + folder  + '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)
    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["num_workers"] = 0
    config["explore"] = False
    config["env_config"] = {
       "settings": settings,
       "main_path": curr_path,
       "train/test": train_test_real,
       "start_test": start_date,
       "end_test": end_date,
       "sc_volt_start_test": sc_volt_test,
    }
    if train_test_real == "real":
        Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")
        file_light = settings[0]["file_light"]
        ID_temp = file_light.split('_')[-1]
        action_file = ID_temp.replace('.txt','_action.json')

    agent = ppo.PPOTrainer(config=config, env="simplePible")
    agent.restore(path[0])
    env = SimplePible(config["env_config"])
    obs = env.reset()
    tot_rew = 0;  energy_used_tot = 0;  energy_prod_tot = 0
    while True:
        #action_0_list = []; action_1_list = []
        #for i in range(0, 20):
        learned_action = agent.compute_action(
                observation = obs,
        )

        #    action_0_list.append(learned_action[0][0])
        #    action_1_list.append(learned_action[1][0][0])
        #action_0_select = max(set(action_0_list), key = action_0_list.count)
        #action_1_select = max(set(action_1_list), key = action_1_list.count)
        #learned_action = [action_0_select, action_1_select]
        print("action_taken: ", learned_action)

        if train_test_real == "real":
            Ember_RL_func.sync_action(action_file, learned_action)
            Ember_RL_func.sync_ID_file_to_BS(settings[0]["pwd"], settings[0]["bs_name"], action_file, "/home/pi/Base_Station_20/ID/")
            if len(learned_action) > 2:
                print("sleeping " + str(learned_action[0][1]) + " mins")
                sleep(int(learned_action[0][1]) * 60)
            else:
                print("sleeping " + str(60) + " mins")
                sleep(60 * 60)
            Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")

        obs, reward, done, info = env.step(learned_action)
        print(obs, reward, done, info)

        energy_used_tot += float(info["energy_used"])
        energy_prod_tot += float(info["energy_prod"])
        tot_rew += reward

        if done:
            obs = env.reset()
            start_date = start_date + datetime.timedelta(days=episode_lenght)
            if start_date >= end_date:
                print("done")
                break

    print("tot reward", round(tot_rew, 3))
    print("Energy Prod per day: ", energy_prod_tot/episode_lenght, "Energy Used: ", energy_used_tot/episode_lenght)
    print("Detected events averaged per day: ", (int(info["PIR_events_detect"]) +int(info["thpl_events_detect"]))/episode_lenght)
    print("Tot events averaged per day: ", (int(info["PIR_tot_events"]) +int(info["thpl_tot_events"]))/episode_lenght)
    if (int(info["PIR_events_detect"]) + int(info["thpl_events_detect"])) != 0 or (int(info["PIR_tot_events"]) + int(info["thpl_tot_events"])) != 0 :
        accuracy = float((int(info["PIR_events_detect"]) +int(info["thpl_events_detect"]))/(int(info["PIR_tot_events"]) +int(info["thpl_tot_events"])))
        accuracy = round(accuracy, 3)* 100
    else:
        accuracy = 0
    print("Accuracy: ", accuracy)

    if train_test_real != "real":
        env.render(tot_rew, title, energy_used_tot, accuracy)
    return path

def training_PPO(start_train_date, end_train_date, resume):
    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["lr"] = 1e-4
    config["num_workers"] = num_cores
    config["env_config"] = {
        "settings": settings,
        "main_path": curr_path,
        "start_train": start_train_date,
        "end_train": end_train_date,
        "train/test": "train",
        "sc_volt_start_train": sc_volt_train,
    }
    trainer = ppo.PPOTrainer(config=config, env="simplePible")
    max_min = -10000

    if resume_path != "":
        print("Restoring checkpoint: ", resume)
        sleep(5)
        trainer.restore(resume) # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(0, int(settings[0]["training_iterations"])):
        result = trainer.train()
        print(pretty_print(result))

        if int(result["training_iteration"]) % 10 == 0:
        #if max_min > int(result["episode_reward_min"])
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    # Find best checkpoint
    path = subprocess.getoutput('eval echo "~$USER"') + "/ray_results/"
    folder, iteration = Ember_RL_func.find_agent_saved(path)
    # Remove previous agents saved into Agents_Saved
    proc = subprocess.Popen("rm -r " + save_agent_folder + "/*", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    sleep(0.5)
    # Save new Agent into Agents_Saved==
    proc = subprocess.Popen("cp -r /home/" + getpass.getuser() + "/ray_results/" + folder + " " + save_agent_folder, stdout=subprocess.PIPE, shell=True)
    sleep(0.5)

def cores_available(): # Find number of cores available in the running system
    print("Number of cores available: ", multiprocessing.cpu_count())
    print("Number of cores to use: ", multiprocessing.cpu_count() - 2)
    return int(multiprocessing.cpu_count()) - 2



if __name__ == "__main__":

    print("RL Agent")
    register_env("simplePible", lambda config: SimplePible(config))

    # Use the following settings
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    title = settings[0]["title"]
    train_test_real = settings[0]["train/test/real"]
    fold = settings[0]["agent_saved_folder"]
    num_cores = settings[0]["num_cores"]
    resume_path = ''
    if num_cores == "max":
        num_cores = cores_available()
    else:
        num_cores = int(num_cores)

    sc_volt_train = float(settings[0]["sc_volt_start_train"])
    sc_volt_test = float(settings[0]["sc_volt_start_test"])

    curr_path = os.getcwd()
    save_agent_folder = curr_path + fold

    ray.init()

    if  train_test_real == "train" or train_test_real == "test":
        start_train_date = datetime.datetime.strptime(settings[0]["start_train"], '%m/%d/%y %H:%M:%S')
        end_train_date = datetime.datetime.strptime(settings[0]["end_train"], '%m/%d/%y %H:%M:%S')

        start_test_date = datetime.datetime.strptime(settings[0]["start_test"], '%m/%d/%y %H:%M:%S')
        end_test_date = datetime.datetime.strptime(settings[0]["end_test"], '%m/%d/%y %H:%M:%S')
    elif train_test_real == "real":
        Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")
        now = datetime.datetime.now()
        start_train_date = now - datetime.timedelta(days=int(settings[0]["real_train_days"]))
        end_train_date = now
        start_test_date = now
        end_test_date = start_test_date + datetime.timedelta(days=1)

    while True:
        print("\nStart Training: ", start_train_date, end_train_date)
        if train_test_real == 'train' or train_test_real == 'real':
            training_PPO(start_train_date, end_train_date, resume_path)

        #start_test = start_train
        #end_test = end_train
        #start_test_date = start_test_date
        #end_test_date = end_test_date

        # Resume folder and best checkpoint from aget_saved_folder
        folder, iteration = Ember_RL_func.find_agent_saved(save_agent_folder)
        #iteration = 30

        print("\nStart Testing: ", start_test_date, end_test_date)
        resume_path = test_and_print_results(folder, iteration, start_test_date, end_test_date, title, curr_path, sc_volt_test, train_test_real)
        resume_path = resume_path[0]

        if train_test_real == 'real':
            Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")
            now = datetime.datetime.now()
            start_train_date = now - datetime.timedelta(days=int(settings[0]["real_train_days"]))
            end_train_date = now
            start_test_date = now
            end_test_date = start_test_date + datetime.timedelta(days=1)
        else:
            break

    print("Done")

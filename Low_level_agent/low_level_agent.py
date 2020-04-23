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


def test_and_print_results(folder, iteration, start_test, end_test, title, curr_path, sc_volt_test):
    path = glob.glob(save_agent_folder + folder  + '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)
    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["num_workers"] = 0
    config["explore"] = False
    config["env_config"] = {
       "settings": settings,
       "main_path": curr_path,
       "train/test": "test",
       "start_test": start_test,
       "end_test": end_test,
       "sc_volt_start_test": sc_volt_test,
    }
    start_date = datetime.datetime.strptime(start_test, '%m/%d/%y %H:%M:%S')
    end_date = datetime.datetime.strptime(end_test, '%m/%d/%y %H:%M:%S')

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

        obs, reward, done, info = env.step(learned_action)
        #print(learned_action)
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
    accuracy = float((int(info["PIR_events_detect"]) +int(info["thpl_events_detect"]))/(int(info["PIR_tot_events"]) +int(info["thpl_tot_events"])))
    accuracy = round(accuracy, 3)* 100
    print("Accuracy: ", accuracy)

    env.render(tot_rew, title, energy_used_tot, accuracy)
    return path

def training_PPO():
    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["lr"] = 1e-4
    config["num_workers"] = num_cores
    config["env_config"] = {
        "settings": settings,
        "main_path": curr_path,
        "start_train": start_train,
        "end_train": end_train,
        "train/test": "train",
        "sc_volt_start_train": sc_volt_train,
    }
    trainer = ppo.PPOTrainer(config=config, env="simplePible")
    max_min = -10000
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
    # Save new Agent into Agents_Saved
    proc = subprocess.Popen("cp -r /home/" + getpass.getuser() + "/ray_results/" + folder + " " + save_agent_folder, stdout=subprocess.PIPE, shell=True)
    sleep(0.5)

def cores_available(): # Find number of cores available in the running system
    print("Number of cores available: ", multiprocessing.cpu_count())
    print("Number of cores to use: ", multiprocessing.cpu_count() - 2)
    return int(multiprocessing.cpu_count()) - 2

if __name__ == "__main__":

    print("Low Level Agent")
    register_env("simplePible", lambda config: SimplePible(config))

    # Use the following settings
    with open('settings_low_level_agent.json', 'r') as f:
        settings = json.load(f)

    title = settings[0]["title"]
    train_test = settings[0]["train/test"]
    fold = settings[0]["agent_saved_folder"]
    num_cores = settings[0]["num_cores"]
    if num_cores == "max":
        num_cores = cores_available()
    else:
        num_cores = int(num_cores)

    sc_volt_train = float(settings[0]["sc_volt_start_train"])
    sc_volt_test = float(settings[0]["sc_volt_start_test"])

    curr_path = os.getcwd()
    save_agent_folder = curr_path + fold

    ray.init()

    start_train = settings[0]["start_train"]
    end_train = settings[0]["end_train"]
    start_test = settings[0]["start_test"]
    end_test = settings[0]["end_test"]

    start_train_date = datetime.datetime.strptime(start_train, '%m/%d/%y %H:%M:%S')
    end_train_date = datetime.datetime.strptime(end_train, '%m/%d/%y %H:%M:%S')
    start_test_date = datetime.datetime.strptime(start_test, '%m/%d/%y %H:%M:%S')
    end_test_date = datetime.datetime.strptime(end_test, '%m/%d/%y %H:%M:%S')

    while True:
        print("\nStart Training: ", start_train_date, end_train_date)
        if train_test != 'test':
            training_PPO()

        #start_test = start_train
        #end_test = end_train
        #start_test_date = start_test_date
        #end_test_date = end_test_date

        # Resume folder and best checkpoint from aget_saved_folder
        folder, iteration = Ember_RL_func.find_agent_saved(save_agent_folder)
        #iteration = 30

        print("\nStart Testing: ", start_test_date, end_test_date)
        resume_path = test_and_print_results(folder, iteration, start_test, end_test, title, curr_path, sc_volt_test)
        break

    print("Done")

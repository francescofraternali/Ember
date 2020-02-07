"""Example of a custom gym environment and model. Run this for a demo.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from ray.tune.logger import pretty_print

import ray
from ray import tune
from ray.tune import grid_search

from Pible_parameters import *
from Pible_class import SimplePible
import Pible_func
#import Gen_Light_Events
import datetime
from time import sleep
import os
import json
import subprocess
import glob
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

def test_and_print_results(folder, iteration, start_train, end_train, start_test, end_test, sc_volt_start_test):
    Agnt = "PPO"
    pattern = subprocess.getoutput('eval echo "~$USER"') + '/ray_results/' + folder + '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration)
    path = glob.glob(pattern, recursive=True)
    print(path)
    assert len(path) == 1, path
    agent = ppo.PPOAgent(config={
        "observation_filter": 'MeanStdFilter',
        "batch_mode": "complete_episodes",
        "env_config": {
            "train/test": "test",
            "start_test": start_test,
            "end_test": end_test,
            "sc_volt_start_test": sc_volt_start_test,
            "num_workers": 0,  # parallelism
         },
    }, env="simplePible")
    print(path[0])
    agent.restore(path[0])
    config = {
            "train/test": "test",
            "start_test": start_test,
            "end_test": end_test,
            "sc_volt_start_test": sc_volt_start_test,
            "num_workers": 0,  # parallelism
        }
    diff_days = datetime.datetime.strptime(end_test, '%m/%d/%y %H:%M:%S') - datetime.datetime.strptime(start_test, '%m/%d/%y %H:%M:%S')
    print("days", diff_days.days)
    #exit()
    Env = SimplePible(config)
    obs = Env.reset()
    pre_action = [0, 0]
    pre_reward = 0
    tot_rew = 0
    repeat = 1
    stop = 0
    death_days_tot = 0
    death_min_tot = 0
    energy_used_tot = 0; energy_prod_tot = 0
    while True:
        action_0_list = []
        action_1_list = []
        for i in range(0,50):
            learned_action = agent.compute_action(
                observation = obs,
                prev_action = pre_action,
                prev_reward = pre_reward,
                #full_fetch=True
            )
            #w = agent.get_weights()
            #print("weight", w)
            #learned_action[0][0] = 0
            #learned_action[1][0] = 1
            #print("action", learned_action)
            action_0_list.append(learned_action[0][0])
            action_1_list.append(learned_action[1][0])

        action_0_avg = sum(action_0_list)/ len(action_0_list)
        action_1_avg = sum(action_1_list)/ len(action_1_list)
        #print(int(round(action_0_avg)), action_1_avg)
        #obs, reward, done, none, energy_prod, energy_used = Env.step(learned_action)
        pre_action = [int(round(action_0_avg)), action_1_avg]
        obs, reward, done, res = Env.step(pre_action)
        energy_used_tot += float(res["Energy_used"])
        energy_prod_tot += float(res["Energy_prod"])
        tot_rew += reward
        pre_reward = reward
        #pre_action = [learned_action[0][0], learned_action[1][0]]
        if done:
            obs = Env.reset()
            print("done")
            #sleep(1)
            stop +=1
            death_days_tot += int(res["Death_days"])
            death_min_tot += int(res["Death_min"])
        if stop >= repeat*(diff_days.days/(1)):
            #print("observation:", obs, "action: ", learned_action, "rew: ", reward)
            break
    print("Energy Prod: ", energy_prod_tot/(repeat*diff_days.days), "Energy Used: ", energy_used_tot/(repeat*diff_days.days) )

    print("Tot reward: ", tot_rew)
    print_start = start_test
    print_end = end_test
    print("Tot events averaged per day: ", int(res["Tot_events"])/(repeat*diff_days.days))
    print("System Death Days Tot times: ", death_days_tot, ". Minutes: ", death_min_tot)
    try:
        percent = round((((res["events_detect"]) /  int(res["Tot_events"])) * 100), 2)
    except:
        percent = 0
    if (res["events_detect"]) == 0  and  int(res["Tot_events"]) == 0:
        percent = 100

    title_final = title + ('{0}%').format(percent)
    Env.render(1, tot_rew, print_start, print_end, title_final)
    volt_diff = float(sc_volt_start_test) - float(res["SC_volt"])
    sc_volt_start_test = float(res["SC_volt"])
    #print("volt", sc_volt_start_test)
    #sleep(5)
    Pible_func.write_results(tot_rew, start_train, end_train, start_test, end_test, percent, diff_days.days,
        energy_prod_tot/(repeat*diff_days.days), energy_used_tot/(repeat*diff_days.days), res["events_detect"],
        int(res["Tot_events"]), death_days_tot, death_min_tot, volt_diff)
    print("Percent detected: ", percent)

    return path, sc_volt_start_test

def PPO(config):

    max = -1000
    resume_path = config["env_config"]["resume"]

    trainer = ppo.PPOTrainer(config=config, env="simplePible")

    if resume_path != "":
        print("Restoring checkpoint: ", resume_path)
        sleep(5)
        trainer.restore(resume_path) # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(0, int(data[0]["training_iterations"])):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if int(result["training_iteration"]) % 10 == 0:
        #if result["episode_reward_mean"] > max:
            #max = result["episode_reward_mean"]
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

def cores_available(): # Find number of cores available in the running system
    proc = subprocess.Popen("nproc", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode()
    spl = out.strip().split('\n')
    print("Number of cores used: ", int(spl[0])-2)
    return int(spl[0])-2

if __name__ == "__main__":
    register_env("simplePible", lambda config: SimplePible(config))

    # Use the following settings
    with open('settings.json', 'r') as f:
        data = json.load(f)

    path_light = data[0]["path_light_data"]
    light_divider = float(data[0]["light_divider"])
    title = data[0]["title"]
    curr_path = os.getcwd()
    resume_path = ""

    # If want to restart from a previous train or from scratch
    if data[0]["start_with_restore_train"] == "yes":
        path = curr_path + "/Agents_Saved"
        folder, iteration = Pible_func.find_agent_saved(path)
        resume_path = path + "/" + folder + '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration)
        print(resume_path)
        #resume_path = curr_path + "/Agents_Saved/PPO_SimplePible_2020-01-10_07-59-45w9ov2_79/checkpoint_10/checkpoint-10"
        print("Restoring checkpoint: ", resume_path)
    else:
        print("Starting a new policy")

    path_light_data = curr_path + path_light;

    ray.init()
    sc_volt_start_test = data[0]["sc_volt_start_test"]
    # Can also register the env creator function explicitly with:
    #register_env("corridor", lambda config: gym_en_harv(config))

    cores = cores_available()

    start_train = data[0]["start_train"]
    time_train = data[0]["time_train"]
    end_train = data[0]["end_train"]
    time_test = data[0]["time_test"]

    start_train_date = datetime.datetime.strptime(start_train, '%m/%d/%y %H:%M:%S')
    end_train_date = datetime.datetime.strptime(end_train, '%m/%d/%y %H:%M:%S')

    end_train_date_temp = start_train_date + datetime.timedelta(hours=time_train)
    end_train_temp = end_train_date_temp.strftime("%m/%d/%y %H:%M:%S")

    while True:
        # Start training
        print("\nTrain ", start_train, end_train_temp)
        sleep(1)

        config={
            "observation_filter": 'MeanStdFilter',
            "batch_mode": "complete_episodes",
            "lr" : grid_search([1e-4]),
            "num_workers": cores,  # parallelism
            "env_config": {
                "train/test": "train",
                "start_train": start_train,
                "end_train": end_train,
                "sc_volt_start_train": 'rand',
                "resume": resume_path,
             },
        }
        tune.run(PPO, config=config)

        # Set Days for Testing
        start_test = end_train_temp

        #start_test = start_train
        start_test_date = datetime.datetime.strptime(start_test, '%m/%d/%y %H:%M:%S')
        end_test_date = start_test_date + datetime.timedelta(hours=time_test)
        end_test = end_test_date.strftime("%m/%d/%y %H:%M:%S")

        # Resume folder and best checkpoint
        path = subprocess.getoutput('eval echo "~$USER"') + "/ray_results/"
        folder, iteration = Pible_func.find_agent_saved(path)
        agent_save = curr_path + "/Agents_Saved"

        # clean folder and save agent
        proc = subprocess.Popen("rm -r " + agent_save + "/*", stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        sleep(1)
        # proc = subprocess.Popen(path + folder + " " + agent_save, stdout=subprocess.PIPE, shell=True)

        # Start Testing
        print("\nTest: ", start_test, end_test)
        sleep(10)
        resume_path, sc_volt_start_test = test_and_print_results(folder, iteration, start_train, end_train_temp, start_test, end_test, sc_volt_start_test)
        resume_path = resume_path[0]

        # Set days for new training
        end_train_temp = end_test
        diff_days_train = end_test_date - start_train_date
        print("diff days train ", diff_days_train.days)
        if (diff_days_train.days*24) > int(data[0]["size_train_max"]):
            start_train_date = end_test_date - datetime.timedelta(hours=int(data[0]["size_train_max"]))
            start_train = start_train_date.strftime('%m/%d/%y %H:%M:%S')

        if end_test_date >= end_train_date:
            break

    print("Done")

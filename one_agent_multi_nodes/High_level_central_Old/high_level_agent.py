"""Example of a custom gym environment and model. Run this for a demo.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import numpy as np
import gym
from ray.tune.logger import pretty_print

import ray
from ray import tune
from ray.tune import grid_search

from Pible_parameters import *
from Pible_class_high_level_agent import SimplePible
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

def test_and_print_results(folder, iteration, start_train, end_train, start_test, end_test, sc_volt_start_test, title):
    path = glob.glob(save_agent_folder + folder  + '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)
    assert len(path) == 1, path
    config = ppo.DEFAULT_CONFIG.copy()
    #config["observation_filter"] = 'MeanStdFilter'
    #config["batch_mode"] = "complete_episodes"
    config["num_workers"] = 0
    config["explore"] = False
    config["env_config"] = {
        "settings": settings,
        "main_path": curr_path,
        "train/test": "test",
        "start_test": start_test,
        "end_test": end_test,
        "sc_volt_start_test": sc_volt_start_test,
    }
    agent = ppo.PPOTrainer(config=config, env="simplePible")
    print("path", path[0])
    agent.restore(path[0])
    env = SimplePible(config["env_config"])
    #env = gym.make("CartPole-v0")
    obs = env.reset()
    print(obs)
    diff_days = datetime.datetime.strptime(end_test, '%m/%d/%y %H:%M:%S') - datetime.datetime.strptime(start_test, '%m/%d/%y %H:%M:%S')
    print("days", diff_days.days)
    #exit()
    prev_action = [0]
    prev_reward = 0
    tot_rew = 0
    repeat = 1
    stop = 0
    info = {}
    death_days_tot = 0; death_min_tot = 0
    energy_used_tot = 0; energy_prod_tot = 0
    while True:
        learn_action = agent.compute_action(
            observation = obs,
            #state=state,
            #prev_action = prev_action,
            #prev_reward = prev_reward,
            #info=info
            #full_fetch=True
        )
        obs, reward, done, info = env.step(learn_action)
        #print(obs)
        #energy_used_tot += float(info["Energy_used"])
        #energy_prod_tot += float(info["Energy_prod"])
        tot_rew += reward
        prev_reward = reward
        #prev_action = learn_action[0]
        #prev_action = [lear_action[0][0], learned_action[1][0]]
        if done:
            obs = env.reset()
            print("done")
            stop +=1
            #death_days_tot += int(info["Death_days"])
            #death_min_tot += int(info["Death_min"])
        if stop >= repeat*(diff_days.days/(episode_lenght)):
            #print("observation:", obs, "action: ", learned_action, "rew: ", reward)
            break

    print("Energy Prod: ", energy_prod_tot/(repeat*diff_days.days), "Energy Used: ", energy_used_tot/(repeat*diff_days.days) )

    print("Tot reward: ", tot_rew)
    print("Tot events averaged per day: ", int(info["Tot_events"])/(repeat*diff_days.days))
    print("System Death Days Tot times: ", death_days_tot, ". Minutes: ", death_min_tot)
    try:
        percent = round((((info["events_detect"]) /  int(info["Tot_events"])) * 100), 2)
    except:
        percent = 0
    if (info["events_detect"]) == 0  and  int(info["Tot_events"]) == 0:
        percent = 100

    title_final = title + ('{0}%').format(percent)
    env.render(1, tot_rew, start_test, end_test, title_final)
    volt_diff = float(sc_volt_start_test) - float(info["SC_volt"])
    sc_volt_start_test = float(info["SC_volt"])
    #print("volt", sc_volt_start_test)
    #sleep(5)
    Pible_func.write_results(tot_rew, start_train, end_train, start_test, end_test, percent, diff_days.days,
        energy_prod_tot/(repeat*diff_days.days), energy_used_tot/(repeat*diff_days.days), info["events_detect"],
        int(info["Tot_events"]), death_days_tot, death_min_tot, volt_diff)
    print("Percent detected: ", percent)

    return path, sc_volt_start_test

def training_PPO(resume_path, start_train, end_train, cores):

    max = -1000
    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["lr"] = 1e-4
    config["num_workers"] = cores
    config["env_config"] = {
        "settings": settings,
        "main_path": curr_path,
        "train/test": "train",
        "start_train": start_train,
        "end_train": end_train,
        "sc_volt_start_train": 3.2,
        "resume": resume_path,
    }
    resume_path = config["env_config"]["resume"]
    trainer = ppo.PPOTrainer(config=config, env="simplePible")

    if resume_path != "":
        print("Restoring checkpoint: ", resume_path)
        sleep(1)
        trainer.restore(resume_path) # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(0, int(settings[0]["training_iterations"])):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if int(result["training_iteration"]) % 10 == 0:
        #if result["episode_reward_mean"] > max:
            #max = result["episode_reward_mean"]
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    # Find best checkpoint
    path = subprocess.getoutput('eval echo "~$USER"') + "/ray_results/"
    folder, iteration = Pible_func.find_agent_saved(path)
    # Remove previous agents saved into Agents_Saved
    proc = subprocess.Popen("rm -r " + save_agent_folder + "/*", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    sleep(0.5)
    # Save new Agent into Agents_Saved
    proc = subprocess.Popen("cp -r /home/francesco/ray_results/" + folder + " " + save_agent_folder, stdout=subprocess.PIPE, shell=True)
    sleep(0.5)

def cores_available(): # Find number of cores available in the running system
    proc = subprocess.Popen("nproc", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode()
    spl = out.strip().split('\n')
    print("Number of cores used: ", spl)
    return int(spl[0])-2


if __name__ == "__main__":

    print("High Level Agent")
    register_env("simplePible", lambda config: SimplePible(config))

    # Use the following settings
    with open('settings_high_level_agent.json', 'r') as f:
        settings = json.load(f)

    title = settings[0]["title"]
    curr_path = os.getcwd()
    save_agent_folder = curr_path + "/Agents_Saved/"
    resume_path = ""

    # If want to restart from a previous train or from scratch
    if settings[0]["start_with_restore_train"] == "yes":
        folder, iteration = Pible_func.find_agent_saved(save_agent_folder)
        resume_path = curr_path + "/" + folder + '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration)
        print("Restoring checkpoint: ", resume_path)
    else:
        print("Starting a new policy")

    ray.init()
    sc_volt_start_test = settings[0]["sc_volt_start_test"]
    # Can also register the env creator function explicitly with:
    #register_env("corridor", lambda config: gym_en_harv(config))

    cores = cores_available()
    cores = 6

    start_train = settings[0]["start_train"]
    end_train = settings[0]["end_train"]
    #time_train = settings[0]["time_train"]
    #time_test = settings[0]["time_test"]

    start_train_date = datetime.datetime.strptime(start_train, '%m/%d/%y %H:%M:%S')
    end_train_date = datetime.datetime.strptime(end_train, '%m/%d/%y %H:%M:%S')

    #end_train_date_temp = start_train_date + datetime.timedelta(hours=time_train)
    #end_train_temp = end_train_date_temp.strftime("%m/%d/%y %H:%M:%S")

    while True:
        print("\nStart Training: ", start_train, end_train)
        training_PPO(resume_path, start_train, end_train, cores)

        # Set Days for Testing
        #start_test = end_train_temp # use this to run test after train
        start_test = start_train # use this to test in the same training data
        end_test = end_train

        start_test_date = datetime.datetime.strptime(start_test, '%m/%d/%y %H:%M:%S')
        end_test_date = datetime.datetime.strptime(end_test, '%m/%d/%y %H:%M:%S')
        #end_test_date = start_test_date + datetime.timedelta(hours=time_test)

        print("\nStart Testing: ", start_test, end_test)
        # Resume folder and best checkpoint from aget_saved_folder
        folder, iteration = Pible_func.find_agent_saved(save_agent_folder)
        #iteration = 50

        resume_path, sc_volt_start_test = test_and_print_results(folder, iteration, start_train, end_train, start_test, end_test, sc_volt_start_test, title)
        resume_path = resume_path[0]
        break
        # Set days for new training
        end_train_temp = end_test
        diff_days_train = end_test_date - start_train_date
        print("diff days train ", diff_days_train.days)
        if (diff_days_train.days*24) > int(settings[0]["size_train_max"]):
            start_train_date = end_test_date - datetime.timedelta(hours=int(settings[0]["size_train_max"]))
            start_train = start_train_date.strftime('%m/%d/%y %H:%M:%S')

        if end_test_date >= end_train_date:
            break

    print("Done")
    exit()

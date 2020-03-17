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
from Pible_parameters import *
from Pible_class_high_level_agent import SimplePible
import Pible_func
import datetime
from time import sleep
import os
import subprocess
import glob
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

def test_and_print_results(folder, iteration, start_test, end_test, title, curr_path):
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
       "sc_volt_start_test": 3.2,
    }
    agent = ppo.PPOTrainer(config=config, env="simplePible")
    agent.restore(path[0])
    env = SimplePible(config["env_config"])
    obs = env.reset()
    tot_rew = 0
    info = {}
    while True:
        learn_action = agent.compute_action(
            observation = obs,
        )
        obs, reward, done, info = env.step(learn_action)
        #print(obs)
        tot_rew += reward

        if done:
            obs = env.reset()
            print("done")
            break
    print("tot reward", round(tot_rew, 3))
    env.render(tot_rew, "title")
    return path

def training_PPO():
    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["lr"] = 1e-4
    config["num_workers"] = 6
    config["env_config"] = {
        "settings": settings,
        "main_path": curr_path,
        "start_train": start_train,
        "end_train": end_train,
        "train/test": "train",
        "sc_volt_start_train": 3.2,
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
    folder, iteration = Pible_func.find_agent_saved(path)
    # Remove previous agents saved into Agents_Saved
    proc = subprocess.Popen("rm -r " + save_agent_folder + "/*", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    sleep(0.5)
    # Save new Agent into Agents_Saved
    proc = subprocess.Popen("cp -r /home/francesco/ray_results/" + folder + " " + save_agent_folder, stdout=subprocess.PIPE, shell=True)
    sleep(0.5)


if __name__ == "__main__":

    print("Disaster Test")
    register_env("simplePible", lambda config: SimplePible(config))

    # Use the following settings
    with open('settings_high_level_agent.json', 'r') as f:
        settings = json.load(f)

    title = settings[0]["title"]
    curr_path = os.getcwd()
    save_agent_folder = curr_path + "/Agents_Saved/"

    ray.init()

    start_train = settings[0]["start_train"]
    end_train = settings[0]["end_train"]

    start_train_date = datetime.datetime.strptime(start_train, '%m/%d/%y %H:%M:%S')
    end_train_date = datetime.datetime.strptime(end_train, '%m/%d/%y %H:%M:%S')

    while True:
        print("\nStart Training: ", start_train_date, end_train_date)
        training_PPO()

        start_test = start_train
        end_test = end_train
        start_test_date = start_train_date
        end_test_date = end_train_date

        # Resume folder and best checkpoint from aget_saved_folder
        folder, iteration = Pible_func.find_agent_saved(save_agent_folder)
        #iteration = 30

        print("\nStart Testing: ", start_test_date, end_test_date)
        resume_path = test_and_print_results(folder, iteration, start_test, end_test, title, curr_path)
        break

    print("Done")

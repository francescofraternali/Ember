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
        "main_path": curr_path,
        "train/test": "test",
        "start_test": start_test,
        "end_test": end_test,
        "sc_volt_start_test": 3.2,
    }
    agent = ppo.PPOTrainer(config=config, env="CartPole-v0")
    print("path", path[0])
    agent.restore(path[0])
    #env = CartPole-v0(config["env_config"])
    env = gym.make("CartPole-v0")
    obs = env.reset()
    prev_action = [0]
    prev_reward = 0
    tot_rew = 0
    info = {}
    while True:
        learn_action = agent.compute_action(
            observation = obs,
        )
        obs, reward, done, info = env.step(learn_action)
        tot_rew += reward

        if done:
            obs = env.reset()
            print("done")
            break
    print("tot reward", round(tot_rew, 3))
    return path

def training_PPO():

    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["lr"] = 1e-4
    config["num_workers"] = 6
    config["env_config"] = {
        "main_path": curr_path,
        "start_train": start_train,
        "end_train": end_train,
        "train/test": "train",
        "sc_volt_start_train": 3.2,
    }
    trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
    max_min = -10000
    for i in range(0, 10):
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

    print("Test")
    register_env("simplePible", lambda config: SimplePible(config))

    title = "title"
    curr_path = os.getcwd()
    save_agent_folder = curr_path + "/Agents_Saved/"

    ray.init()

    start_train = "10/7/19 00:00:00"
    end_train = "10/14/19 00:00:00"

    start_train_date = datetime.datetime.strptime(start_train, '%m/%d/%y %H:%M:%S')
    end_train_date = datetime.datetime.strptime(end_train, '%m/%d/%y %H:%M:%S')

    while True:
        print("\n Start Training: ", start_train_date, end_train_date)
        training_PPO()

        start_test = start_train
        end_test = end_train
        start_test_date = start_train_date
        end_test_date = end_train_date

        # Resume folder and best checkpoint from aget_saved_folder
        folder, iteration = Pible_func.find_agent_saved(save_agent_folder)
        #iteration = 100

        print("\n Start Testing: ", start_test_date, end_test_date)
        resume_path = test_and_print_results(folder, iteration, start_test, end_test, title, curr_path)
        break

    print("Done")

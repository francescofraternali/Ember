""" Hierarchical RL for Pible
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from time import sleep
import numpy as np
import gym
from ray.tune.logger import pretty_print
import ray
from ray import tune
from ray.tune import grid_search
import json
from Pible_param_func import *
from Pible_class import SimplePible
import Ember_RL_func
import datetime
import os
import glob
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
import getpass
import sys


def test_and_print_results(agent_folder, iteration, start_date, end_date, title, curr_path, sc_volt_test, train_test_real, diff_days, GT_hour):
    train_test_real_orig = train_test_real
    train_test_real = 'test' if train_test_real == 'train' else train_test_real
    path = glob.glob(agent_folder + '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)

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
       "diff_days": diff_days,
       "GT_hour_start": GT_hour,
       "resume_from_iter": iteration,
    }
    if train_test_real == "real":
        #Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")
        fold = os.path.basename(os.getcwd())
        ID_temp = fold.split('_')[-1]
        action_file = ID_temp + "_action.json"

    agent = ppo.PPOTrainer(config=config, env="simplePible")
    agent.restore(path[0])
    env = SimplePible(config["env_config"])
    obs = env.reset()
    tot_rew = 0;  energy_used_tot = 0;  energy_prod_tot = 0
    print("initial observations: ", obs)
    while True:

        learned_action = agent.compute_action(
                observation = obs,
        )

        if train_test_real == "real":
            learned_action = Ember_RL_func.correct_action(obs, learned_action)
            Ember_RL_func.sync_action(action_file, learned_action, settings[0]["PIR_or_thpl"])
            Ember_RL_func.sync_ID_file_to_BS(settings[0]["pwd"], settings[0]["bs_name"], action_file, "/home/pi/Base_Station_20/ID/")
            print("action_taken: ", learned_action)
            if isinstance(learned_action, list):
                if len(learned_action) == 1:
                    print("sleeping 60 mins fixed")
                    sleep(60 * 60)
                elif len(learned_action) == 2:
                    print("len action is 2. sleeping 60 mins fixed")
                    sleep(60 * 60)
                else:
                    print("there is a problem. Check check")
                    exit()
            else:
                print("sleeping fixed " + str(60) + " mins. Else case")
                sleep(60 * 60)
            Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")

        obs, reward, done, info = env.step(learned_action)
        #print(obs)
        #print(learned_action, reward, info["thpl_tot_events"])

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
    accuracy = Ember_RL_func.calc_accuracy(info)
    print("Accuracy: ", accuracy)

    #if train_test_real_orig == "test" or train_test_real_orig == "train":
    env.render(tot_rew, title, energy_used_tot, accuracy)

    return path, info["SC_volt"], int(info["GT_hours_start"])

def training_PPO(start_train_date, end_train_date, resume, diff_days):
    config = ppo.DEFAULT_CONFIG.copy()
    config["observation_filter"] = 'MeanStdFilter'
    config["batch_mode"] = "complete_episodes"
    config["lr"] = 1e-4
    config["num_workers"] = num_cores
    #config["num_sgd_iter"] = 5 # default 30
    config["env_config"] = {
        "settings": settings,
        "main_path": curr_path,
        "start_train": start_train_date,
        "end_train": end_train_date,
        "train/test": "train",
        "sc_volt_start_train": sc_volt_train,
        "diff_days": diff_days,
        "GT_hour_start": 0,
    }
    trainer = ppo.PPOTrainer(config=config, env="simplePible")

    if resume_path != "":
        print("Restoring checkpoint: ", resume)
        sleep(5)
        trainer.restore(resume) # Can optionally call trainer.restore(path) to load a checkpoint.

    global prev_res
    prev_res = []

    for i in range(0, int(settings[0]["training_iterations"])):
        result = trainer.train()
        print(pretty_print(result))

        if int(result["training_iteration"]) % 10 == 0:
        #if max_min > int(result["episode_reward_mean"])
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
            checkp_split = checkpoint.split('/')
            parent_dir = '/'.join(checkp_split[0:-2])

            curr_res = float(result["episode_reward_mean"])
            #if (int(result["training_iteration"]) > 10) and prev_res != []:

            if len(prev_res) >= 5 and curr_res != 0.0:
                avg_res = sum(prev_res)/len(prev_res)
                print(curr_res, avg_res)
                diff_perc = (((curr_res - avg_res)/curr_res) * 100)
                print("\nDiff Percentage: ", diff_perc)
                if diff_perc < 3 and diff_perc > -3:
                        print("Converged!")
                        sleep(2)
                        break

            if len(prev_res) >= 5:
                prev_res = np.roll(prev_res, 1)
                prev_res[0] = curr_res
            else:
                prev_res.append(curr_res)

            #print(prev_res)
            #sleep(4)
   # Remove previous agents and save bew agetn into Agents_Saved
    #print("out", parent_dir, save_agent_folder)
    Ember_RL_func.rm_old_save_new_agent(parent_dir, save_agent_folder)



if __name__ == "__main__":


    print("Starting RL Agent")

    register_env("simplePible", lambda config: SimplePible(config))

    #print("curr path: " , sys.argv[1])
    #curr_path = sys.argv[1]
    curr_path = os.getcwd()

    # Use the following settings
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    if "update_folder" in settings[0]:
        update_folder_path = settings[0]["update_folder"]
        Ember_RL_func.update_code(update_folder_path)
    else:
        print("Not updating code")
        sleep(2)

    title = settings[0]["title"]
    train_test_real = settings[0]["train/test/real"]
    fold = settings[0]["agent_saved_folder"]
    num_cores = settings[0]["num_cores"]

    if settings[0]["train_gt/rm_miss/inject"] == '1' or settings[0]["train_gt/rm_miss/inject"] == '2':
        Ember_RL_func.restore_orig_data(settings[0]["file_light"])

    resume_path = ''
    prev_res = []
    if num_cores == "max":
        num_cores = Ember_RL_func.cores_available()
    else:
        num_cores = int(num_cores)

    sc_volt_train = float(settings[0]["sc_volt_start_train"])
    sc_volt_test = float(settings[0]["sc_volt_start_test"])
    train_days = int(settings[0]["real_train_days"])
    save_agent_folder = curr_path + "/" + fold
    GT_hour_start = 6

    ray.init()

    if  train_test_real == "train" or train_test_real == "test":
        start_train_date = datetime.datetime.strptime(settings[0]["start_train"], '%m/%d/%y %H:%M:%S')
        end_train_date = datetime.datetime.strptime(settings[0]["end_train"], '%m/%d/%y %H:%M:%S')

        start_test_date = datetime.datetime.strptime(settings[0]["start_test"], '%m/%d/%y %H:%M:%S')
        end_test_date = datetime.datetime.strptime(settings[0]["end_test"], '%m/%d/%y %H:%M:%S')
    elif train_test_real == "real":
        #Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")
        now = datetime.datetime.now()
        start_train_date = now - datetime.timedelta(days=train_days)
        end_train_date = now

    while True:
        diff_days = (end_train_date - start_train_date).days
        print("Diff train days: ", diff_days)
        if diff_days == 7:
            resume_path = '' # remove previous learning and redo using a week

        if train_test_real == 'train' or train_test_real == 'real':
            print("\nStart Training: ", start_train_date, end_train_date)
            training_PPO(start_train_date, end_train_date, resume_path, diff_days)

        #start_test = start_train
        #end_test = end_train
        #start_test_date = start_test_date
        #end_test_date = end_test_date

        # Find best checkpoint
        agent_fold = save_agent_folder + '/' + os.listdir(save_agent_folder)[0]
        iteration = Ember_RL_func.find_best_checkpoint(agent_fold)
        #folder, iteration = Ember_RL_func.find_agent_saved(save_agent_folder)
        #iteration = 30
        if train_test_real == "real":
            start_test_date = datetime.datetime.now()
            end_test_date = start_test_date + datetime.timedelta(days=1)

        print("\nStart Testing: ", start_test_date, end_test_date)
        resume_path, sc_volt_test, GT_hour_start = test_and_print_results(agent_fold, iteration, start_test_date, end_test_date, title, curr_path, sc_volt_test, train_test_real, diff_days, GT_hour_start)
        resume_path = resume_path[0]

        if train_test_real == 'real':
            Ember_RL_func.sync_input_data(settings[0]["pwd"], settings[0]["bs_name"], settings[0]["file_light"], "")
            now = datetime.datetime.now()
            train_days += 1
            if train_days > int(settings[0]["real_train_max"]):
                train_days = int(settings[0]["real_train_max"])

            start_train_date = now - datetime.timedelta(days=train_days)
            end_train_date = now
            start_test_date = now
            end_test_date = start_test_date + datetime.timedelta(days=1)
        else:
            #start_train_date = start_train_date + datetime.timedelta(days=episode_lenght)
            end_train_date = end_train_date + datetime.timedelta(days=episode_lenght)
            start_test_date = start_test_date + datetime.timedelta(days=episode_lenght)
            end_test_date = end_test_date + datetime.timedelta(days=episode_lenght)
            #break
        break

    print("Done Done")

import glob
import subprocess
import ray
from ray.rllib.agents import ddpg
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents import dqn
#from gym_envs.envs import Pible_env
from time import sleep
import os
import datetime
import numpy as np
import json

with open('settings.json', 'r') as f:
    data = json.load(f)

#path_data = subprocess.getoutput('eval echo "~$USER"') + "/Desktop/Ray-RLlib-Pible/gym_envs/envs"
path_data = os.getcwd()
print(path_data)

# Read data
#observations, actions, consumptions, time_range, scalers = read_data(start_time, end_time, '2146', normalize=True)

# Init exp
from training_pible import SimplePible
#from gym_envs.envs.Pible_env import pible_env_creator
#register_env('Pible-v2', pible_env_creator)
Agnt = 'PPO'
title = data[0]["title"]
#assert len(observations.shape) == 2

ray.init()

folder = data[0]["agent_saved_folder"]
print("\nFound folder: ", folder)

max_mean = - 10000
count = 0
for line in open(path_data + "/" + folder + "/result.json", 'r'):
    count += 1
    dict = json.loads(line)
    if dict['episode_reward_mean'] >= max_mean:
        max_mean = dict['episode_reward_mean']
        iteration = count
    #data = json.loads(text)

    #for p in data["episode_reward_mean"]:
    #    print(p)
iter_str = str(iteration)
iteration = (int(iter_str[:-1])* 10) + 10
print("Best checkpoint found:", iteration, ". Mean Reward Episode: ", max_mean)

sleep(1)
if True:
    #path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/' + Agnt +'/' + folder  +
    path = glob.glob(folder +
    #path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/DDPG/DDPG_VAV-v0_0_2019-05-10_20-02-38zocesjrb' +
                     '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)
    assert len(path) == 1, path
    #start = "11/24/19 00:00:00"
    #end = "12/02/19 00:00:00"
    #start = "06/23/19 00:00:00"
    #end = "06/30/19 00:00:00"
    agent = ppo.PPOAgent(config={
    #agent = dqn.DQNAgent(config={
    #agent = ddpg.DDPGAgent(config={
        #"vf_share_layers": True,
        "observation_filter": 'MeanStdFilter',
        "batch_mode": "complete_episodes",
        #"path": path_data,
        "env_config": {
            "train/test": "test",
            #"start": "11/24/19 00:00:00",
            #"end": "12/02/19 00:00:00",
            "start_train": data[0]["start_train"],
            "end_train": data[0]["end_train"],
            "start_test": data[0]["start_test"],
            "end_test": data[0]["end_test"],
            "sc_volt_start_train": 'rand',
            "sc_volt_start_test": '3.5',
         },
    #}, env='Pible-v2')
    }, env=SimplePible)
    print(path[0])
    agent.restore(path[0])
    config = {
        "train/test": data[0]["train/test"],
        #"start": "11/24/19 00:00:00",
        #"end": "12/02/19 00:00:00",
        "start_train": data[0]["start_train"],
        "end_train": data[0]["end_train"],
        "start_test": data[0]["start_test"],
        "end_test": data[0]["end_test"],
        "sc_volt_start_train": 'rand',
        "sc_volt_start_test": '3.5',
        }
    diff_days = datetime.datetime.strptime(data[0]["end_test"], '%m/%d/%y %H:%M:%S') - datetime.datetime.strptime(data[0]["start_test"], '%m/%d/%y %H:%M:%S')
    print("days", diff_days.days)
    #exit()
    Env = SimplePible(config)
    obs = Env.reset()
    pre_action = [0, 0]
    pre_reward = 0
    tot_rew = 0
    stop = 0
    repeat = data[0]["repeat"]
    energy_used_tot = 0; energy_prod_tot = 0
    while True:
        action_0_list = []
        action_1_list = []
        for i in range(0,200):
            learned_action = agent.compute_action(
                observation = obs,
                prev_action = pre_action,
                prev_reward = pre_reward,
                #full_fetch=True
            )
            #w = agent.get_weights()
            #print("weight", w)
            #learned_action[0][0] = 1
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
            stop +=1
        if stop >= repeat*(diff_days.days):
            print("observation:", obs, "action: ", learned_action, "rew: ", reward)
            break
    print("Energy Prod: ", energy_prod_tot/(repeat*diff_days.days), "Energy Used: ", energy_used_tot/(repeat*diff_days.days) )

print("Tot reward: ", tot_rew)
print_start = ""
print_start = data[0]["start_print"]
print_end = data[0]["end_print"]
print("Tot events averaged per day: ", int(res["Tot_events"])/(repeat*diff_days.days))
percent = round((((res["events_detect"]) / int(res["Tot_events"])) * 100), 2)
title_final = title + ('{0}%').format(percent)
Env.render(1, tot_rew, print_start, print_end, title_final)

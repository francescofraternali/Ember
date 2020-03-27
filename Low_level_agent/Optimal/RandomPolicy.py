# A random policy that samples action randomly from the action space to interact with the environment.
# author: yubaidi

import json
import numpy as np
import os
from PibleEnv import SimplePible
from Pible_parameters import episode_length

class RandomPolicy():
    def __init__(self):
        super()

    def get_action(self, obs):
        return np.random.randint(0, 2) # generate a random action from {0, 1}

class OptimalPolicy():
    def __init__(self):
        super()

    def get_action(self, obs):
        """
        if self.time.minute == 0 and (self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17):
            self.PIR_on_off = 1
            self.next_wake_up_time = 1
        elif self.time.minute == 1 and (self.time.hour == 8 or self.time.hour == 10 or self.time.hour == 17):
            self.PIR_on_off = 0
            self.next_wake_up_time = 59
        else:
            self.PIR_on_off = 0
            self.next_wake_up_time = 60
        # ending optimal solution
        """
        pass

def initialize_env(settings, current_path, start, end, stage):
    env_config = {
        "settings": settings,
        "main_path": current_path,
        "train/test": stage,
        "start_test": start,
        "end_test": end,
        "sc_volt_start_test": 3.2,
    }
    env = SimplePible(env_config)
    return env

if __name__ == "__main__":
    print("Random Low-level agent Policy")
    with open('settings_low_level_agent.json', 'r') as f:
        settings = json.load(f)

    start_test = settings[0]["start_train"]
    end_test = settings[0]["end_train"]
    stage = "test"

    current_path = os.getcwd()
    env = initialize_env(settings, current_path, start_test, end_test, stage)
    obs = env.reset() # TODO: check if this complies with gym API
    done = False

    policy = RandomPolicy()
    total_reward, energy_used_tot, energy_prod_tot = 0, 0, 0
    while not done:
        action = policy.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        # print("The state action pair is --> State: %s; Action: %s; Next State: %s" % (obs, action, next_obs))
        obs = next_obs
        energy_used_tot += float(info["energy_used"])
        energy_prod_tot += float(info["energy_prod"])
        total_reward += reward

    print("Done!")
    print("Total reward", round(total_reward, 3))
    print("Energy Prod per day: ", energy_prod_tot / episode_length, "Energy Used: ", energy_used_tot / episode_length)
    print("Tot events averaged per day: ", int(info["tot_events"]) / episode_length)
    env.render(total_reward, "title")


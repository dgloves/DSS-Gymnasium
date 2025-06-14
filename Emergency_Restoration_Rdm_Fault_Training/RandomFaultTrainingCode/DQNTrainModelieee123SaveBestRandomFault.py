# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:48:24 2021 Reference
https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/4_callbacks_hyperparameter_tuning.ipynb
@author: Hongda
"""

import gym
# from IEEE123envV2 import rlEnv 
# from IEEE123envOpendssDirectMultiSWRandomOpen import rlEnv

# from IEEE123envODMultSWRndmOpen import rlEnv
# from IEEE123nodeFixFaultWithSWpwrs import rlEnv
from IEEE123nodeRandomFaultSWpwrsENV0912 import rlEnv

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


import os
import numpy as np
# from TrainModelieee123SaveEveryTimeStep import MyMonitorWrapper #Record every step of training process
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from typing import Callable
#matplotlib notebook

# from stable_baselines3.common.env_checker import check_env
# env = rlEnv()
# # If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)


from stable_baselines3.common.callbacks import BaseCallback
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True




def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
# By default, `reset_num_timesteps` is True, in which case the learning rate schedule resets.
# progress_remaining = 1.0 - (num_timesteps / total_timesteps)
# Initial learning rate of 0.001

# Create log dir
# log_dir = r"/home/hongda.ren/IEEE123/FixTraining/LR0.001NN64" #Change for cases
log_dir = r"/home/hongda.ren/IEEE123/RadomFaultTraining/LR0.00NN6464record" #Change to your local folder path
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
# env = make_vec_env(rlEnv, n_envs=1, monitor_dir=log_dir)
# it is equivalent to:
# env = gym.make(rlEnv)

SwitchOpenNoList=[[12,13],[11],[14,3],[14,15],[22,23],[22,5],[18,5],[4,17,18],[17,19],[19,20],[20,21],[2,16]] #total 12 cases
actionHuman=[[9],[7],[7],[9],[7],[7],[7],[7,10],[10],[10],[10],[7]]
# rewardHuman=[3090.98, 3168.24,3070.79,3053.43,3086.45, 3066.03,3046.85, 3036.55, 3075.18, 3093.83, 2918.83]

env = rlEnv(SwitchOpenNoList)
env = Monitor(env, log_dir)
os.makedirs(log_dir, exist_ok=True)
# env = MyMonitorWrapper(env)
# env = DummyVecEnv([lambda: env])


# Create Callback
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)
# callbackEveryStep = 
# Create environment
# env = rlEnv


MlpPolicy.net_arch = [64,64] #action duplicate and no optimal for all



# Instantiate the agent #linear_schedule(0.0002)
model = DQN(MlpPolicy, env, learning_rate=0.0001, buffer_size=20000, learning_starts=1, gamma=1.0, target_update_interval=1000,exploration_final_eps=0.05,verbose=1)
# Train the agent
model.learn(total_timesteps=30000, callback=callback, log_interval=100)


# # Instantiate the agent Best settings for random fault case
# model = DQN(MlpPolicy, env, learning_rate=0.002, buffer_size=20000, learning_starts=1, gamma=1.0, target_update_interval=1200,exploration_final_eps=0.05,verbose=1)
# # Train the agent
# model.learn(total_timesteps=30000, callback=callback, log_interval=500)
# # Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# model = DQN.load("dqn_lunar")

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



# Plot results

from stable_baselines3.common import results_plotter

# Helper from the library
results_plotter.plot_results([log_dir], 3e4, results_plotter.X_TIMESTEPS, "DQN rewards")


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results(log_dir)                               

# Load the trained agent

model = DQN.load(os.path.join(log_dir, 'best_model.zip')) 

# # Enjoy trained agent
obs = env.reset()
for i in range(5):
    print("Step {}".format(i ))
    action, _states = model.predict(obs, deterministic=True) #return action and next state
    obs, rewards, dones, info = env.step(action)
    np.set_printoptions(precision=3)
    print('action=',action,'obs= ', obs, 'reward=', rewards, 'done=', dones, '\n', info)
    # env.render()

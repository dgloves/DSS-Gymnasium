"""
This file calls the Stable Baselines3 library to train and evaluate a DRL agent using your customized
DSS-Gymnasium Environment
"""

from build_environment import myAgent
from stable_baselines3 import A2C  # select algorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
import os

# run a test on your environment to validate Gymnasium protocol
gym_env = myAgent()
check_env(gym_env, warn=True)  # print warnings

# set params for training
# set your local path for logging training data, saving model
log_path = os.getcwd()
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])  # save progress metrics
gamma = 0.99  # discount param
learning_rate = 0.0001  # nn lrate
total_steps = 2400  # 100 episodes of 24 hour simulation

model = A2C("MlpPolicy", env=myAgent, gamma=gamma, learning_rate=learning_rate,
            tensorboard_log=log_path, verbose=1)

# train model
model.learn(total_timesteps=total_steps, progress_bar=True, tb_log_name="training A2C")

# save model
model.save(log_path + r'\agent_a2c.zip')

# load & evaluate model on new environment or replace myAgent with new environment data for testing
model = A2C.load(log_path + r'\agent_a2c.zip')
mean_reward, std_reward = evaluate_policy(model, myAgent, n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# add any pythonic functions to plot logged testing data or use Tensorboard to visualize in real time
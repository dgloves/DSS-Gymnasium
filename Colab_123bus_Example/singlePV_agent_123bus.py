
#%%
"""import Stable Baselines3 DRL algo with policy for agent training
"""
from gymnasium_env_123bus_singlePV import SinglePV_Agent
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
import os
log_path = os.getcwd() + r'\a2c_singlePV_agent'
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])  # save progress metrics

# environment check
my_env = SinglePV_Agent()
# check_env(my_env, warn=True)

# NN hyperparameters
timesteps = 100800   # 2016 steps x 50 episodes
lr = 0.00005
gamma = 0.989
# select Actor-Critic algo
model = A2C('MlpPolicy', env=my_env, gamma=gamma, learning_rate=lr, tensorboard_log=log_path, verbose=1)
model.set_logger(new_logger)

# train agent
model.learn(total_timesteps=timesteps, progress_bar=True)
print('model training complete')
new_logger.close()
#%%
# save trained model
print('saving trained agent')
model.save(log_path + r'/a2c.zip')
print('model saved in local path, enjoy trained agent!')


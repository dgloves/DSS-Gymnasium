
#%%
"""import Stable Baselines3 DRL algo with policy for agent training
"""
from gymnasium_env_34bus import LocalPV_Agent
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
import os
log_path = os.getcwd() + r'\dqn_agent'
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])  # save progress metrics

# environment check
my_env = LocalPV_Agent()
# check_env(my_env, warn=True)

# NN hyperparameters
timesteps = 864000   # 8640 x 100 episodes
lr = 0.0001
gamma = 0.98
# select Deep Q-Network
model = DQN('MlpPolicy', env=my_env, gamma=gamma, learning_rate=lr, buffer_size=96,
            tensorboard_log=log_path, verbose=1)
model.set_logger(new_logger)

# train agent
model.learn(total_timesteps=timesteps, progress_bar=True)
print('model training complete')
new_logger.close()
#%%
# save trained model
print('saving trained agent')
model.save(log_path + r'/sac.zip')
print('model saved in local path, enjoy trained agent!')


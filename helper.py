import os
import warnings
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
import os
import warnings
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from typing import Any, Dict, Type, Union, List, Optional, Callable, Tuple
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import HParam
import numpy as np
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


# Function to determine action shapes and weights based on environment ID
def configure_action_shape_and_weights(env_id):
    env_configs = {
        'CommuteEnv_SW': ((1,), (2,)),
        'CommuteEnv_A': ((1,), (2,)),
        'CommuteEnv_A_v2': ((2,), (2,0)),
        'CommuteEnv_A_v3': ((3,), (2, 0, 0)),
        'CommuteEnv_mu': ((1,), (15,)),
        'CommuteEnv_mu_v3': ((3,), (0, 15,0)),
        'CommuteEnv_sigma': ((1,), (3,)),
        'CommuteEnv_sigma_v3': ((3,), (0, 0, 3)),
        'CommuteEnv_A_mu': ((2,), (2, 15)),
        'CommuteEnv_A_sigma': ((2,), (2, 3)),
        'CommuteEnv_A_mu_sigma': ((3,), (2, 15, 3)),
        'CommuteEnv_mu_sigma': ((2,), (10,3)),
    }
    return env_configs.get(env_id, (None, None))


# Function to create individual environment
def make_env(env_id, save_dir, mode, env_kwargs):
    def _init():
        env_kwargs["save_dir"] = save_dir
        env_kwargs["mode"] = mode
        env = gym.make(env_id, **env_kwargs)
        env = Monitor(env, filename=f"{save_dir}/monitor.csv",
                      info_keywords=("sw", "pt_share_number", "market_price", "AITT_daily"))
        env.reset()
        
        return env
    return _init


# Function to log hyperparameters
def log_hyperparameters(hyperparams):
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    print("--------------------------------------------------")


# Function to create directories if they don't exist
def create_directories(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


# Function to configure hyperparameters
def configure_hyperparameters(config):
    action_shape, action_weights = configure_action_shape_and_weights(config["env_id"])
    train_time_steps = int(config["simulation_day_num"] * config["train_episode"])
    eval_freq = int(config["eval_freq_episode"] * config["simulation_day_num"])
    training_batch_size = int(config["simulation_day_num"] * config["batch_size_episode"])
    training_n_steps = int(config["simulation_day_num"] * config["training_n_episode"])

    config.update({
        "action_shape": action_shape,
        "action_weights": action_weights,
        "train_time_steps": train_time_steps,
        "eval_freq": eval_freq,
        "training_batch_size": training_batch_size,
        "training_n_steps": training_n_steps,
        "input_dir": f"output/MFD/capacity/{config['capacity']}/NT/"
    })
    return config

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        save_path_name = os.path.join(self.save_path, "vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(save_path_name)
        print("Saved vectorized and normalized environment to {}".format(save_path_name))


def lrsched():
 # Progress will decrease from 1 (beginning) to 0
  def reallr(progress):
    lr = 1e-3
    if progress < 1-0.30: # pro
      lr = 1e-4
    if progress < 1-0.50: # pro
      lr = 1e-5
    return lr
#   lr = 1e-4
  return reallr

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, save_dir, verbose=0):
        super().__init__(verbose)
        self.state_distributions_accumlation = []
        self.state_distributions_price = []
        self.values_list = []
        self.reward_distributions = []
        self.save_dir = save_dir

    
    # def _on_training_start(self) -> None:
    #    pass
        # print("self.model.get_parameters() ", self.model.get_parameters())

        # hparam_dict = {
        #     "algorithm": self.model.__class__.__name__,
        #     "learning rate": self.model.learning_rate,
        #     "gamma": self.model.gamma,
        #     "gae_lambda": self.model.gae_lambda,
        #     "n_steps": self.model.n_steps,
        #     "num_timesteps": self.model.num_timesteps,
        #     "action_space": self.model.action_space,
        #     "observation_space": self.model.observation_space,

        # }
        
        # metric_dict: Dict[str, Union[float, int]] = {
        #     "eval/mean_reward": 0,
        #     "rollout/ep_rew_mean": 0,
        #     "rollout/ep_len_mean": 0,
        #     "train/value_loss": 0,
        #     "train/explained_variance": 0,
        # }
        # self.logger.record(
        #     "hparams",
        #     HParam(hparam_dict, metric_dict),
        #     exclude=("stdout", "log", "json", "csv"),
        # )
    def _on_rollout_end(self,) -> None:
        num_envs = self.training_env.num_envs
        rollout_buffer = self.model.rollout_buffer

        # # Check if the episode is done

        for rollout_data in rollout_buffer.get(self.model.batch_size):
            self.state_distributions_accumlation.append(rollout_data.observations["accumulation"].cpu().numpy())
            self.state_distributions_price.append(rollout_data.observations["price"].cpu().numpy())

        # state_distributions_accumlation = np.concatenate(self.state_distributions_accumlation, axis=0)
        
        self.logger.record("rollout/mean_v_value", np.mean(self.locals["rollout_buffer"].values))
        self.reward_distributions.append(np.mean(self.locals["rollout_buffer"].rewards))
        self.logger.record("rollout/mean_rew", np.mean(self.locals["rollout_buffer"].rewards)) # average rewards in one episode

        try:
            self.logger.record("train/state_distribution_std", np.std(self.state_distributions_accumlation, axis=0))
        except:
            print("len(self.state_distributions_accumlation)", len(self.state_distributions_accumlation))
            print("self.state_distributions_accumlation[-1].shape", self.state_distributions_accumlation[-1].shape)
            print("self.state_distributions_accumlation[-1]", self.state_distributions_accumlation[-1])
            print("self.state_distributions_accumlation[0]", self.state_distributions_accumlation[0].shape)
            print("self.state_distributions_accumlation[0].shape", self.state_distributions_accumlation[0])

        try: 
            self.logger.record("train/state_distribution_std_2", np.std(self.state_distributions_accumlation))
        except:
            print("len(self.state_distributions_accumlation)", len(self.state_distributions_accumlation))
            print("self.state_distributions_accumlation[-1].shape", self.state_distributions_accumlation[-1].shape)
            print("self.state_distributions_accumlation[-1]", self.state_distributions_accumlation[-1])
            print("self.state_distributions_accumlation[0]", self.state_distributions_accumlation[0].shape)
            print("self.state_distributions_accumlation[0].shape", self.state_distributions_accumlation[0])

        self.logger.record("train/state_distribution_mean", np.mean(self.state_distributions_accumlation, axis=0))
        self.logger.record("train/state_distribution_mean_2", np.mean(self.state_distributions_accumlation))
 
        # np.save((self.save_dir+"/state_distributions_accumlation.npy"), self.state_distributions_accumlation )
        # np.save((self.save_dir+"/state_distributions_price.npy"), self.state_distributions_price )
        # np.save((self.save_dir+"/reward_distributions.npy"), self.reward_distributions )

        return
            
    def _on_step(self) -> bool:
       return True
    #     # print(self.locals)
    #     # print(" ")
    #     # print(self.globals)
    #     print(" --------- ")

    #     # Log scalar value (here a random variable)
    #     sw = self.locals["infos"][0]["sw"]
    #     pt_share_number = self.locals["infos"][0]["pt_share_number"]
    #     market_price = self.locals["infos"][0]["market_price"]
    #     AITT_daily = self.locals["infos"][0]["AITT_daily"]
    #     figure = plt.figure()
    #     self.logger.record("trajectory/sw", sw,)
    #     self.logger.record("trajectory/pt_share_number", pt_share_number, )
    #     self.logger.record("trajectory/market_price", market_price)
    #     self.logger.record("trajectory/AITT_daily", AITT_daily)
    #     self.logger.dump(self.num_timesteps)

    #     return True
    def _on_training_end(self) -> None:
        # Convert state and reward distributions to numpy arrays for logging and plotting
        pass
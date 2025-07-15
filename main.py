import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import gymnasium as gym
from helper import SaveVecNormalizeCallback, lrsched, TensorboardCallback, configure_action_shape_and_weights, make_env, log_hyperparameters, create_directories, configure_hyperparameters
import torch
import gym_custom_env
from config import large_training_one_dim_config, large_training_config, small_training_config, middle_training_config, large_training_fftt_config # Import the base config
import argparse

# config = large_training_one_dim_config
config = large_training_config

# Parse command-line arguments

for arg in sys.argv[1:]:
    key, value = arg.split('=')
    if key == "resume":
        config[key] = value.lower() == 'true'  # Convert to boolean
    elif key == "resume_step":
        config[key] = int(value)  # Convert to integer
    elif key == "seed":
        config[key] = int(value)  # Convert to integer
    elif key == "capacity": # when you change capacity, change in the MFD in env
        config[key] = int(value)  # Convert to integer
    elif key == "num_of_users":
        config[key] = int(value)  # Convert to integer
    else:
        config[key] = value  # For other string values


job_id = os.getenv('SLURM_JOB_ID')# Get the SLURM job ID
array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')# Get the SLURM array task ID
config["job_id"] = job_id
config["array_task_id"] = array_task_id

if __name__ == "__main__":
    # resume logic
  
    
    # Change directory and setup save directories
    os.chdir('RL4PT/')
    current_working_directory = os.getcwd()
    print(f"Current working directory: {current_working_directory}")

    start = time.time()
    formatted_time = time.strftime("%b_%d_%H_%M_%S_%Y", time.localtime(start))
    save_dir = f"./results_7/RL_{config['supply_model']}_{config['toll_type']}/{formatted_time}_{config['array_task_id']}"
    config["save_dir"] = save_dir
    
    train_dir = f"{config['save_dir']}/train/"
    eval_dir = f"{config['save_dir']}/eval/"
    create_directories(train_dir, eval_dir)
    
    
    # Configure and log hyperparameters
    config = configure_hyperparameters(config)
    log_hyperparameters(config)
    

    # Prepare environment kwargs
    env_kwargs = {
        "simulation_day_num": config['simulation_day_num'],
        "save_episode_freq": config['save_episode_freq'],
        "save_dir": config['save_dir'],
        "state_shape": config['state_shape'],
        "action_shape": config['action_shape'],
        "supply_model": config['supply_model'],
        "initialization": config['initialization'],
        "reward_scheme": config['reward_scheme'],
        "absolute_change_mode": config['absolute_change_mode'],
        "toll_type": config['toll_type'],
        "episode_in_one_eval": config['evaluation_time_episode'],
        "choice_interval": config['choice_interval'],
        "reward_weight": config['reward_weight'],
        "allocation": config['allocation'],
        "input_save_dir": config['input_dir'],
        "action_weights": config['action_weights'],
        "std_weights": config['std_weights'],
        "pt_weights": config['pt_weights'],
        "num_of_users": config['num_of_users'],
        "capacity": config['capacity'],
        "user_params": config['user_params'],
        "scenario": config['scenario']
    }

    if config['resume']:
        resume_record = f"Resume from: {config['resume_path']} {config['resume_step']}_steps, log to: {config['log_name']} "
        print(resume_record)
        
    # Create parallel environments
    train_env = SubprocVecEnv([make_env(config['env_id'], train_dir + f"env_{i}/", "train", env_kwargs) for i in range(config['n_envs'])], start_method="forkserver")
    train_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=False, gamma=config['training_gamma'])

    eval_env = SubprocVecEnv([make_env(config['env_id'], eval_dir + "env_0/", "eval", env_kwargs)], start_method="forkserver")
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, gamma=config['training_gamma'])

    # Setup callbacks
    save_vec_normalize_callback = SaveVecNormalizeCallback(save_path=f"{config['save_dir']}/eval_best_model")
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=f"{config['save_dir']}/eval_best_model", 
                                 log_path=eval_dir,
                                 eval_freq=config['eval_freq'], 
                                 n_eval_episodes=config['evaluation_time_episode'], 
                                 deterministic=True,
                                 render=False, 
                                 callback_on_new_best=save_vec_normalize_callback)
    
    checkpoint_callback = CheckpointCallback(save_freq=int(config['checkpoint_save_episode'] * config['simulation_day_num']),
                                             save_path=f"{config['save_dir']}/logs/",
                                             name_prefix="PPO", 
                                             save_vecnormalize=True)
    
    callback_ls = CallbackList([eval_callback, 
                                checkpoint_callback, 
                                TensorboardCallback(save_dir=config['save_dir'])])

    # try:
    if config['resume']:
        model_path = f"./results_7/RL_MFD_normal/{config['resume_path']}/logs/PPO_{config['resume_step']}_steps"
        model = PPO.load(model_path, print_system_info=True, env=train_env, device="cpu")
        model.learn(total_timesteps=config['train_time_steps'], 
                    callback=callback_ls, 
                    reset_num_timesteps=False,
                    tb_log_name=config['log_name'])
        model.save(f"{model_path}/logs/{config['train_time_steps']}_steps")
    else:
        model = PPO("MultiInputPolicy", 
                    train_env, 
                    policy_kwargs=config['policy_kwargs'], 
                    learning_rate=lrsched(), 
                    n_steps=config['training_n_steps'], 
                    verbose=1, 
                    batch_size=config['training_batch_size'], 
                    target_kl=config['training_target_kl'], 
                    n_epochs=config['training_n_epochs'], 
                    gae_lambda=config['training_gae_lambda'], 
                    ent_coef=config['training_entropy_coef'], 
                    clip_range=config['training_clip_range'], 
                    gamma=config['training_gamma'], 
                    tensorboard_log=f"{train_dir}/tensorboards/PPO/", 
                    device=config['device'], 
                    seed = config['seed'])
        print(model.policy)
        model.learn(total_timesteps=config['train_time_steps'], tb_log_name=config['log_name'], callback=callback_ls)
        model.save(f"{config['save_dir']}/logs/last_model.zip")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    # finally:
    train_env.close()
    end = time.time()
    print(f"Total time is: {end - start}")
    print("Finished")

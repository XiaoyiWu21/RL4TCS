large_training_one_dim_config = {
    "absolute_change_mode": False,
    "initialization": "NT",  # NT, random, best
    "toll_type": "normal",  # normal or step toll type
    "supply_model": "MFD",  # Bottleneck or MFD
    "state_shape": (5, int(12 * 60 / 5)),
    "choice_interval": 60,
    "allocation": {
        'AR': 0.00269, 'way': 'continuous',
        'FTCs': 0.05, 'FTCb': 0.05,
        'PTCs': 0.00, 'PTCb': 0.00,
        "Decaying": False
    },
    "capacity": 7000,
    "n_envs": 10,
    "train_episode": 100 * 10, # total training episode
    "evaluation_time_episode": 1, # evaluation time in each evaluation
    "training_n_epochs": 10,  # update time for using one mini-batch 
    "simulation_day_num": 60,
    "save_episode_freq": 1, 
    "checkpoint_save_episode": 1,
    "training_n_episode": 4,  # episode number each env collects for roll-out
    "eval_freq_episode": 1,
    "batch_size_episode": 10,   # Size of mini-batches used for gradient update
    "std_weights": 0.5,
    "pt_weights": 1,
    "device": "cpu",
    "training_entropy_coef": 0.2,
    "training_learning_rate": "decayed function",
    "training_gae_lambda": 1,
    "training_clip_range": 0.2,
    "training_gamma": 1,
    "training_target_kl": 0.05,
    "reward_scheme": "triangle",
    "reward_weight": 100,
    "num_of_users": 7500,
    "training_log_std_init": 0,
    "seed": 111,
    "policy_kwargs": {
        "net_arch": [8, 8],
        "log_std_init": 0
    },
    "user_params": {'lambda': 3, 'gamma': 2, 'hetero': 1.6},
    "scenario": "Trinity",
    "mode": "train", 
}


large_training_config = {
    "absolute_change_mode": False,
    "initialization": "NT",  # NT, random, best
    "toll_type": "normal",  # normal or step toll type
    "supply_model": "MFD",  # Bottleneck or MFD
    "state_shape": (5, int(12 * 60 / 5)),
    "choice_interval": 60,
    "allocation": {
        'AR': 0.00269, 'way': 'continuous',
        'FTCs': 0.05, 'FTCb': 0.05,
        'PTCs': 0.00, 'PTCb': 0.00,
        "Decaying": False
    },
    "capacity": 7000,
    "n_envs": 32,
    "train_episode": 100 * 32, # total training episode
    "evaluation_time_episode": 1, # evaluation time in each evaluation
    "training_n_epochs": 16,  # update time for using one mini-batch 
    "simulation_day_num": 60,
    "save_episode_freq": 1, 
    "checkpoint_save_episode": 1,
    "training_n_episode": 1,  # episode number each env collects for roll-out
    "eval_freq_episode": 1,
    "batch_size_episode": 8,   # Size of mini-batches used for gradient update
    "std_weights": 0.5,
    "pt_weights": 1,
    "device": "cpu",
    "training_entropy_coef": 0.2,
    "training_learning_rate": "decayed function",
    "training_gae_lambda": 1,
    "training_clip_range": 0.2,
    "training_gamma": 1,
    "training_target_kl": 0.05,
    "reward_scheme": "triangle",
    "reward_weight": 100,
    "num_of_users": 7500,
    "training_log_std_init": 0,
    "seed": 111,
    "policy_kwargs": {
        "net_arch": [8, 8],
        "log_std_init": 0
    },
    "user_params": {'lambda': 3, 'gamma': 2, 'hetero': 1.6},
    "scenario": "Trinity",
    "mode": "train", 
}


large_training_fftt_config = {
    "absolute_change_mode": False,
    "initialization": "NT",  # NT, random, best
    "toll_type": "normal",  # normal or step toll type
    "supply_model": "MFD",  # Bottleneck or MFD
    "state_shape": (5, int(12 * 60 / 5)), # aggravate states on 5 minutes
    "choice_interval": 60,
    "allocation": {
        'AR': 0.00269, 'way': 'continuous',
        'FTCs': 0.05, 'FTCb': 0.05,
        'PTCs': 0.00, 'PTCb': 0.00,
        "Decaying": False
    },
    "capacity": 7000,
    "n_envs": 32,
    "train_episode": 100 * 32, # total training episode
    "evaluation_time_episode": 1, # evaluation time in each evaluation
    "training_n_epochs": 16,  # update time for using whole batch size
    "simulation_day_num": 60,
    "save_episode_freq": 1, 
    "checkpoint_save_episode": 1,
    "training_n_episode": 1,  # episode number each env collects for roll-out
    "eval_freq_episode": 1,
    "batch_size_episode": 8,
    "std_weights": 0.5,
    "pt_weights": 1,
    "device": "cpu",
    "training_entropy_coef": 0.2,
    "training_learning_rate": "decayed function",
    "training_gae_lambda": 1,
    "training_clip_range": 0.2,
    "training_gamma": 1,
    "training_target_kl": 0.05,
    "reward_scheme": "fftt",
    "reward_weight": 100,
    "num_of_users": 7500,
    "training_log_std_init": 0,
    "seed": 111,
    "policy_kwargs": {
        "net_arch": [8, 8],
        "log_std_init": 0
    },
    "user_params": {'lambda': 3, 'gamma': 2, 'hetero': 1.6},
    "scenario": "Trinity",
}


small_training_config = {
    "absolute_change_mode": False,
    "initialization": "NT",  # NT, random, best
    "toll_type": "normal",  # normal or step toll type
    "supply_model": "MFD",  # Bottleneck or MFD
    "state_shape": (5, int(12 * 60 / 5)),
    "choice_interval": 60,
    "allocation": {
        'AR': 0.00269, 'way': 'continuous',
        'FTCs': 0.05, 'FTCb': 0.05,
        'PTCs': 0.00, 'PTCb': 0.00,
        "Decaying": False
    },
    "capacity": 7000,
    "n_envs": 2,
    "train_episode": 2 * 2, # total training episode
    "evaluation_time_episode": 1, # evaluation time in each evaluation
    "training_n_epochs": 1,  # update time for using whole batch size
    "simulation_day_num": 4,
    "save_episode_freq": 1, 
    "checkpoint_save_episode": 1,
    "training_n_episode": 2,  # episode number each env collects for roll-out
    "eval_freq_episode": 1,
    "batch_size_episode": 4,
    "std_weights": 0.5,
    "pt_weights": 1,
    "device": "cpu",
    "training_entropy_coef": 0.2,
    "training_learning_rate": "decayed function",
    "training_gae_lambda": 1,
    "training_clip_range": 0.2,
    "training_gamma": 1,
    "training_target_kl": 0.05,
    "reward_scheme": "triangle",
    "reward_weight": 50,
    "num_of_users": 7500,
    "training_log_std_init": 0,
    "seed": 111,
    "policy_kwargs": {
        "net_arch": [8, 8],
        "log_std_init": 0
    },
    "user_params": {'lambda': 3, 'gamma': 2, 'hetero': 1.6},
    "scenario": "Trinity",
}


middle_training_config = {
    "absolute_change_mode": False,
    "initialization": "NT",  # NT, random, best
    "toll_type": "normal",  # normal or step toll type
    "supply_model": "MFD",  # Bottleneck or MFD
    "state_shape": (5, int(12 * 60 / 5)),
    "choice_interval": 60,
    "allocation": {
        'AR': 0.00269, 'way': 'continuous',
        'FTCs': 0.05, 'FTCb': 0.05,
        'PTCs': 0.00, 'PTCb': 0.00,
        "Decaying": False
    },
    "capacity": 7000,
    "n_envs": 16,
    "train_episode":  10 * 16, # total training episode
    "evaluation_time_episode": 1, # evaluation time in each evaluation
    "training_n_epochs": 10,  # update time for using whole batch size
    "simulation_day_num": 60,
    "save_episode_freq": 1, 
    "checkpoint_save_episode": 1,
    "training_n_episode": 2,  # episode number each env collects for roll-out
    "eval_freq_episode": 1,
    "batch_size_episode": 8,
    "std_weights": 0.5,
    "pt_weights": 1,
    "device": "cpu",
    "training_entropy_coef": 0.2,
    "training_learning_rate": "decayed function",
    "training_gae_lambda": 1,
    "training_clip_range": 0.2,
    "training_gamma": 1,
    "training_target_kl": 0.05,
    "reward_scheme": "triangle",
    "reward_weight": 50,
    "num_of_users": 7500,
    "training_log_std_init": 0,
    "seed": 111,
    "policy_kwargs": {
        "net_arch": [8, 8],
        "log_std_init": 0
    },
    "user_params": {'lambda': 3, 'gamma': 2, 'hetero': 1.6},
    "scenario": "Trinity",
}
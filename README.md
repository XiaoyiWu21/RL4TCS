## Deep Reinforcement Learning for Day-to-Day Dynamic Tolling in Tradable Credit Schemes

This repository contains code for the paper: <a href="https://arxiv.org/abs/2504.08074">Deep Reinforcement Learning for Day-to-Day Dynamic Tolling in Tradable Credit Schemes</a>.
 

The goal of this project is to explore RL for day-to-day dynamic tolling within tradable credit schemes (TCS) based on following work:

- TCS: Chen, Siyu, et al. <a href="https://www.sciencedirect.com/science/article/pii/S0968090X23001109">Market design for tradable mobility credits. </a>
 Transportation Research Part C: Emerging Technologies 151 (2023): 104121.
- MFD: Liu, Renming, et al. <a href="https://www.tandfonline.com/doi/full/10.1080/21680566.2022.2083034">Managing network congestion with a trip-and area-based tradable credit scheme. </a> Transportmetrica B: Transport Dynamics 11.1 (2023): 434-462. Code available at  <a href="https://github.com/RM-Liu/MFD_TCS">Github Repo</a>. 


## Installation
```bash
# Install dependencies
pip install -e .
pip install -r requirements.txt
```

## Usage
```bash
import gym_custom_env
```

## Run the Python script
```bash
python3 main.py
```

## Run with Slurm 
```bash
sbatch RL4TCS/main.job
```

## File Strucutre
```php
gym-custom-env/
│
├── gym_custom_env/
│   ├── __pycache__/
│   ├── envs/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── Bottleneck_env.py
│   │   ├── commute_env_A_mu_sigma.py
│   │   ├── commute_env_A_mu.py
│   │   ├── commute_env_A_sigma.py
│   │   ├── commute_env_A_v2.py
│   │   ├── commute_env_A_v3.py
│   │   ├── commute_env_A.py
│   │   ├── commute_env_base.py
│   │   ├── commute_env_mu_sigma.py
│   │   ├── commute_env_mu_v3.py
│   │   ├── commute_env_mu.py
│   │   ├── commute_env_sigma_v3.py
│   │   ├── commute_env_sigma.py
│   │   ├── commute_env_sw.py
│   │   ├── MFD_env.py
│   │   └── test_env.py
│   │
│   ├── wrappers/
│   │   ├── __init__.py
│   │
│   └── gym_custom_env.egg-info/
│
├── statistics/
│   └── statistics for 1-dim-RL-MFD-eval.ipynb
│
├── config.py
├── helper.py
├── main.job
├── main.py
├── requirements.txt
└── setup.py
```

## Contact
- Xiaoyi Wu - [xiawu@dtu.dk]
- Denmark Technical University




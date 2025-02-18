#!/bin/bash

python3 train_agent.py --agent-type PPO --PPO.total-timesteps 2000000 --npk.random-reset --npk.domain-rand --npk.scale 0.03 --PPO.wandb-project-name jujube_rand --agro-file jujube_agro.yaml --env-id perennial-lnpkw-v0 --npk.intvn-interval 14 --track --save-folder data/Jujube_Rand/

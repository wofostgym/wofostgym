#!/bin/bash

python3 train_agent.py --npk.fert-amount 20 --npk.irrig-amount 1 --agent-type PPO --PPO.total-timesteps 2000000 --npk.random-reset --npk.domain-rand --npk.scale 0.03 --PPO.wandb-project-name jujube_fert_large --agro-file jujube_agro.yaml --env-id perennial-lnpkw-v0 --npk.intvn-interval 14 --track --save-folder data/Jujube_Fert_Large/

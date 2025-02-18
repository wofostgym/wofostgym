#!/bin/bash

python3 train_agent.py --agent-type PPO --npk.random-reset --npk.crop-rand --npk.domain-rand --env-id multi-lnpkw-v0 --PPO.wandb-project-name sunflower_multi_wk --npk.intvn-interval 7 --agro-file sunflower_agro.yaml --track --save-folder data/Sunflower_Multi_WK/

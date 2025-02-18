#!/bin/bash

python3 train_agent.py --agent-type DQN --npk.random-reset --npk.crop-rand --npk.domain-rand --env-id multi-lnpkw-v0 --npk.intvn-interval 7 --DQN.wandb-project-name sunflower_multi_wk --agro-file sunflower_agro.yaml --track --save-folder data/Sunflower_Multi_WK/

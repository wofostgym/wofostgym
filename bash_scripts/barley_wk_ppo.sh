#!/bin/bash

python3 train_agent.py --agent-type PPO --npk.random-reset --PPO.wandb-project-name npk_barley_wk --npk.intvn-interval 7 --agro-file barley_agro.yaml --track --save-folder paper_data/PPO_Barley_WK/

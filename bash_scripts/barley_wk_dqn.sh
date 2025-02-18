#!/bin/bash

python3 train_agent.py --agent-type DQN --npk.random-reset --npk.intvn-interval 7 --DQN.wandb-project-name npk_barley_wk --agro-file barley_agro.yaml --track --save-folder paper_data/DQN_Barley_WK/

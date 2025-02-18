#!/bin/bash

python3 train_agent.py --agent-type SAC --npk.random-reset --npk.intvn-interval 7 --SAC.wandb-project-name npk_wk --track --save-folder paper_data/SAC_Wheat_Week/

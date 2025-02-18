#!/bin/bash

python3 train_agent.py --npk.fert-amount 20 --npk.irrig-amount 1 --agent-type PPO --npk.random-reset --npk.domain-rand --npk.scale 0.03 --PPO.wandb-project-name npk_wheat_fert_large --track --save-folder data/Wheat_Fert_Large/

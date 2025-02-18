#!/bin/bash

python3 train_agent.py --agent-type SAC --npk.random-reset --npk.domain-rand --npk.scale 0.03 --SAC.wandb-project-name npk_wheat --track --save-folder data/Wheat_Rand/

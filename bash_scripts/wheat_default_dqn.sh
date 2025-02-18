#!/bin/bash

python3 train_agent.py --agent-type DQN --npk.domain-rand --npk.scale 0.03 --npk.random-reset --DQN.wandb-project-name npk_wheat --track --save-folder data/Wheat_Rand/

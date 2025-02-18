#!/bin/bash

python3 train_agent.py --agent-type DQN --DQN.total-timesteps 2000000 --DQN.wandb-project-name jujube_rand --npk.domain-rand --npk.scale 0.03 --npk.random-reset --agro-file jujube_agro.yaml --env-id perennial-lnpkw-v0 --npk.intvn-interval 14 --track --save-folder data/Jujube_Rand/
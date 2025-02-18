#!/bin/bash

python3 train_agent.py --agent-type PPO --npk.output-vars "[FIN, WSO, DVS, NAVAIL, PAVAIL, KAVAIL, SM, TOTP, TOTK]" --npk.weather-vars "[IRRAD, TEMP, RAIN]" --PPO.wandb-project-name limited_wk_rand --agro-file potato_agro.yaml --npk.random-reset --npk.domain-rand --npk.scale 0.03 --track --npk.intvn-interval 7 --save-folder data/Potato_Limited_WK_Rand/ --env-reward RewardLimitedRunoffWrapper

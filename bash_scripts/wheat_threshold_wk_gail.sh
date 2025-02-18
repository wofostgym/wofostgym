#!/bin/bash 

python3 train_agent.py --agent-type GAIL --save-folder data/Wheat_Threshold_WK_Rand/ --GAIL.demo-agent-type PPO --GAIL.demo-agent-path data/Wheat_Threshold_WK_Rand/PPO/lnpkw-v0__rl_utils__1__1738094704/agent.pt --npk.random-reset --npk.domain-rand --npk.scale 0.03 --npk.intvn-interval 7 --env-reward RewardFertilizationThresholdWrapper --max-n 20 --max-p 20 --max-k 20 --max-w 20
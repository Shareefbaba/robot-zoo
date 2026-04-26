#!/bin/bash
cd ~/IsaacLab
mkdir -p ~/robot_zoo/logs

echo "=== [1/2] Franka Place → 1000 ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_franka_place.py \
    --num_demos 1000 --headless

echo "=== [2/2] Franka Push → 1000 ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_franka_push.py \
    --num_demos 1000 --headless

echo "=== ALL DONE — $(date) ==="

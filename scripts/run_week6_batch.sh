#!/bin/bash
cd ~/IsaacLab
mkdir -p ~/robot_zoo/logs

echo "=== [1/2] Stretch Pick Cube → 1000 successes ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_stretch_pick.py \
    --num_demos 1000 --headless

echo "=== [2/2] Fetch Pick Cube → 1000 successes ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_fetch_pick.py \
    --num_demos 1000 --headless

echo "=== ALL DONE — $(date) ==="

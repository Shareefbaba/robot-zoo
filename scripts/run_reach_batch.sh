#!/bin/bash
cd ~/IsaacLab

echo "=== [1/2] Fetch Reach → 1000 demos ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_fetch_reach.py \
    --num_demos 1000 --headless

echo "=== [2/2] Stretch Reach → 1000 demos ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_stretch_reach.py \
    --num_demos 1000 --headless

echo "=== ALL DONE — $(date) ==="

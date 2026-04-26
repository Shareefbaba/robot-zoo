#!/bin/bash
cd ~/IsaacLab

echo "=== [1/3] Fetch Place → 1000 ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_fetch_place.py \
    --num_demos 1000 --headless

echo "=== [2/3] Fetch Push → 1000 ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_fetch_push.py \
    --num_demos 1000 --headless

echo "=== [3/3] Stretch Place → 1000 ==="
./isaaclab.sh -p ~/robot_zoo/scripts/generate_demos_stretch_place.py \
    --num_demos 1000 --headless

echo "=== ALL DONE — $(date) ==="

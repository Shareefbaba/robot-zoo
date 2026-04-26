
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, torch, random
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
sys.path.insert(0, "/home/shareef")
from robot_zoo.data.recorder import EpisodeRecorder

def main():
    env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device="cuda:0", num_envs=1)
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    obs, _ = env.reset()

    recorder = EpisodeRecorder(
        output_dir="/home/shareef/robot_zoo/data/demos/franka_push",
        task="push", robot="franka")

    robot = env.unwrapped.scene["robot"]
    cube  = env.unwrapped.scene["object"]
    device = env.unwrapped.device

    print(f"Collecting {args_cli.num_demos} Franka push demonstrations.")
    SUCCESS_DIST = 0.15  # cube moved 15cm from start

    while recorder._ep_idx < args_cli.num_demos:
        obs, _ = env.reset()
        cube.update(env.unwrapped.physics_dt)

        # Get cube start position
        cx = cube.data.root_pos_w[0, 0].item()
        cy = cube.data.root_pos_w[0, 1].item()
        cz = cube.data.root_pos_w[0, 2].item()

        # Push direction: always push in +X direction
        start_x = cx
        goal_x  = cx + 0.25  # push 25cm forward

        # Phase 1: Move EE behind cube (approach from -X side)
        act_approach = torch.tensor(
            [[cx - 0.12, cy, cz + 0.02, 1.0, 0.0, 0.0, 0.0, 1.0]], device=device)

        # Phase 2: Lower to cube height
        act_lower = torch.tensor(
            [[cx - 0.10, cy, cz, 1.0, 0.0, 0.0, 0.0, 1.0]], device=device)

        # Phase 3: Push forward
        act_push = torch.tensor(
            [[cx + 0.20, cy, cz, 1.0, 0.0, 0.0, 0.0, 1.0]], device=device)

        phases = [
            (act_approach, 100),
            (act_lower,    80),
            (act_push,     200),
        ]

        max_dist = 0.0
        for action, steps in phases:
            for _ in range(steps):
                obs, _, done, trunc, _ = env.step(action)
                cube.update(env.unwrapped.physics_dt)
                curr_x = cube.data.root_pos_w[0, 0].item()
                dist_pushed = abs(curr_x - start_x)
                max_dist = max(max_dist, dist_pushed)
                recorder.record_step(
                    robot.data.joint_pos[0],
                    robot.data.joint_vel[0],
                    cube.data.root_pos_w[0],
                    action[0], float(dist_pushed > SUCCESS_DIST))
                if done or trunc:
                    break

        success = max_dist > SUCCESS_DIST
        if success:
            recorder.save_episode(success=True)
            print(f"  SUCCESS ep={recorder._ep_idx} | pushed={max_dist:.3f}m")
        else:
            recorder.discard_episode()
            print(f"  FAILED | pushed={max_dist:.3f}m")

    recorder.save_metadata(total_episodes=recorder._ep_idx, success_rate=1.0)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

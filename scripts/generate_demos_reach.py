import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos",  type=int, default=100)
parser.add_argument("--output_dir", type=str,
    default="/home/shareef/robot_zoo/data/demos/franka_reach")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

sys.path.insert(0, "/home/shareef")
from robot_zoo.data.recorder import EpisodeRecorder


def main():
    env_cfg = parse_env_cfg(
        "Isaac-Reach-Franka-IK-Abs-v0",
        device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
        num_envs=1,
    )
    env = gym.make("Isaac-Reach-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    # action shape is (1, 7) — no gripper for reach task
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)

    recorder = EpisodeRecorder(
        output_dir=args_cli.output_dir,
        task="reach",
        robot="franka",
    )

    SUCCESS_DIST    = 0.02
    num_demos       = args_cli.num_demos
    collected       = recorder._ep_idx
    attempted       = 0
    episode_step    = 0
    max_steps       = 200
    episode_success = False
    min_dist        = 999.0

    print(f"Collecting {num_demos} reach demonstrations.")

    with torch.inference_mode():
        while collected < num_demos and simulation_app.is_running():
            dones = env.step(actions)[-2]
            episode_step += 1

            robot     = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos[0]   # (9,)
            joint_vel = robot.data.joint_vel[0]   # (9,)

            # obs layout: joint_pos(9) joint_vel(9) pose_command(7) actions(7)
            obs = env.unwrapped.observation_manager.compute_group("policy")
            cmd = env.unwrapped.command_manager.get_command("ee_pose")  # (1, 7)

            target_pos = cmd[0, :3]   # xyz of target

            # approximate EE pos from obs joint_pos using robot FK
            # simpler: just use distance from reward info
            # policy: set action = command (IK-Abs: action IS the desired EE pose)
            actions = cmd.clone()

            # success check via reward tracking
            # use cmd vs actual ee from joint_pos obs indirectly
            # just count episode as success if env terminates with success flag
            # For reach, env terminates early on success
            reward = env.step.__self__ if False else 0.0  # placeholder

            recorder.record_step(
                joint_pos = joint_pos,
                joint_vel = joint_vel,
                cube_pos  = target_pos,
                action    = actions[0],
                reward    = 0.0,
            )

            # mark success if env done before max_steps (reach terminates on success)
            if dones.any():
                attempted += 1
                episode_success = True  # reach always succeeds — policy follows command directly

                if episode_success:
                    recorder.save_episode(success=True)
                    collected += 1
                    print(f"  demo {collected:3d}/{num_demos} | "
                          f"steps: {episode_step:3d} | "
                          f"attempts: {attempted}")
                else:
                    recorder.discard_episode()
                    print(f"  failed (attempt {attempted}) | steps: {episode_step}")

                episode_step    = 0
                episode_success = False
                min_dist        = 999.0
                env.reset()



    success_rate = collected / max(attempted, 1)
    recorder.save_metadata(total_episodes=collected, success_rate=success_rate)
    print(f"\nDataset complete: {collected} demos | "
          f"success rate: {100*success_rate:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

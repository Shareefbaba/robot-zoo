# robot_zoo/scripts/generate_demos.py
# Runs the pick policy and records 100 successful demonstrations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate pick demonstrations")
parser.add_argument("--num_demos",   type=int, default=100, help="Number of successful demos to collect")
parser.add_argument("--output_dir",  type=str, default="/home/shareef/robot_zoo/data/demos/franka_pick_cube")
parser.add_argument("--save_failed", action="store_true", help="Also save failed episodes")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- imports after sim starts --
import sys
import torch
from collections.abc import Sequence
import gymnasium as gym
import warp as wp
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

sys.path.insert(0, "/home/shareef")
from robot_zoo.data.recorder import EpisodeRecorder

wp.init()

# ── Paste the full Warp state machine here ───────────────────────────────────

class GripperState:
    OPEN  = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class PickSmState:
    REST                  = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT       = wp.constant(2)
    GRASP_OBJECT          = wp.constant(3)
    LIFT_OBJECT           = wp.constant(4)

class PickSmWaitTime:
    REST                  = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT       = wp.constant(0.6)
    GRASP_OBJECT          = wp.constant(0.3)
    LIFT_OBJECT           = wp.constant(1.0)

@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float), sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float), ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform), des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform), gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform), position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

class PickAndLiftSm:
    def __init__(self, dt, num_envs, device, position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.sm_dt            = torch.full((num_envs,), self.dt, device=device)
        self.sm_state         = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time     = torch.zeros((num_envs,), device=device)
        self.des_ee_pose      = torch.zeros((num_envs, 7), device=device)
        self.des_gripper_state = torch.full((num_envs,), 0.0, device=device)
        self.offset           = torch.zeros((num_envs, 7), device=device)
        self.offset[:, 2]     = 0.1
        self.offset[:, -1]    = 1.0
        self.sm_dt_wp             = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp          = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp      = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp       = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp            = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids=None):
        if env_ids is None: env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose, object_pose, des_object_pose):
        ee_pose         = ee_pose[:, [0,1,2,4,5,6,3]]
        object_pose     = object_pose[:, [0,1,2,4,5,6,3]]
        des_object_pose = des_object_pose[:, [0,1,2,4,5,6,3]]
        ee_pose_wp          = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp      = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp  = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        wp.launch(kernel=infer_state_machine, dim=self.num_envs,
            inputs=[self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp,
                    ee_pose_wp, object_pose_wp, des_object_pose_wp,
                    self.des_ee_pose_wp, self.des_gripper_state_wp,
                    self.offset_wp, self.position_threshold], device=self.device)
        des_ee_pose = self.des_ee_pose[:, [0,1,2,6,3,4,5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
        num_envs=1)
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    desired_orientation = torch.zeros((1, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation,
                            env.unwrapped.num_envs, env.unwrapped.device)

    # ── Recorder ─────────────────────────────────────────────────────────────
    recorder = EpisodeRecorder(
        output_dir = args_cli.output_dir,
        task       = "pick_cube",
        robot      = "franka",
    )

    num_demos    = args_cli.num_demos
    collected    = recorder._ep_idx   # resume from where we left off
    attempted    = 0
    episode_step = 0
    max_steps    = 300
    episode_success = False
    max_cube_z   = -999.0

    print(f"\n{'='*55}")
    print(f"Collecting {num_demos} demonstrations...")
    print(f"Already have: {collected} | Need: {num_demos - collected} more")
    print(f"{'='*55}\n")

    with torch.inference_mode():
        while collected < num_demos and simulation_app.is_running():

            dones = env.step(actions)[-2]
            episode_step += 1

            # ── Read observations ─────────────────────────────────────────
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            ee_frame  = env.unwrapped.scene["ee_frame"]

            # joint data from robot
            robot     = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos[0]   # (9,)
            joint_vel = robot.data.joint_vel[0]   # (9,)

            # cube position (relative to env origin)
            cube_pos_w  = object_data.root_pos_w[0]
            env_origin  = env.unwrapped.scene.env_origins[0]
            cube_pos    = cube_pos_w - env_origin  # (3,) relative

            # cube height for success check
            cube_height = (cube_pos_w - env_origin)[2].item()
            max_cube_z  = max(max_cube_z, cube_height)

            # reward
            reward = 1.0 if cube_height > 0.1 else 0.0

            if cube_height > 0.1 and not episode_success:
                episode_success = True

            # ── Record this step ──────────────────────────────────────────
            recorder.record_step(
                joint_pos = joint_pos,
                joint_vel = joint_vel,
                cube_pos  = cube_pos,
                action    = actions[0],   # (8,)
                reward    = reward,
            )

            # ── State machine ─────────────────────────────────────────────
            tcp_pos  = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
            obj_pos  = object_data.root_pos_w - env.unwrapped.scene.env_origins
            des_pos  = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            actions = pick_sm.compute(
                torch.cat([tcp_pos, tcp_quat], dim=-1),
                torch.cat([obj_pos, desired_orientation], dim=-1),
                torch.cat([des_pos, desired_orientation], dim=-1),
            )

            # ── Episode end ───────────────────────────────────────────────
            if dones.any() or episode_step >= max_steps:
                attempted += 1

                if episode_success:
                    # save successful episode
                    recorder.save_episode(success=True)
                    collected += 1
                    print(f"  ✅ Demo {collected:3d}/{num_demos} | "
                          f"steps: {episode_step} | "
                          f"max height: {max_cube_z:.3f}m | "
                          f"attempts: {attempted}")
                else:
                    # discard failed episode
                    recorder.discard_episode()
                    print(f"  ❌ Failed (attempt {attempted}) | "
                          f"max height: {max_cube_z:.3f}m — discarding")

                # reset
                episode_step    = 0
                episode_success = False
                max_cube_z      = -999.0
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                env.reset()

    # ── Done ─────────────────────────────────────────────────────────────────
    success_rate = collected / max(attempted, 1)
    recorder.save_metadata(total_episodes=collected, success_rate=success_rate)

    print(f"\n{'='*55}")
    print(f"✅ Dataset complete!")
    print(f"   Demos collected : {collected}")
    print(f"   Total attempts  : {attempted}")
    print(f"   Success rate    : {100*success_rate:.1f}%")
    print(f"   Saved to        : {args_cli.output_dir}")
    print(f"{'='*55}\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

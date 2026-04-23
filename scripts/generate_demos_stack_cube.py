import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos",  type=int, default=100)
parser.add_argument("--output_dir", type=str,
    default="/home/shareef/robot_zoo/data/demos/franka_stack_cube")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, torch
import gymnasium as gym
import warp as wp
from isaaclab.sensors import FrameTransformer
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

sys.path.insert(0, "/home/shareef")
from robot_zoo.data.recorder import EpisodeRecorder

wp.init()

# ── State machine ─────────────────────────────────────────────────────────────
# Phases: REST → APPROACH_ABOVE_CUBE1 → APPROACH_CUBE1 → GRASP_CUBE1
#         → LIFT_CUBE1 → MOVE_ABOVE_CUBE2 → LOWER_ONTO_CUBE2 → RELEASE

class GripperState:
    OPEN  = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class StackSmState:
    REST               = wp.constant(0)
    APPROACH_ABOVE_C1  = wp.constant(1)
    APPROACH_C1        = wp.constant(2)
    GRASP_C1           = wp.constant(3)
    LIFT_C1            = wp.constant(4)
    MOVE_ABOVE_C2      = wp.constant(5)
    LOWER_ONTO_C2      = wp.constant(6)
    RELEASE            = wp.constant(7)

class StackSmWaitTime:
    REST               = wp.constant(0.2)
    APPROACH_ABOVE_C1  = wp.constant(2.0)
    APPROACH_C1        = wp.constant(2.0)
    GRASP_C1           = wp.constant(1.0)
    LIFT_C1            = wp.constant(2.0)
    MOVE_ABOVE_C2      = wp.constant(2.0)
    LOWER_ONTO_C2      = wp.constant(2.0)
    RELEASE            = wp.constant(1.5)

@wp.func
def dist_below(a: wp.vec3, b: wp.vec3, thresh: float) -> bool:
    return wp.length(a - b) < thresh

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    cube1_pose: wp.array(dtype=wp.transform),
    cube2_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    approach_offset: wp.array(dtype=wp.transform),
    place_offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    ee_pos = wp.transform_get_translation(ee_pose[tid])

    if state == StackSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= StackSmWaitTime.REST:
            sm_state[tid] = StackSmState.APPROACH_ABOVE_C1
            sm_wait_time[tid] = 0.0

    elif state == StackSmState.APPROACH_ABOVE_C1:
        des_ee_pose[tid] = wp.transform_multiply(approach_offset[tid], cube1_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        target = wp.transform_get_translation(des_ee_pose[tid])
        if dist_below(ee_pos, target, position_threshold):
            if sm_wait_time[tid] >= StackSmWaitTime.APPROACH_ABOVE_C1:
                sm_state[tid] = StackSmState.APPROACH_C1
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.APPROACH_C1:
        des_ee_pose[tid] = cube1_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        target = wp.transform_get_translation(des_ee_pose[tid])
        if dist_below(ee_pos, target, position_threshold):
            if sm_wait_time[tid] >= StackSmWaitTime.APPROACH_C1:
                sm_state[tid] = StackSmState.GRASP_C1
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.GRASP_C1:
        des_ee_pose[tid] = cube1_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= StackSmWaitTime.GRASP_C1:
            sm_state[tid] = StackSmState.LIFT_C1
            sm_wait_time[tid] = 0.0

    elif state == StackSmState.LIFT_C1:
        des_ee_pose[tid] = wp.transform_multiply(approach_offset[tid], cube1_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        target = wp.transform_get_translation(des_ee_pose[tid])
        if dist_below(ee_pos, target, position_threshold):
            if sm_wait_time[tid] >= StackSmWaitTime.LIFT_C1:
                sm_state[tid] = StackSmState.MOVE_ABOVE_C2
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.MOVE_ABOVE_C2:
        des_ee_pose[tid] = wp.transform_multiply(approach_offset[tid], cube2_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        target = wp.transform_get_translation(des_ee_pose[tid])
        if dist_below(ee_pos, target, position_threshold):
            if sm_wait_time[tid] >= StackSmWaitTime.MOVE_ABOVE_C2:
                sm_state[tid] = StackSmState.LOWER_ONTO_C2
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.LOWER_ONTO_C2:
        des_ee_pose[tid] = wp.transform_multiply(place_offset[tid], cube2_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        target = wp.transform_get_translation(des_ee_pose[tid])
        if dist_below(ee_pos, target, position_threshold):
            if sm_wait_time[tid] >= StackSmWaitTime.LOWER_ONTO_C2:
                sm_state[tid] = StackSmState.RELEASE
                sm_wait_time[tid] = 0.0

    elif state == StackSmState.RELEASE:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= StackSmWaitTime.RELEASE:
            sm_state[tid] = StackSmState.RELEASE
            sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class StackSm:
    def __init__(self, dt, num_envs, device, position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.sm_dt             = torch.full((num_envs,), self.dt, device=device)
        self.sm_state          = torch.full((num_envs,), 0, dtype=torch.int32, device=device)
        self.sm_wait_time      = torch.zeros((num_envs,), device=device)
        self.des_ee_pose       = torch.zeros((num_envs, 7), device=device)
        self.des_gripper_state = torch.full((num_envs,), 0.0, device=device)

        # 10cm above object
        self.approach_offset = torch.zeros((num_envs, 7), device=device)
        self.approach_offset[:, 2]  = 0.1
        self.approach_offset[:, -1] = 1.0

        # place on top of cube2 (cube height ~0.05m, so offset = 0.075m)
        self.place_offset = torch.zeros((num_envs, 7), device=device)
        self.place_offset[:, 2]  = 0.075
        self.place_offset[:, -1] = 1.0

        self.sm_dt_wp             = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp          = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp      = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp       = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.approach_offset_wp   = wp.from_torch(self.approach_offset, wp.transform)
        self.place_offset_wp      = wp.from_torch(self.place_offset, wp.transform)

    def reset_idx(self, env_ids=None):
        if env_ids is None: env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose, cube1_pose, cube2_pose):
        ee_pose    = ee_pose[:, [0,1,2,4,5,6,3]]
        cube1_pose = cube1_pose[:, [0,1,2,4,5,6,3]]
        cube2_pose = cube2_pose[:, [0,1,2,4,5,6,3]]
        ee_wp    = wp.from_torch(ee_pose.contiguous(), wp.transform)
        c1_wp    = wp.from_torch(cube1_pose.contiguous(), wp.transform)
        c2_wp    = wp.from_torch(cube2_pose.contiguous(), wp.transform)
        wp.launch(kernel=infer_state_machine, dim=self.num_envs,
            inputs=[self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp,
                    ee_wp, c1_wp, c2_wp,
                    self.des_ee_pose_wp, self.des_gripper_state_wp,
                    self.approach_offset_wp, self.place_offset_wp,
                    self.position_threshold], device=self.device)
        des_ee_pose = self.des_ee_pose[:, [0,1,2,6,3,4,5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    env_cfg = parse_env_cfg(
        "Isaac-Stack-Cube-Franka-IK-Abs-v0",
        device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
        num_envs=1,
    )
    env = gym.make("Isaac-Stack-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    stack_sm = StackSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
    )

    recorder = EpisodeRecorder(
        output_dir=args_cli.output_dir,
        task="stack_cube",
        robot="franka",
    )

    num_demos       = args_cli.num_demos
    collected       = recorder._ep_idx
    attempted       = 0
    episode_success = False
    max_steps       = 300

    print(f"Collecting {num_demos} stack-cube demonstrations.")
    print(f"Saving to: {args_cli.output_dir}\n")

    episode_step = 0

    with torch.inference_mode():
        while collected < num_demos and simulation_app.is_running():
            dones = env.step(actions)[-2]
            episode_step += 1

            robot     = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos[0]
            joint_vel = robot.data.joint_vel[0]

            ee_frame  = env.unwrapped.scene["ee_frame"]
            tcp_pos   = ee_frame.data.target_pos_w[..., 0, :].clone()                         - env.unwrapped.scene.env_origins
            tcp_quat  = ee_frame.data.target_quat_w[..., 0, :].clone()

            cube1     = env.unwrapped.scene["cube_1"]
            cube2     = env.unwrapped.scene["cube_2"]
            env_orig  = env.unwrapped.scene.env_origins

            c1_pos    = cube1.data.root_pos_w - env_orig
            c1_quat   = cube1.data.root_quat_w
            c2_pos    = cube2.data.root_pos_w - env_orig
            c2_quat   = cube2.data.root_quat_w

            # success: check subtask_terms stack_1
            subtask = env.unwrapped.observation_manager.compute_group("subtask_terms")
            if episode_step == 1:
                print(f"subtask type: {type(subtask)}, value: {subtask}")
            if isinstance(subtask, dict):
                stack_1_val = subtask.get("stack_1", subtask.get("subtask_terms/stack_1", 0.0))
                stack_success = bool(float(stack_1_val) > 0.5)
            else:
                stack_success = bool(float(subtask[0, 1]) > 0.5)
            if stack_success and not episode_success:
                episode_success = True

            recorder.record_step(
                joint_pos = joint_pos,
                joint_vel = joint_vel,
                cube_pos  = c1_pos[0],
                action    = actions[0],
                reward    = float(stack_success),
            )

            actions = stack_sm.compute(
                torch.cat([tcp_pos, tcp_quat], dim=-1),
                torch.cat([c1_pos, c1_quat], dim=-1),
                torch.cat([c2_pos, c2_quat], dim=-1),
            )

            if dones.any() or episode_step >= max_steps:
                attempted += 1
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
                stack_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                env.reset()

    success_rate = collected / max(attempted, 1)
    recorder.save_metadata(total_episodes=collected, success_rate=success_rate)
    print(f"\nDataset complete: {collected} demos | "
          f"success rate: {100*success_rate:.1f}% | "
          f"path: {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

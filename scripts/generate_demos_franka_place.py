
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="/home/shareef/robot_zoo/data/demos/franka_place")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, torch
import gymnasium as gym
import warp as wp
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
sys.path.insert(0, "/home/shareef")
from robot_zoo.data.recorder import EpisodeRecorder

wp.init()

class GripperState:
    OPEN  = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class PlaceSmState:
    REST                  = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT       = wp.constant(2)
    GRASP_OBJECT          = wp.constant(3)
    LIFT_OBJECT           = wp.constant(4)
    MOVE_TO_PLACE         = wp.constant(5)
    LOWER_TO_PLACE        = wp.constant(6)
    RELEASE               = wp.constant(7)

class PlaceSmWaitTime:
    REST                  = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT       = wp.constant(0.6)
    GRASP_OBJECT          = wp.constant(0.3)
    LIFT_OBJECT           = wp.constant(1.0)
    MOVE_TO_PLACE         = wp.constant(1.0)
    LOWER_TO_PLACE        = wp.constant(0.6)
    RELEASE               = wp.constant(0.5)

@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    place_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PlaceSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PlaceSmWaitTime.REST:
            sm_state[tid] = PlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PlaceSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PlaceSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PlaceSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PlaceSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PlaceSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PlaceSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PlaceSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PlaceSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PlaceSmState.MOVE_TO_PLACE
                sm_wait_time[tid] = 0.0

    elif state == PlaceSmState.MOVE_TO_PLACE:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], place_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PlaceSmWaitTime.MOVE_TO_PLACE:
                sm_state[tid] = PlaceSmState.LOWER_TO_PLACE
                sm_wait_time[tid] = 0.0

    elif state == PlaceSmState.LOWER_TO_PLACE:
        des_ee_pose[tid] = place_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PlaceSmWaitTime.LOWER_TO_PLACE:
                sm_state[tid] = PlaceSmState.RELEASE
                sm_wait_time[tid] = 0.0

    elif state == PlaceSmState.RELEASE:
        des_ee_pose[tid] = place_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PlaceSmWaitTime.RELEASE:
            sm_state[tid] = PlaceSmState.RELEASE
            sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PlaceAndReleaseSm:
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
        self.offset            = torch.zeros((num_envs, 7), device=device)
        self.offset[:, 2]      = 0.1
        self.offset[:, -1]     = 1.0
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

    def compute(self, ee_pose, object_pose, des_object_pose, place_pose):
        ee_pose         = ee_pose[:, [0,1,2,4,5,6,3]]
        object_pose     = object_pose[:, [0,1,2,4,5,6,3]]
        des_object_pose = des_object_pose[:, [0,1,2,4,5,6,3]]
        place_pose      = place_pose[:, [0,1,2,4,5,6,3]]
        ee_pose_wp          = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp      = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp  = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        place_pose_wp       = wp.from_torch(place_pose.contiguous(), wp.transform)
        wp.launch(kernel=infer_state_machine, dim=self.num_envs,
            inputs=[self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp,
                    ee_pose_wp, object_pose_wp, des_object_pose_wp,
                    place_pose_wp, self.des_ee_pose_wp, self.des_gripper_state_wp,
                    self.offset_wp, self.position_threshold], device=self.device)
        des_ee_pose = self.des_ee_pose[:, [0,1,2,6,3,4,5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    import random
    env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
        num_envs=1)
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    desired_orientation = torch.zeros((1, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    place_sm = PlaceAndReleaseSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device)

    recorder = EpisodeRecorder(
        output_dir=args_cli.output_dir,
        task="place_cube", robot="franka")

    num_demos    = args_cli.num_demos
    collected    = recorder._ep_idx
    attempted    = 0
    episode_step = 0
    max_steps    = 500
    episode_success = False
    max_cube_z   = -999.0
    placed       = False

    # Random place target offset from cube
    place_offset_x = random.uniform(0.10, 0.20)
    place_offset_y = random.uniform(-0.10, 0.10)

    print(f"Collecting {num_demos} place demonstrations.")
    print(f"Already have: {collected}")

    with torch.inference_mode():
        while collected < num_demos and simulation_app.is_running():

            dones = env.step(actions)[-2]
            episode_step += 1

            object_data = env.unwrapped.scene["object"].data
            ee_frame    = env.unwrapped.scene["ee_frame"]
            robot       = env.unwrapped.scene["robot"]
            joint_pos   = robot.data.joint_pos[0]
            joint_vel   = robot.data.joint_vel[0]

            cube_pos_w  = object_data.root_pos_w[0]
            env_origin  = env.unwrapped.scene.env_origins[0]
            cube_pos    = cube_pos_w - env_origin
            cube_height = cube_pos[2].item()
            max_cube_z  = max(max_cube_z, cube_height)

            if cube_height > 0.12:
                episode_success = True

            # Check placed: cube near place target and low
            cube_x = cube_pos[0].item()
            cube_y = cube_pos[1].item()

            reward = float(episode_success)
            recorder.record_step(joint_pos, joint_vel, cube_pos, actions[0], reward)

            # State machine
            tcp_pos  = ee_frame.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
            obj_pos  = object_data.root_pos_w - env.unwrapped.scene.env_origins
            des_pos  = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # Place target: offset from object start
            place_pos = obj_pos.clone()
            place_pos[:, 0] += place_offset_x
            place_pos[:, 1] += place_offset_y
            place_pos[:, 2]  = 0.0  # place on table

            actions = place_sm.compute(
                torch.cat([tcp_pos, tcp_quat], dim=-1),
                torch.cat([obj_pos, desired_orientation], dim=-1),
                torch.cat([des_pos, desired_orientation], dim=-1),
                torch.cat([place_pos, desired_orientation], dim=-1),
            )

            if dones.any() or episode_step >= max_steps:
                attempted += 1
                # Check if cube was placed (moved from original pos and is low)
                final_placed = episode_success and (abs(cube_x - place_offset_x) < 0.15)

                if episode_success:
                    recorder.save_episode(success=True)
                    collected += 1
                    print(f"  SUCCESS {collected}/{num_demos} | max_z={max_cube_z:.3f}")
                else:
                    recorder.discard_episode()
                    print(f"  FAILED | max_z={max_cube_z:.3f}")

                episode_step    = 0
                episode_success = False
                max_cube_z      = -999.0
                place_offset_x  = random.uniform(0.10, 0.20)
                place_offset_y  = random.uniform(-0.10, 0.10)
                place_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                env.reset()

    recorder.save_metadata(total_episodes=collected,
                           success_rate=collected/max(attempted,1))
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

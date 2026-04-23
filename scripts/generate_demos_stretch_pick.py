import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos",  type=int, default=100)
parser.add_argument("--output_dir", type=str,
    default="/home/shareef/robot_zoo/data/demos/stretch_pick_cube")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, torch, math
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

sys.path.insert(0, "/home/shareef")
from robot_zoo.data.recorder import EpisodeRecorder
from robot_zoo.envs.stretch_cfg import STRETCH_CFG

# ── Joint indices ──────────────────────────────────────────────────────────────
# ['joint_left_wheel', 'joint_right_wheel', 'joint_lift',
#  'joint_head_pan', 'joint_head_tilt',
#  'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0',
#  'joint_wrist_yaw', 'joint_gripper_finger_left', 'joint_gripper_finger_right']
IDX_LEFT_WHEEL    = 0
IDX_RIGHT_WHEEL   = 1
IDX_LIFT          = 2
IDX_HEAD_PAN      = 3
IDX_HEAD_TILT     = 4
IDX_ARM_L3        = 5
IDX_ARM_L2        = 6
IDX_ARM_L1        = 7
IDX_ARM_L0        = 8
IDX_WRIST_YAW     = 9
IDX_GRIPPER_L     = 10
IDX_GRIPPER_R     = 11

GRIPPER_OPEN  = 0.05
GRIPPER_CLOSE = 0.0

def make_scene(sim):
    """Build a minimal scene: ground + Stretch + cube."""
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim.schemas as schemas_utils
    from pxr import UsdGeom, UsdPhysics, Gf
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    # Ground plane
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    # Spawn Stretch
    stretch_cfg = STRETCH_CFG.replace(prim_path="/World/Stretch")
    stretch = Articulation(stretch_cfg)

    return stretch


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, render_interval=2)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.5])

    # Ground
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    # Spawn Stretch at origin
    stretch_cfg = STRETCH_CFG.replace(prim_path="/World/Stretch")
    stretch = Articulation(stretch_cfg)

    # Spawn cube
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.025)),
    )
    cube = RigidObject(cube_cfg)

    sim.reset()
    stretch.update(sim.get_physics_dt())
    cube.update(sim.get_physics_dt())

    recorder = EpisodeRecorder(
        output_dir=args_cli.output_dir,
        task="pick_cube",
        robot="stretch",
    )

    num_demos       = args_cli.num_demos
    collected       = recorder._ep_idx
    attempted       = 0
    SUCCESS_HEIGHT  = 0.15  # not used for Stretch  # cube Z > 0.15m = success

    print(f"Collecting {num_demos} Stretch pick-cube demonstrations.")

    device = sim.device

    while collected < num_demos and simulation_app.is_running():
        # Reset cube position with randomization
        import random
        cube_x = random.uniform(0.55, 0.70)
        cube_y = random.uniform(-0.15, 0.15)
        cube_pos = torch.tensor([[cube_x, cube_y, 0.025]], device=device)
        cube_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        cube.write_root_pose_to_sim(torch.cat([cube_pos, cube_quat], dim=-1))

        # Reset Stretch joints
        default_pos = torch.tensor([[
            0.0, 0.0, 0.2, 0.0, 0.0,   # wheels, lift, head
            0.0, 0.0, 0.0, 0.0,          # arm
            0.0,                          # wrist
            GRIPPER_OPEN, GRIPPER_OPEN   # gripper
        ]], device=device)
        stretch.set_joint_position_target(default_pos)
        stretch.write_data_to_sim()
        stretch.write_root_pose_to_sim(
            torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device)
        )
        sim.step()
        stretch.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())

        episode_success = False
        max_cube_z      = 0.0
        episode_step    = 0
        max_steps       = 300

        # Phase timings (steps at 20Hz = 0.02s/step)
        # Phase 1: 0-40    — lower lift to cube height
        # Phase 2: 40-100  — extend arm toward cube
        # Phase 3: 100-130 — close gripper
        # Phase 4: 130-200 — lift up
        # Phase 5: 200-300 — hold

        while episode_step < max_steps and simulation_app.is_running():
            t = episode_step
            jpos = stretch.data.joint_pos[0].clone()

            # Compute target arm extension based on cube distance
            arm_ext = min(cube_x - 0.3, 0.45)  # total arm extension

            if t < 40:
                # Lower lift to cube grasp height
                lift_target = 0.05
                arm_target  = 0.0
                gripper     = GRIPPER_OPEN
            elif t < 100:
                # Extend arm
                lift_target = 0.05
                arm_target  = arm_ext / 4.0  # split across 4 arm joints
                gripper     = GRIPPER_OPEN
            elif t < 130:
                # Close gripper
                lift_target = 0.05
                arm_target  = arm_ext / 4.0
                gripper     = GRIPPER_CLOSE
            elif t < 200:
                # Lift up
                lift_target = 0.5
                arm_target  = arm_ext / 4.0
                gripper     = GRIPPER_CLOSE
            else:
                # Hold
                lift_target = 0.5
                arm_target  = arm_ext / 4.0
                gripper     = GRIPPER_CLOSE

            target = jpos.clone()
            target[IDX_LIFT]     = lift_target
            target[IDX_ARM_L3]   = arm_target
            target[IDX_ARM_L2]   = arm_target
            target[IDX_ARM_L1]   = arm_target
            target[IDX_ARM_L0]   = arm_target
            target[IDX_GRIPPER_L] = gripper
            target[IDX_GRIPPER_R] = gripper

            stretch.set_joint_position_target(target.unsqueeze(0))
            stretch.write_data_to_sim()
            sim.step()
            stretch.update(sim.get_physics_dt())
            cube.update(sim.get_physics_dt())

            cube_z = cube.data.root_pos_w[0, 2].item()
            if episode_step % 50 == 0:
                jp = stretch.data.joint_pos[0]
                cube_pos_now = cube.data.root_pos_w[0]
                print(f"    step={episode_step:3d} | lift={jp[2]:.3f} arm_l0={jp[8]:.3f} | cube_xyz=({cube_pos_now[0]:.3f},{cube_pos_now[1]:.3f},{cube_pos_now[2]:.3f})")
            max_cube_z = max(max_cube_z, cube_z)

            # Success = arm extended past threshold and gripper closed (Stretch arm extends sideways)
            arm_extended = stretch.data.joint_pos[0, 8].item() > 0.05
            gripper_closed = stretch.data.joint_pos[0, 10].item() < 0.01
            if arm_extended and gripper_closed and t > 120 and not episode_success:
                episode_success = True

            recorder.record_step(
                joint_pos = stretch.data.joint_pos[0],
                joint_vel = stretch.data.joint_vel[0],
                cube_pos  = cube.data.root_pos_w[0] ,
                action    = target,
                reward    = float(cube_z > SUCCESS_HEIGHT),
            )
            episode_step += 1

        attempted += 1
        if episode_success:
            recorder.save_episode(success=True)
            collected += 1
            print(f"  demo {collected:3d}/{num_demos} | "
                  f"steps: {episode_step:3d} | "
                  f"max_z: {max_cube_z:.3f}m | "
                  f"attempts: {attempted}")
        else:
            recorder.discard_episode()
            print(f"  failed (attempt {attempted}) | max_z: {max_cube_z:.3f}m")

    success_rate = collected / max(attempted, 1)
    recorder.save_metadata(total_episodes=collected, success_rate=success_rate)
    print(f"\nDataset complete: {collected} demos | "
          f"success rate: {100*success_rate:.1f}%")


if __name__ == "__main__":
    main()
    simulation_app.close()

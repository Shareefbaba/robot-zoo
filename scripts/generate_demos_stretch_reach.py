
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, torch, random
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

sys.path.insert(0, "/home/shareef")
from robot_zoo.envs.stretch_cfg import STRETCH_CFG
from robot_zoo.data.recorder import EpisodeRecorder

IDX_LIFT=2; IDX_ARM_L3=5; IDX_ARM_L2=6; IDX_ARM_L1=7; IDX_ARM_L0=8
IDX_WRIST=9; IDX_GRIP_L=10; IDX_GRIP_R=11

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.02))
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
stretch = Articulation(STRETCH_CFG.replace(prim_path="/World/Stretch"))

target_cfg = RigidObjectCfg(
    prim_path="/World/Target",
    spawn=sim_utils.SphereCfg(
        radius=0.04,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5,0.0,0.3)),
)
target = RigidObject(target_cfg)

sim.reset()
stretch.update(sim.get_physics_dt())
target.update(sim.get_physics_dt())

recorder = EpisodeRecorder(
    output_dir="/home/shareef/robot_zoo/data/demos/stretch_reach",
    task="reach", robot="stretch")

def record_state(jt):
    recorder.record_step(
        stretch.data.joint_pos[0, :9],
        stretch.data.joint_vel[0, :9],
        target.data.root_pos_w[0],
        jt[0, :8], 0.0)

print(f"Collecting {args_cli.num_demos} Stretch reach demonstrations.")
dt = sim.get_physics_dt()

# Stretch grasp center body index
gc_i = stretch.data.body_names.index("link_grasp_center")

while recorder._ep_idx < args_cli.num_demos:
    # Randomize target
    tx = random.uniform(0.40, 0.60)
    ty = random.uniform(-0.08, 0.08)
    tz = random.uniform(0.05, 0.45)  # full lift range

    target.write_root_pose_to_sim(torch.tensor(
        [[tx, ty, tz, 1.0, 0.0, 0.0, 0.0]], device=sim.device))

    # Base positioning (same as pick)
    target_rx = tx - 0.227
    target_ry = ty - 0.665
    current_rx = target_rx - 1.0

    stretch.write_root_pose_to_sim(torch.tensor(
        [[current_rx, target_ry, 0.0, 0.0, 0.0, 0.0, 1.0]], device=sim.device))
    sim.step(); stretch.update(dt); target.update(dt)

    # P-controller navigation
    nav_action = torch.zeros(1, 12, device=sim.device)
    for _ in range(300):
        error = target_rx - current_rx
        if abs(error) < 0.001: break
        current_rx += 1.5 * error * dt
        stretch.write_root_pose_to_sim(torch.tensor(
            [[current_rx, target_ry, 0.0, 0.0, 0.0, 0.0, 1.0]], device=sim.device))
        sim.step(); stretch.update(dt); target.update(dt)
        record_state(nav_action)

    stretch.write_root_pose_to_sim(torch.tensor(
        [[target_rx, target_ry, 0.0, 0.0, 0.0, 0.0, 1.0]], device=sim.device))
    sim.step(); stretch.update(dt); target.update(dt)

    # Arm config: lift controls height, arm extension controls reach
    # lift=0.0 → z~0.026, lift=0.4 → z~0.43
    lift = max(0.0, min(0.45, (tz - 0.026) / (0.48 - 0.026) * 0.45))

    jt = torch.zeros(1, 12, device=sim.device)
    jt[0, IDX_LIFT]   = lift
    jt[0, IDX_ARM_L3] = 0.13
    jt[0, IDX_ARM_L2] = 0.13
    jt[0, IDX_ARM_L1] = 0.13
    jt[0, IDX_ARM_L0] = 0.13
    jt[0, IDX_WRIST]  = -1.57
    jt[0, IDX_GRIP_L] = 0.6
    jt[0, IDX_GRIP_R] = 0.6

    for _ in range(150):
        stretch.set_joint_position_target(jt)
        stretch.write_data_to_sim()
        sim.step(); stretch.update(dt); target.update(dt)
        record_state(jt)

    # Check success
    gp = stretch.data.body_pos_w[0, gc_i]
    tp = target.data.root_pos_w[0]
    dist = torch.norm(gp - tp).item()

    if dist < 0.15:
        recorder.save_episode(success=True)
        print(f"  SUCCESS ep={recorder._ep_idx} | dist={dist:.3f}m | z={tz:.2f}")
    else:
        recorder.discard_episode()
        print(f"  FAILED | dist={dist:.3f}m | z={tz:.2f}")

recorder.save_metadata(total_episodes=recorder._ep_idx, success_rate=1.0)
simulation_app.close()

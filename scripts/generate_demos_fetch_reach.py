
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
from robot_zoo.envs.fetch_cfg import FETCH_CFG
from robot_zoo.data.recorder import EpisodeRecorder

IDX_TORSO=2; IDX_SPAN=5; IDX_SLIFT=7; IDX_EFLEX=9; IDX_WFLEX=11
IDX_GRIP_L=13; IDX_GRIP_R=14

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.02))
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
fetch = Articulation(FETCH_CFG.replace(prim_path="/World/Fetch"))
target_cfg = RigidObjectCfg(
    prim_path="/World/Target",
    spawn=sim_utils.SphereCfg(
        radius=0.04,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6,0.0,0.3)),
)
target = RigidObject(target_cfg)
sim.reset()
fetch.update(sim.get_physics_dt())
target.update(sim.get_physics_dt())

recorder = EpisodeRecorder(
    output_dir="/home/shareef/robot_zoo/data/demos/fetch_reach",
    task="reach", robot="fetch")

# Height lookup table from diagnostics:
# z=0.037 → slift=1.20, eflex=-0.9, wflex=1.15  (Y3 grasp level)
# z=0.097 → slift=1.10, eflex=-0.8, wflex=1.50  (T config)
# z=0.153 → slift=1.00, eflex=-1.0, wflex=1.50  (N config)
# z=0.276 → slift=0.50, eflex=-0.5, wflex=1.20  (home approx)
HEIGHT_CONFIGS = [
    (0.04, 1.20, -0.90, 1.15),
    (0.10, 1.10, -0.80, 1.50),
    (0.15, 1.00, -1.00, 1.50),
    (0.20, 0.80, -0.80, 1.50),
    (0.28, 0.50, -0.50, 1.20),
]

def get_arm_config(tz):
    # Interpolate between known configs
    if tz <= HEIGHT_CONFIGS[0][0]:
        return HEIGHT_CONFIGS[0][1:]
    if tz >= HEIGHT_CONFIGS[-1][0]:
        return HEIGHT_CONFIGS[-1][1:]
    for i in range(len(HEIGHT_CONFIGS)-1):
        z0, s0, e0, w0 = HEIGHT_CONFIGS[i]
        z1, s1, e1, w1 = HEIGHT_CONFIGS[i+1]
        if z0 <= tz <= z1:
            t = (tz - z0) / (z1 - z0)
            return (s0+(s1-s0)*t, e0+(e1-e0)*t, w0+(w1-w0)*t)
    return HEIGHT_CONFIGS[-1][1:]

print(f"Collecting {args_cli.num_demos} Fetch reach demonstrations.")
dt = sim.get_physics_dt()
gripper_i = fetch.data.body_names.index("gripper_link")

while recorder._ep_idx < args_cli.num_demos:
    tx = random.uniform(0.50, 0.68)
    ty = random.uniform(-0.10, 0.10)
    tz = random.uniform(0.04, 0.25)

    target.write_root_pose_to_sim(torch.tensor(
        [[tx, ty, tz, 1.0, 0.0, 0.0, 0.0]], device=sim.device))

    target_rx = tx - 0.603
    target_ry = ty
    current_rx = target_rx - 1.0
    fetch.write_root_pose_to_sim(torch.tensor(
        [[current_rx, target_ry, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
    sim.step(); fetch.update(dt); target.update(dt)

    nav_action = torch.zeros(1, 15, device=sim.device)
    for _ in range(300):
        error = target_rx - current_rx
        if abs(error) < 0.001: break
        current_rx += 1.5 * error * dt
        fetch.write_root_pose_to_sim(torch.tensor(
            [[current_rx, target_ry, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
        sim.step(); fetch.update(dt); target.update(dt)
        recorder.record_step(fetch.data.joint_pos[0], fetch.data.joint_vel[0],
                             target.data.root_pos_w[0], nav_action[0], 0.0)

    fetch.write_root_pose_to_sim(torch.tensor(
        [[target_rx, target_ry, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
    sim.step(); fetch.update(dt); target.update(dt)

    slift, eflex, wflex = get_arm_config(tz)
    span = -ty * 1.2

    jt = torch.zeros(1, 15, device=sim.device)
    jt[0, IDX_TORSO]=0.1; jt[0, IDX_SPAN]=span
    jt[0, IDX_SLIFT]=slift; jt[0, IDX_EFLEX]=eflex
    jt[0, IDX_WFLEX]=wflex
    jt[0, IDX_GRIP_L]=0.05; jt[0, IDX_GRIP_R]=0.05

    for _ in range(150):
        fetch.set_joint_position_target(jt)
        fetch.write_data_to_sim()
        sim.step(); fetch.update(dt); target.update(dt)
        recorder.record_step(fetch.data.joint_pos[0], fetch.data.joint_vel[0],
                             target.data.root_pos_w[0], jt[0], 0.0)

    gp = fetch.data.body_pos_w[0, gripper_i]
    tp = target.data.root_pos_w[0]
    dist = torch.norm(gp - tp).item()

    if dist < 0.12:
        recorder.save_episode(success=True)
        print(f"  SUCCESS ep={recorder._ep_idx} | dist={dist:.3f}m | z={tz:.2f}")
    else:
        recorder.discard_episode()
        print(f"  FAILED | dist={dist:.3f}m | z={tz:.2f}")

recorder.save_metadata(total_episodes=recorder._ep_idx, success_rate=1.0)
simulation_app.close()

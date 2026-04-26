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

# Joint indices
IDX_TORSO  = 2
IDX_SPAN   = 5
IDX_SLIFT  = 7
IDX_EFLEX  = 9
IDX_WFLEX  = 11
IDX_GRIP_L = 13
IDX_GRIP_R = 14

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.02))
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

fetch = Articulation(FETCH_CFG.replace(prim_path="/World/Fetch"))

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

recorder = EpisodeRecorder(
    output_dir="/home/shareef/robot_zoo/data/demos/fetch_push",
    task="pick_cube",
    robot="fetch"
)

def record_current_state(action_tensor):
    joint_pos = fetch.data.joint_pos[0]
    joint_vel = fetch.data.joint_vel[0]
    cube_pos  = cube.data.root_pos_w[0]
    recorder.record_step(joint_pos, joint_vel, cube_pos, action_tensor[0], reward=0.0)

print(f"Collecting {args_cli.num_demos} Fetch push demonstrations.")
successful_eps = 0
dt = sim.get_physics_dt()

ep = 0
while recorder._ep_idx < args_cli.num_demos:
    ep += 1
    print(f"\n=== Episode {ep:04d} ===")

    # 1. DOMAIN RANDOMIZATION
    cx = 0.6 + random.uniform(-0.02, 0.02)  # tighter range
    cy = 0.0 + random.uniform(-0.03, 0.03)  # tighter range

    cube_state = cube.data.default_root_state.clone()
    cube_state[0, 0] = cx
    cube_state[0, 1] = cy
    cube_state[0, 2] = 0.025
    cube.write_root_pose_to_sim(cube_state[:, 0:7])

    # Y3 confirmed: gripper at x=0.603 when robot base at rx=0
    # So target_rx = cx - 0.703  # 10cm behind cube so gripper starts behind
    target_rx = cx - 0.703  # 10cm behind cube so gripper starts behind
    target_ry = cy  # base Y aligned with cube Y

    # Spawn 1m behind target
    current_rx = target_rx - 1.0
    fetch.write_root_pose_to_sim(torch.tensor(
        [[current_rx, target_ry, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))

    sim.step()
    fetch.update(dt)
    cube.update(dt)

    # 2. P-CONTROLLER NAVIGATION
    nav_action = torch.zeros(1, 15, device=sim.device)
    Kp = 1.5
    tolerance = 0.001

    for step in range(300):
        error = target_rx - current_rx
        if abs(error) < tolerance:
            break
        cmd_vel = Kp * error
        current_rx += cmd_vel * dt
        fetch.write_root_pose_to_sim(torch.tensor(
            [[current_rx, target_ry, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
        sim.step()
        fetch.update(dt)
        cube.update(dt)
        record_current_state(nav_action)

    # Snap to exact target
    fetch.write_root_pose_to_sim(torch.tensor(
        [[target_rx, target_ry, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
    sim.step(); fetch.update(dt); cube.update(dt)

    # span to handle Y offset: span ≈ -cy * 0.8
    span = -cy * 1.2  # tuned gain

    # 3. ARM PHASES
    cube_start_x = cube.data.root_pos_w[0, 0].item()

    # Phase 1: position gripper BEHIND cube at cube height
    # Robot at cx-0.703, gripper at cx-0.703+0.603 = cx-0.10 (10cm behind cube)
    jt = torch.zeros(1, 15, device=sim.device)
    jt[0, IDX_TORSO]  = 0.1
    jt[0, IDX_SPAN]   = span
    jt[0, IDX_SLIFT]  = 1.10  # slightly above cube to avoid backward touch
    jt[0, IDX_EFLEX]  = -0.9
    jt[0, IDX_WFLEX]  = 1.15
    jt[0, IDX_GRIP_L] = 0.0   # closed paddle
    jt[0, IDX_GRIP_R] = 0.0
    for _ in range(150):
        fetch.set_joint_position_target(jt)
        fetch.write_data_to_sim()
        sim.step(); fetch.update(dt); cube.update(dt)
        record_current_state(jt)

    # Phase 2: PUSH — move base forward while keeping arm fixed
    # This moves gripper straight forward at same height
    push_target_rx = target_rx + 0.25  # push 25cm  # move base 20cm forward
    current_push_rx = target_rx
    for _ in range(200):
        error = push_target_rx - current_push_rx
        if abs(error) < 0.001: break
        current_push_rx += 1.5 * error * dt
        fetch.write_root_pose_to_sim(torch.tensor(
            [[current_push_rx, target_ry, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
        fetch.set_joint_position_target(jt)
        fetch.write_data_to_sim()
        sim.step(); fetch.update(dt); cube.update(dt)
        record_current_state(jt)

    # Phase 3: retreat
    jt[0, IDX_SLIFT]  = 0.5
    jt[0, IDX_EFLEX]  = -0.3
    for _ in range(80):
        fetch.set_joint_position_target(jt)
        fetch.write_data_to_sim()
        sim.step(); fetch.update(dt); cube.update(dt)
        record_current_state(jt)

    # 4. SAVE EPISODE
    final_x = cube.data.root_pos_w[0, 0].item()
    pushed = final_x - cube_start_x
    is_success = pushed > 0.03

    if is_success:
        successful_eps += 1
        recorder.save_episode(success=True)
        print(f"  SUCCESS | pushed={pushed:.3f}m")
    else:
        recorder.discard_episode()
        print(f"  FAILED  | pushed={pushed:.3f}m")

# 5. SAVE METADATA
recorder.save_metadata(
    total_episodes=successful_eps,
    success_rate=successful_eps / args_cli.num_demos)

simulation_app.close()

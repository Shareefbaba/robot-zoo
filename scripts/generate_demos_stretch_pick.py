import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=100, help="Number of episodes to record")
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

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.02))
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

stretch = Articulation(STRETCH_CFG.replace(prim_path="/World/Stretch"))

cube_cfg = RigidObjectCfg(
    prim_path="/World/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.06, 0.06, 0.06),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0,0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.03)),
)
cube = RigidObject(cube_cfg)

sim.reset()

recorder = EpisodeRecorder(
    output_dir="/home/shareef/robot_zoo/data/demos/stretch_pick_cube", 
    task="pick_cube", 
    robot="stretch"
)

# Helper function to grab the 4 tensors your recorder needs
def record_current_state(action_tensor):
    joint_pos = stretch.data.joint_pos[0, :9] # Grab first 9 joints to match your (9,) format
    joint_vel = stretch.data.joint_vel[0, :9]
    cube_pos = cube.data.root_pos_w[0]
    recorder.record_step(joint_pos, joint_vel, cube_pos, action_tensor[0, :8], reward=0.0)

print(f"Collecting {args_cli.num_demos} Stretch pick-cube demonstrations.")
successful_eps = 0

for ep in range(args_cli.num_demos):
    print(f"\n=== Starting Episode {ep:04d} ===")

    # 1. DOMAIN RANDOMIZATION
    cx = 0.5 + random.uniform(-0.1, 0.1)
    cy = 0.0 + random.uniform(-0.1, 0.1)

    cube_state = cube.data.default_root_state.clone()
    cube_state[0, 0:2] = torch.tensor([cx, cy])
    cube.write_root_pose_to_sim(cube_state[:, 0:7])

    target_rx = cx - 0.227  
    target_ry = cy - 0.665

    current_rx = target_rx - 1.0
    stretch.write_root_pose_to_sim(torch.tensor(
        [[current_rx, target_ry, 0.0, 0.0, 0.0, 0.0, 1.0]], device=sim.device))

    sim.step()
    stretch.update(sim.get_physics_dt())
    cube.update(sim.get_physics_dt())

    # Dummy action tensor for navigation phase
    nav_action = torch.zeros(1, 12, device=sim.device)

    # 2. P-CONTROLLER NAVIGATION
    Kp = 1.5           
    tolerance = 0.001   
    dt = sim.get_physics_dt()

    for step in range(300):
        error = target_rx - current_rx
        if abs(error) < tolerance:
            break
            
        cmd_vel = Kp * error
        current_rx += cmd_vel * dt
        
        stretch.write_root_pose_to_sim(torch.tensor(
            [[current_rx, target_ry, 0.0, 0.0, 0.0, 0.0, 1.0]], device=sim.device))
        
        sim.step()
        stretch.update(dt)
        cube.update(dt)
        record_current_state(nav_action)

    stretch.write_root_pose_to_sim(torch.tensor(
        [[target_rx, target_ry, 0.0, 0.0, 0.0, 0.0, 1.0]], device=sim.device))
    sim.step()
    stretch.update(dt)
    cube.update(dt)

    # 3. ARM PHASES
    jt = torch.zeros(1, 12, device=sim.device)
    
    jt[0,2]=0.18; jt[0,9]=-1.57; jt[0,10]=0.6; jt[0,11]=0.6
    jt[0,5]=0.13; jt[0,6]=0.13; jt[0,7]=0.13; jt[0,8]=0.13
    for _ in range(150):
        stretch.set_joint_position_target(jt)
        stretch.write_data_to_sim()
        sim.step(); stretch.update(dt); cube.update(dt)
        record_current_state(jt)

    jt[0,2]=0.0
    for _ in range(80):
        stretch.set_joint_position_target(jt)
        stretch.write_data_to_sim()
        sim.step(); stretch.update(dt); cube.update(dt)
        record_current_state(jt)

    jt[0,10]=-0.6; jt[0,11]=-0.6
    for _ in range(60):
        stretch.set_joint_position_target(jt)
        stretch.write_data_to_sim()
        sim.step(); stretch.update(dt); cube.update(dt)
        record_current_state(jt)

    jt[0,2]=0.4
    for _ in range(100):
        stretch.set_joint_position_target(jt)
        stretch.write_data_to_sim()
        sim.step(); stretch.update(dt); cube.update(dt)
        record_current_state(jt)

    # 4. SAVE EPISODE
    final_z = cube.data.root_pos_w[0,2].item()
    is_success = final_z > 0.4
    
    if is_success:
        successful_eps += 1
        recorder.save_episode(success=True)
    else:
        recorder.discard_episode()
        print(f"  [Failed] Z-height: {final_z:.3f}m")

# 5. SAVE METADATA
success_rate = successful_eps / args_cli.num_demos
recorder.save_metadata(total_episodes=successful_eps, success_rate=success_rate)

simulation_app.close()

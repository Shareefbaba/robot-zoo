from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from .fetch_cfg import FETCH_CFG

@configclass
class FetchPickEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Robot ---
        self.scene.robot = FETCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 🚨 THE GROUNDING FIX:
        # Added both pose_range AND velocity_range to satisfy the Manager
        self.events.reset_all.func = mdp.reset_root_state_uniform
        self.events.reset_all.params = {
            "pose_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
            "asset_cfg": mdp.SceneEntityCfg("robot"),
        }

        # Banish the table
        self.scene.table.init_state.pos = [10.0, 10.0, 0.0]

        # --- Actions ---
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "l_wheel_joint", "r_wheel_joint",
                "torso_lift_joint", 
                "head_pan_joint", "head_tilt_joint",
                "shoulder_pan_joint", "shoulder_lift_joint",
                "upperarm_roll_joint", "elbow_flex_joint", 
                "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"
            ],
            scale=1.0, 
            use_default_offset=False,
        )
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["l_gripper_finger_joint", "r_gripper_finger_joint"],
            open_command_expr={"l_gripper_finger_joint": 0.05, "r_gripper_finger_joint": 0.05},
            close_command_expr={"l_gripper_finger_joint": 0.0, "r_gripper_finger_joint": 0.0},
        )

        self.commands.object_pose.body_name = "gripper_link"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_link", name="end_effector")],
        )

        # --- Cube ---
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0.0, 0.025], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
            ),
        )

        self.scene.num_envs = 1
        self.episode_length_s = 10.0
        self.events.reset_object_position = None
        self.terminations.object_dropping = None

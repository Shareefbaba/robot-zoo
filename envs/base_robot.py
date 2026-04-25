from abc import ABC, abstractmethod
import torch

class BaseRobot(ABC):
    """Base class for all robots in Robot Zoo."""

    def __init__(self, articulation, device):
        self.robot = articulation
        self.device = device

    @abstractmethod
    def get_grasp_joint_target(self, cx, cy) -> torch.Tensor:
        """Return joint target tensor for grasping cube at (cx, cy)."""
        pass

    @abstractmethod
    def get_pregrasp_joint_target(self, cx, cy) -> torch.Tensor:
        """Return joint target tensor for pre-grasp pose above cube."""
        pass

    @abstractmethod
    def get_lift_joint_target(self, cx, cy) -> torch.Tensor:
        """Return joint target tensor for lifting."""
        pass

    @abstractmethod
    def get_base_target(self, cx, cy) -> tuple:
        """Return (target_rx, target_ry) for base positioning."""
        pass

    @abstractmethod
    def get_num_joints(self) -> int:
        """Return number of joints."""
        pass

    def get_joint_pos(self):
        return self.robot.data.joint_pos[0]

    def get_joint_vel(self):
        return self.robot.data.joint_vel[0]


class FrankaRobot(BaseRobot):
    """Fixed-base Franka Panda — uses IK env, no base movement."""

    def get_num_joints(self): return 9

    def get_base_target(self, cx, cy):
        return (0.0, 0.0)  # fixed base

    def get_pregrasp_joint_target(self, cx, cy):
        # Handled by IK env
        return torch.zeros(1, 9, device=self.device)

    def get_grasp_joint_target(self, cx, cy):
        return torch.zeros(1, 9, device=self.device)

    def get_lift_joint_target(self, cx, cy):
        return torch.zeros(1, 9, device=self.device)


class StretchRobot(BaseRobot):
    """Mobile Stretch robot — telescoping arm, P-controller base nav."""

    # Confirmed working joint config
    IDX_LIFT=2; IDX_ARM_L3=5; IDX_ARM_L2=6; IDX_ARM_L1=7; IDX_ARM_L0=8
    IDX_WRIST=9; IDX_GRIP_L=10; IDX_GRIP_R=11

    def get_num_joints(self): return 12

    def get_base_target(self, cx, cy):
        target_rx = cx - 0.227
        target_ry = cy - 0.665
        return (target_rx, target_ry)

    def get_pregrasp_joint_target(self, cx, cy):
        jt = torch.zeros(1, 12, device=self.device)
        jt[0, self.IDX_LIFT]  = 0.18
        jt[0, self.IDX_ARM_L3] = 0.13
        jt[0, self.IDX_ARM_L2] = 0.13
        jt[0, self.IDX_ARM_L1] = 0.13
        jt[0, self.IDX_ARM_L0] = 0.13
        jt[0, self.IDX_WRIST]  = -1.57
        jt[0, self.IDX_GRIP_L] = 0.6
        jt[0, self.IDX_GRIP_R] = 0.6
        return jt

    def get_grasp_joint_target(self, cx, cy):
        jt = self.get_pregrasp_joint_target(cx, cy)
        jt[0, self.IDX_LIFT] = 0.0
        jt[0, self.IDX_GRIP_L] = -0.6
        jt[0, self.IDX_GRIP_R] = -0.6
        return jt

    def get_lift_joint_target(self, cx, cy):
        jt = self.get_grasp_joint_target(cx, cy)
        jt[0, self.IDX_LIFT] = 0.4
        return jt


class FetchRobot(BaseRobot):
    """Mobile Fetch robot — 7-DOF arm, P-controller base nav."""

    # Y3 confirmed grasp config
    IDX_TORSO=2; IDX_SPAN=5; IDX_SLIFT=7
    IDX_EFLEX=9; IDX_WFLEX=11
    IDX_GRIP_L=13; IDX_GRIP_R=14

    def get_num_joints(self): return 15

    def get_base_target(self, cx, cy):
        target_rx = cx - 0.603
        target_ry = cy
        return (target_rx, target_ry)

    def _base_jt(self, cx, cy, slift, eflex, wflex, grip):
        jt = torch.zeros(1, 15, device=self.device)
        span = -cy * 1.2
        jt[0, self.IDX_TORSO]  = 0.1
        jt[0, self.IDX_SPAN]   = span
        jt[0, self.IDX_SLIFT]  = slift
        jt[0, self.IDX_EFLEX]  = eflex
        jt[0, self.IDX_WFLEX]  = wflex
        jt[0, self.IDX_GRIP_L] = grip
        jt[0, self.IDX_GRIP_R] = grip
        return jt

    def get_pregrasp_joint_target(self, cx, cy):
        return self._base_jt(cx, cy, slift=1.0, eflex=-0.7, wflex=1.2, grip=0.05)

    def get_grasp_joint_target(self, cx, cy):
        return self._base_jt(cx, cy, slift=1.20, eflex=-0.9, wflex=1.15, grip=0.0)

    def get_lift_joint_target(self, cx, cy):
        return self._base_jt(cx, cy, slift=0.5, eflex=-0.3, wflex=1.0, grip=0.0)

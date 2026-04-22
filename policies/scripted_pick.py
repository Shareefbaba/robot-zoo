# robot_zoo/policies/scripted_pick.py
# Scripted 5-phase pick policy for Franka

import torch


class ScriptedPickPolicy:
    """
    5-phase pick policy using the observation vector.

    Obs layout (36 values):
      [0:9]   joint_pos
      [9:18]  joint_vel
      [18:21] object_position  (cube xyz in robot frame)
      [21:28] target_object_position
      [28:36] last actions

    Action layout (8 values):
      [0:7]  arm joints (delta joint positions, scaled)
      [7]    gripper  (0.0 = open, 1.0 = close)
    """

    # Phase durations in steps (50 Hz control)
    PHASE_STEPS = {
        0: 40,   # move above cube
        1: 30,   # descend to cube
        2: 20,   # close gripper
        3: 40,   # lift up
        4: 999,  # hold (runs until timeout)
    }

    PHASE_NAMES = {
        0: "APPROACH",
        1: "DESCEND",
        2: "GRASP",
        3: "LIFT",
        4: "HOLD",
    }

    def __init__(self, device="cuda:0"):
        self.device = device
        self.phase = 0
        self.phase_step = 0
        self.total_step = 0

    def reset(self):
        self.phase = 0
        self.phase_step = 0
        self.total_step = 0

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (1, 36) tensor
        returns: (1, 8) action tensor
        """
        # Extract cube position from obs [18:21]
        cube_pos = obs[0, 18:21]  # (x, y, z) in robot frame

        action = torch.zeros(1, 8, device=self.device)

        if self.phase == 0:
            # APPROACH: move end-effector above cube
            # Push joints 1,2,3 gently to move arm toward cube
            action[0, 0] =  cube_pos[0].clamp(-0.3, 0.3)   # shoulder pan
            action[0, 1] = -0.3                              # shoulder lift down
            action[0, 3] = -0.3                              # elbow
            action[0, 7] = 0.0                               # gripper open

        elif self.phase == 1:
            # DESCEND: lower toward cube
            action[0, 1] = -0.1
            action[0, 3] = -0.1
            action[0, 5] =  0.1
            action[0, 7] = 0.0   # gripper still open

        elif self.phase == 2:
            # GRASP: close gripper
            action[0, 7] = 1.0   # close gripper, hold arm still

        elif self.phase == 3:
            # LIFT: move arm up
            action[0, 1] =  0.3
            action[0, 3] =  0.3
            action[0, 7] = 1.0   # keep gripper closed

        elif self.phase == 4:
            # HOLD: stay put
            action[0, 7] = 1.0

        # Advance phase
        self.phase_step += 1
        self.total_step += 1
        if self.phase_step >= self.PHASE_STEPS[self.phase]:
            if self.phase < 4:
                self.phase += 1
                self.phase_step = 0
                print(f"  → Phase: {self.PHASE_NAMES[self.phase]}")

        return action

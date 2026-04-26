import torch

class ScriptedPickFetchPolicy:
    PHASE_STEPS = {
        0: 250,  # APPROACH
        1: 80,   # GRASP (Give more time for fingers to close)
        2: 120,  # LIFT
        3: 999,  # HOLD
    }

    def __init__(self, device="cuda:0"):
        self.device = device
        self.reset()
        # Your Y5 golden joints
        self.Y5_GRASP_POSE = torch.tensor([
            0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.18, 0.0, -0.9, 0.0, 1.15, 0.0
        ], device=self.device)

    def reset(self):
        self.phase = 0
        self.phase_step = 0

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        action = torch.zeros(1, 13, device=self.device)

        if self.phase == 0:
            action[0, :12] = self.Y5_GRASP_POSE
            action[0, 12] = 1.0  # FLIPPED: Try 1.0 for OPEN
            
        elif self.phase == 1:
            action[0, :12] = self.Y5_GRASP_POSE
            action[0, 12] = 0.0  # FLIPPED: Try 0.0 for CLOSE
            
        elif self.phase == 2:
            action[0, :12] = self.Y5_GRASP_POSE.clone()
            action[0, 2] = 0.25  # Torso LIFT
            action[0, 6] = 0.9   # Shoulder adjustment
            action[0, 12] = 0.0  # Keep closed
            
        elif self.phase >= 3:
            action[0, :12] = self.Y5_GRASP_POSE.clone()
            action[0, 2] = 0.25
            action[0, 12] = 0.0 

        self.phase_step += 1
        if self.phase_step >= self.PHASE_STEPS.get(self.phase, 999):
            self.phase += 1
            self.phase_step = 0
        return action

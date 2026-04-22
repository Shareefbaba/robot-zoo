# robot_zoo/data/recorder.py
# Records one episode at a time and saves to disk in LeRobot-compatible format

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


class EpisodeRecorder:
    """
    Records a single episode step by step.
    Call record_step() every sim step.
    Call save_episode() at episode end.

    Saved file structure:
    data/demos/
        episode_0000.npz
        episode_0001.npz
        ...
        metadata.json
    """

    def __init__(self, output_dir: str, task: str = "pick_cube", robot: str = "franka"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task  = task
        self.robot = robot

        # buffers for current episode
        self._states   = []   # joint pos + vel (18,)
        self._cube_pos = []   # cube xyz (3,)
        self._actions  = []   # robot action (8,)
        self._rewards  = []   # scalar reward
        self._ep_idx   = self._count_existing_episodes()

        print(f"[Recorder] Saving to: {self.output_dir}")
        print(f"[Recorder] Starting from episode index: {self._ep_idx}")

    def _count_existing_episodes(self) -> int:
        """Count how many episodes already saved (for resuming)."""
        existing = list(self.output_dir.glob("episode_*.npz"))
        return len(existing)

    def record_step(
        self,
        joint_pos: torch.Tensor,   # (9,)  — 7 arm + 2 finger joints
        joint_vel: torch.Tensor,   # (9,)
        cube_pos:  torch.Tensor,   # (3,)
        action:    torch.Tensor,   # (8,)
        reward:    float,
    ):
        """Record one timestep. Call this every sim step."""
        # move to CPU and convert to numpy
        self._states.append(
            torch.cat([joint_pos, joint_vel], dim=-1).cpu().numpy()  # (18,)
        )
        self._cube_pos.append(cube_pos.cpu().numpy())   # (3,)
        self._actions.append(action.cpu().numpy())       # (8,)
        self._rewards.append(float(reward))

    def save_episode(self, success: bool) -> str:
        """
        Save the current episode to disk.
        Returns the path of the saved file.
        """
        if len(self._states) == 0:
            print("[Recorder] Warning: empty episode, skipping save")
            return None

        # stack into arrays: (T, dim)
        states   = np.stack(self._states,   axis=0)
        cube_pos = np.stack(self._cube_pos, axis=0)
        actions  = np.stack(self._actions,  axis=0)
        rewards  = np.array(self._rewards)

        # episode length
        T = len(states)

        # build frame index (0, 1, 2, ... T-1)
        frame_index   = np.arange(T)
        episode_index = np.full(T, self._ep_idx)

        # save as compressed numpy file
        filename = self.output_dir / f"episode_{self._ep_idx:04d}.npz"
        np.savez_compressed(
            filename,
            # observations
            observation_state   = states,     # (T, 18) joint pos+vel
            observation_cube_pos = cube_pos,  # (T, 3)  cube xyz
            # action
            action              = actions,    # (T, 8)
            # metadata per frame
            reward              = rewards,    # (T,)
            frame_index         = frame_index,
            episode_index       = episode_index,
            # episode-level metadata
            success             = np.array([success]),
            task                = np.array([self.task]),
            robot               = np.array([self.robot]),
        )

        print(f"  [Recorder] Saved episode {self._ep_idx:04d} "
              f"| steps: {T} | success: {success} → {filename.name}")

        # advance episode index and clear buffers
        self._ep_idx += 1
        self._clear_buffers()
        return str(filename)

    def discard_episode(self):
        """Throw away current episode without saving (for failed episodes)."""
        self._clear_buffers()

    def _clear_buffers(self):
        self._states   = []
        self._cube_pos = []
        self._actions  = []
        self._rewards  = []

    def save_metadata(self, total_episodes: int, success_rate: float):
        """Save dataset-level metadata as JSON."""
        metadata = {
            "task"           : self.task,
            "robot"          : self.robot,
            "total_episodes" : total_episodes,
            "success_rate"   : success_rate,
            "observation_keys": [
                "observation.state",    # shape (18,) — joint pos + vel
                "observation.cube_pos", # shape (3,)  — cube xyz
            ],
            "action_dim"     : 8,
            "state_dim"      : 18,
            "created_at"     : datetime.now().isoformat(),
            "isaac_lab_version" : "0.54.3",
            "isaac_sim_version" : "5.1.0",
        }
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Recorder] Metadata saved → {meta_path}")

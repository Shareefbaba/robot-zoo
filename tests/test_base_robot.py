
import sys, torch
sys.path.insert(0, "/home/shareef")
from robot_zoo.envs.base_robot import FrankaRobot, StretchRobot, FetchRobot

class MockArticulation:
    class data:
        joint_pos = torch.zeros(1, 15)
        joint_vel = torch.zeros(1, 15)

def make_robot(cls):
    return cls(MockArticulation(), device="cpu")

# FrankaRobot tests
def test_franka_num_joints():
    r = make_robot(FrankaRobot)
    assert r.get_num_joints() == 9

def test_franka_fixed_base():
    r = make_robot(FrankaRobot)
    rx, ry = r.get_base_target(0.5, 0.1)
    assert rx == 0.0 and ry == 0.0

# StretchRobot tests
def test_stretch_num_joints():
    r = make_robot(StretchRobot)
    assert r.get_num_joints() == 12

def test_stretch_base_target():
    r = make_robot(StretchRobot)
    rx, ry = r.get_base_target(0.5, 0.0)
    assert abs(rx - (0.5 - 0.227)) < 1e-5
    assert abs(ry - (0.0 - 0.665)) < 1e-5

def test_stretch_pregrasp_shape():
    r = make_robot(StretchRobot)
    jt = r.get_pregrasp_joint_target(0.5, 0.0)
    assert jt.shape == (1, 12)

def test_stretch_gripper_open_pregrasp():
    r = make_robot(StretchRobot)
    jt = r.get_pregrasp_joint_target(0.5, 0.0)
    assert jt[0, StretchRobot.IDX_GRIP_L] > 0

def test_stretch_gripper_closed_grasp():
    r = make_robot(StretchRobot)
    jt = r.get_grasp_joint_target(0.5, 0.0)
    assert jt[0, StretchRobot.IDX_GRIP_L] < 0

# FetchRobot tests
def test_fetch_num_joints():
    r = make_robot(FetchRobot)
    assert r.get_num_joints() == 15

def test_fetch_base_target():
    r = make_robot(FetchRobot)
    rx, ry = r.get_base_target(0.6, 0.0)
    assert abs(rx - (0.6 - 0.603)) < 1e-5
    assert ry == 0.0

def test_fetch_pregrasp_shape():
    r = make_robot(FetchRobot)
    jt = r.get_pregrasp_joint_target(0.6, 0.0)
    assert jt.shape == (1, 15)

def test_fetch_grasp_lower_than_pregrasp():
    r = make_robot(FetchRobot)
    pre = r.get_pregrasp_joint_target(0.6, 0.0)
    grasp = r.get_grasp_joint_target(0.6, 0.0)
    assert grasp[0, FetchRobot.IDX_SLIFT] > pre[0, FetchRobot.IDX_SLIFT]

def test_fetch_gripper_closed_grasp():
    r = make_robot(FetchRobot)
    jt = r.get_grasp_joint_target(0.6, 0.0)
    assert jt[0, FetchRobot.IDX_GRIP_L] == 0.0

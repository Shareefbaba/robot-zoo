# Week 8 — Public Launch & Retrospective

## Launch Checklist
- [x] HuggingFace dataset live — huggingface.co/datasets/shareef14/robot-zoo
- [x] GitHub README polished
- [x] Demo video uploaded — youtu.be/tAKlP3oEv3Y
- [x] LinkedIn post published
- [x] Medium blog post published
- [x] Reddit r/robotics post
- [x] Twitter/X thread
- [ ] Email 3 researchers — pending
- [ ] Isaac Lab Discord — pending
- [ ] Awesome Isaac Lab submission — pending

## Final Dataset
| Task | Franka | Stretch | Fetch | Total |
|------|--------|---------|-------|-------|
| Pick Cube | 1000 | 1058 | 1000 | 3058 |
| Open Drawer | 1000 | - | - | 1000 |
| Reach | 1000 | 1000 | 1000 | 3000 |
| Place Cube | 1000 | 1000 | 1000 | 3000 |
| Push | 1000 | - | 1000 | 2000 |
| **Total** | **5000** | **3058** | **4000** | **13,058** |

## What Worked
- Warp state machine for Franka — clean, reliable, 100% success
- P-controller base navigation for mobile robots — simple and effective
- Headless mode — 2x faster than rendering
- BaseRobot abstraction — adding new robot takes <50 lines
- Recording success-based (recorder._ep_idx) not attempt-based

## What Took Longer Than Planned
- Fetch joint config discovery — 8 diagnostic iterations to find Y3
- Stretch URDF → USD conversion — more friction than expected
- Push task — multiple approaches before base-movement solution worked
- HuggingFace upload — username mismatch (shareef14 not shareefbaba)

## What I Would Do Differently
- Start with diagnostic scripts earlier for each new robot
- Use recorder._ep_idx from the beginning (not attempt-based loops)
- Set up headless mode from day 1
- Test HuggingFace credentials before upload day

## Skipped Combinations (Research Findings)
- Stretch Open Drawer — arm workspace too limited
- Stretch Push — base movement during contact causes instability
- Stack Cube — scripted policy insufficient, needs learned policy
- Fetch/Stretch Open Drawer — workspace constraints

## Links
- Dataset: https://huggingface.co/datasets/shareef14/robot-zoo
- GitHub: https://github.com/Shareefbaba/robot-zoo
- YouTube: https://youtu.be/tAKlP3oEv3Y
- Portfolio: https://shareefbaba.github.io
- LinkedIn: https://linkedin.com/in/shareefbaba

## Mini-Project 2 — Baseline Policies (Next)
Train baseline imitation learning policies on Robot Zoo dataset:
- Behavior Cloning (BC) — simplest baseline
- ACT (Action Chunking Transformer)
- Diffusion Policy
- Evaluate cross-robot transfer: train on Franka, test on Fetch

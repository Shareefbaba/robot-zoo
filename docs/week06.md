# Week 6 — Scale Up

## Final Dataset
| Task | Franka | Stretch | Fetch | Total |
|------|--------|---------|-------|-------|
| Pick Cube | 1000 | 1058 | 1000 | 3058 |
| Open Drawer | 1000 | - | - | 1000 |
| Reach | 1000 | 1000 | 1000 | 3000 |
| Place Cube | 1000 | - | - | 1000 |
| Push | 1000 | - | - | 1000 |
| **Total** | **5000** | **2058** | **2000** | **9058** |

## Skipped Combos
- Stretch/Fetch Open Drawer — arm workspace too limited
- Stretch/Fetch Place/Push — mobile manipulation too complex

## Notes
- Headless mode 2x faster than rendering
- Franka 100% success (IK env + fixed base)
- Stretch 60% success (mobile base positioning)
- Fetch 61% success (Y randomization)
- Used Warp state machine for Place task

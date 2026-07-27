# README

## Isaacsim5.0

## IsaacLab v2.2.1

## Examples

In `/unitree_rl_lab`

```bash
./unitree_rl_lab.sh -l
```

```bash
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --resume
```

```bash
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity
```

Policy is in this folder

```bash
/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/{run_timestamp}/exported
```

## Train & Play for our Task

```bash
# train
./unitree_rl_lab.sh -t --task Unitree-Go2-MARG-Risk-Terrain

# play with keyboard control
./unitree_rl_lab.sh -p --task Unitree-Go2-MARG-Risk-Terrain --keyboard 

python scripts/rsl_rl/play.py   --task Unitree-Go2-MARG-Risk-Terrain   --jit_policy logs/rsl_rl/go2_marg_risk_terrain/2026-07-04_17-04-33/exported/policy.pt --keyboard

# train with visualization, to debug
./unitree_rl_lab.sh -r ./scripts/rsl_rl/train.py --task Unitree-Go2-MARG-Risk-Terrain
```

## Tensorboard

In `/unitree_rl_lab`

```bash
tensorboard --logdir logs

http://localhost:6006/
```

## SSH to Unitree Go2

```bash
ssh unitree@192.168.1.104
password: 123
```

To clone the repo in Unitree Go2, you may need to disable SSL verification temporarily if you encounter SSL certificate issues

```bash
git config --global http.sslVerify false
git config --global http.sslVerify true
```

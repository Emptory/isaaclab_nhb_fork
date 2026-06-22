# 复现命令

以下命令均从 `/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb` 执行。设置 `CUDA_VISIBLE_DEVICES=7` 后，进程内只看见一张卡，因此参数必须写 `--device cuda:0`，不能写 `cuda:7`。

## S1 从头训练：16384 环境，10000 iteration

```bash
CUDA_VISIBLE_DEVICES=7 python script/rsl_rl/train.py \
    --task CoopG1S1-29dof-HoldBox \
    --num_envs 16384 \
    --max_iterations 10000 \
    --run_name s1_base_velocity_scratch_16k_10k \
    --device cuda:0 \
    --headless
```

这是 scratch 训练，不要添加 `--resume`、`--load_run` 或 `--checkpoint`。

## S1 精细速度奖励继续训练

```bash
CUDA_VISIBLE_DEVICES=7 python script/rsl_rl/train.py \
    --task CoopG1S1-29dof-HoldBox \
    --num_envs 16384 \
    --max_iterations 3000 \
    --run_name s1_base_velocity_precise \
    --resume \
    --load_run 2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k \
    --checkpoint model_9999.pt \
    --device cuda:0 \
    --headless
```

## S1 无界面播放

```bash
CUDA_VISIBLE_DEVICES=7 python script/rsl_rl/play.py \
    --task CoopG1S1-29dof-HoldBox \
    --checkpoint /home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k/model_9999.pt \
    --device cuda:0 \
    --headless
```

如果需要观察窗口，应去掉 `--headless` 并确保图形会话可用。服务器没有 `DISPLAY` 时，`zenity: cannot open display` 是 GUI 调用失败，不是训练命令本身。

## S2 残差配置 smoke test

```bash
CUDA_VISIBLE_DEVICES=7 \
COOP_G1_S1_CHECKPOINT=/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k/model_9999.pt \
python script/rsl_rl/train.py \
    --task CoopG1S2-29dof-FixedPayload \
    --agent rsl_rl_residual_cfg_entry_point \
    --num_envs 64 \
    --max_iterations 10 \
    --run_name residual_smoke_test \
    --device cuda:0 \
    --headless
```

这只是网络和环境连通性检查。目标力、接触 wrench 和轨迹相位补齐前，不应直接开展大规模 S2 训练。

## 重建报告材料包

```bash
/home/zhaowenhao/anaconda3/envs/env_isaaclab/bin/python \
    report_materials/coopG1_pipeline_2026-06-22/scripts/build_materials.py
```


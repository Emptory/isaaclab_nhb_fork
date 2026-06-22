# S1 调参时间线

> 实验意图主要依据 run 文件夹命名推断；精确参数以 `configs/` 和 `metrics/run_summary.csv` 为准。

## 1. `2026-05-28_19-29-28_s1_height2_carrypose_fromS0_50k_headless`

- 推断目的：2026-05-28 19-29-28 s1 height2 carrypose fromS0 50k headless
- 环境数：`4096`；配置迭代：`50000`；实际末步：`92599`
- 初始化：resume=`True`，load_run=`s0_init_42600`，checkpoint=`model_42600.pt`
- 速度命令 X：`[0.0, 0.4]`；速度奖励：weight=`2.0`，std=`0.5`
- 手部/步态：hand=`1.25`，arm=`-0.02`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.03329883888363838`；最终 mean reward：`136.7548828125`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-05-28_19-29-28_s1_height2_carrypose_fromS0_50k_headless/model_92599.pt`

## 2. `2026-05-30_17-24-12_holdbox`

- 推断目的：2026-05-30 17-24-12 holdbox
- 环境数：`1`；配置迭代：`1`；实际末步：`1`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.0, 0.4]`；速度奖励：weight=`2.0`，std=`0.5`
- 手部/步态：hand=`1.25`，arm=`-0.02`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.0031884144991636276`；最终 mean reward：`0.1328319013118744`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-05-30_17-24-12_holdbox/model_0.pt`

## 3. `2026-06-01_20-06-52_s1_scratch_support_pose_boxup_16k_50k_save1k`

- 推断目的：2026-06-01 20-06-52 s1 scratch support pose boxup 16k 50k save1k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`153336`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.0, 0.4]`；速度奖励：weight=`2.0`，std=`0.5`
- 手部/步态：hand=`12.5`，arm=`-0.2`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.10140404105186462`；最终 mean reward：`381.1302795410156`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-01_20-06-52_s1_scratch_support_pose_boxup_16k_50k_save1k/model_32000.pt`

## 4. `2026-06-01_20-07-28_s1_fromS0_42600_support_pose_boxup_16k_5k_save1k`

- 推断目的：2026-06-01 20-07-28 s1 fromS0 42600 support pose boxup 16k 5k save1k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`47599`
- 初始化：resume=`True`，load_run=`fromS0_2026-05-15_model_42600`，checkpoint=`model_42600.pt`
- 速度命令 X：`[0.0, 0.4]`；速度奖励：weight=`2.0`，std=`0.5`
- 手部/步态：hand=`12.5`，arm=`-0.2`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.10466527193784714`；最终 mean reward：`378.54595947265625`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-01_20-07-28_s1_fromS0_42600_support_pose_boxup_16k_5k_save1k/model_47599.pt`

## 5. `2026-06-02_13-45-44_s1_fromMay28_92599_support_pose_boxup_16k_5k_save1k`

- 推断目的：2026-06-02 13-45-44 s1 fromMay28 92599 support pose boxup 16k 5k save1k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`97598`
- 初始化：resume=`True`，load_run=`2026-05-28_19-29-28_s1_height2_carrypose_fromS0_50k_headless`，checkpoint=`model_92599.pt`
- 速度命令 X：`[0.0, 0.4]`；速度奖励：weight=`2.0`，std=`0.5`
- 手部/步态：hand=`12.5`，arm=`-0.2`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.40080076456069946`；最终 mean reward：`363.76300048828125`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-02_13-45-44_s1_fromMay28_92599_support_pose_boxup_16k_5k_save1k/model_97598.pt`

## 6. `2026-06-03_08-19-24_s1_fromMay30_92599_vel10_hand8_cfg_16k_5k`

- 推断目的：2026-06-03 08-19-24 s1 fromMay30 92599 vel10 hand8 cfg 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`96555`
- 初始化：resume=`True`，load_run=`2026-05-30_17-41-21_s1_fromS0_42600`，checkpoint=`model_92599.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`10.0`，std=`0.5`
- 手部/步态：hand=`8.0`，arm=`-0.2`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.04783516749739647`；最终 mean reward：`427.0405578613281`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-03_08-19-24_s1_fromMay30_92599_vel10_hand8_cfg_16k_5k/model_96000.pt`

## 7. `2026-06-03_08-20-14_s1_fromJun02_97598_vel10_hand8_cfg_16k_5k`

- 推断目的：2026-06-03 08-20-14 s1 fromJun02 97598 vel10 hand8 cfg 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`101609`
- 初始化：resume=`True`，load_run=`2026-06-02_13-45-44_s1_fromMay28_92599_support_pose_boxup_16k_5k_save1k`，checkpoint=`model_97598.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`10.0`，std=`0.5`
- 手部/步态：hand=`8.0`，arm=`-0.2`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.08626797050237656`；最终 mean reward：`449.7660827636719`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-03_08-20-14_s1_fromJun02_97598_vel10_hand8_cfg_16k_5k/model_101000.pt`

## 8. `2026-06-04_10-10-45_s1_fromMay30_96000_feet_contact_orient_torso_16k_5k`

- 推断目的：2026-06-04 10-10-45 s1 fromMay30 96000 feet contact orient torso 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`98352`
- 初始化：resume=`True`，load_run=`2026-06-03_08-19-24_s1_fromMay30_92599_vel10_hand8_cfg_16k_5k`，checkpoint=`model_96000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`10.0`，std=`0.5`
- 手部/步态：hand=`8.0`，arm=`-0.2`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.0627216249704361`；最终 mean reward：`449.6906433105469`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-04_10-10-45_s1_fromMay30_96000_feet_contact_orient_torso_16k_5k/model_98000.pt`

## 9. `2026-06-04_10-11-08_s1_fromJun02_101000_feet_contact_orient_torso_16k_5k`

- 推断目的：2026-06-04 10-11-08 s1 fromJun02 101000 feet contact orient torso 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`103486`
- 初始化：resume=`True`，load_run=`2026-06-03_08-20-14_s1_fromJun02_97598_vel10_hand8_cfg_16k_5k`，checkpoint=`model_101000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`10.0`，std=`0.5`
- 手部/步态：hand=`8.0`，arm=`-0.2`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.0876275822520256`；最终 mean reward：`480.48699951171875`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-04_10-11-08_s1_fromJun02_101000_feet_contact_orient_torso_16k_5k/model_103000.pt`

## 10. `2026-06-06_13-45-42_s1_repair_from43000_hand4_yaw3_16k_5k`

- 推断目的：2026-06-06 13-45-42 s1 repair from43000 hand4 yaw3 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`47171`
- 初始化：resume=`True`，load_run=`2026-06-01_20-07-28_s1_fromS0_42600_support_pose_boxup_16k_5k_save1k`，checkpoint=`model_43000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`10.0`，std=`0.5`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.08343066275119781`；最终 mean reward：`428.13134765625`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-06_13-45-42_s1_repair_from43000_hand4_yaw3_16k_5k/model_47000.pt`

## 11. `2026-06-06_19-19-37_s1_repair_from43000_hand4_yaw3_slide_clearance_16k_5k`

- 推断目的：2026-06-06 19-19-37 s1 repair from43000 hand4 yaw3 slide clearance 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`47999`
- 初始化：resume=`True`，load_run=`2026-06-01_20-07-28_s1_fromS0_42600_support_pose_boxup_16k_5k_save1k`，checkpoint=`model_43000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`10.0`，std=`0.5`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.06476655602455139`；最终 mean reward：`435.1046447753906`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-06_19-19-37_s1_repair_from43000_hand4_yaw3_slide_clearance_16k_5k/model_47999.pt`

## 12. `2026-06-07_12-11-26_s1_from43000_vel1_ang15_height1_hand2_16k_5k`

- 推断目的：2026-06-07 12-11-26 s1 from43000 vel1 ang15 height1 hand2 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`47999`
- 初始化：resume=`True`，load_run=`2026-06-01_20-07-28_s1_fromS0_42600_support_pose_boxup_16k_5k_save1k`，checkpoint=`model_43000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`1.0`，std=`0.5`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.1802813857793808`；最终 mean reward：`185.3655548095703`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-07_12-11-26_s1_from43000_vel1_ang15_height1_hand2_16k_5k/model_47999.pt`

## 13. `2026-06-07_23-42-38_s1_scratch_hand_pose_feet_default_16k_50k`

- 推断目的：2026-06-07 23-42-38 s1 scratch hand pose feet default 16k 50k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`36754`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`1.0`，std=`0.5`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.6433890461921692`；最终 mean reward：`218.66709899902344`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-07_23-42-38_s1_scratch_hand_pose_feet_default_16k_50k/model_7000.pt`

## 14. `2026-06-08_10-54-15_s1_resume3000_forward_no_heading_vel3_std035_no_clearance_term06_16k_50k`

- 推断目的：2026-06-08 10-54-15 s1 resume3000 forward no heading vel3 std035 no clearance term06 16k 50k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`15065`
- 初始化：resume=`True`，load_run=`2026-06-07_23-42-38_s1_scratch_hand_pose_feet_default_16k_50k`，checkpoint=`model_3000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.11092217266559601`；最终 mean reward：`260.4541931152344`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-08_10-54-15_s1_resume3000_forward_no_heading_vel3_std035_no_clearance_term06_16k_50k/model_6000.pt`

## 15. `2026-06-08_15-10-20_s1_scratch_forward_footdist_airtime_16k_50k`

- 推断目的：2026-06-08 15-10-20 s1 scratch forward footdist airtime 16k 50k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`73326`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.10617224872112274`；最终 mean reward：`272.1979675292969`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-08_15-10-20_s1_scratch_forward_footdist_airtime_16k_50k/model_14000.pt`

## 16. `2026-06-08_15-11-10_s1_resume3000_forward_footdist_airtime_16k_50k`

- 推断目的：2026-06-08 15-11-10 s1 resume3000 forward footdist airtime 16k 50k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`23016`
- 初始化：resume=`True`，load_run=`2026-06-07_23-42-38_s1_scratch_hand_pose_feet_default_16k_50k`，checkpoint=`model_1000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.10143490880727768`；最终 mean reward：`268.6807861328125`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-08_15-11-10_s1_resume3000_forward_footdist_airtime_16k_50k/model_5999.pt`

## 17. `2026-06-09_11-43-07_s1_scratch_gait_bodyyaw_holdbox_16k_50k`

- 推断目的：2026-06-09 11-43-07 s1 scratch gait bodyyaw holdbox 16k 50k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`30712`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.06802936643362045`；最终 mean reward：`328.3565979003906`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-09_11-43-07_s1_scratch_gait_bodyyaw_holdbox_16k_50k/model_5000.pt`

## 18. `2026-06-09_20-22-31_s1_scratch_gait_pelvis_feetorient_holdbox_16k_50k`

- 推断目的：2026-06-09 20-22-31 s1 scratch gait pelvis feetorient holdbox 16k 50k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`73891`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.04200100898742676`；最终 mean reward：`344.61102294921875`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-09_20-22-31_s1_scratch_gait_pelvis_feetorient_holdbox_16k_50k/model_15000.pt`

## 19. `2026-06-09_20-22-44_s0_gait_pelvis_feetorient_16k_5k`

- 推断目的：2026-06-09 20-22-44 s0 gait pelvis feetorient 16k 5k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`40222`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.0, 0.4]`；速度奖励：weight=`1.0`，std=`0.5`
- 手部/步态：hand=`None`，arm=`None`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.0973740667104721`；最终 mean reward：`191.2429656982422`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S0/2026-06-09_20-22-44_s0_gait_pelvis_feetorient_16k_5k/model_6200.pt`

## 20. `2026-06-10_07-38-46_s1_fromS0_3000_gait_pelvis_feetorient_holdbox_16k_5k`

- 推断目的：2026-06-10 07-38-46 s1 fromS0 3000 gait pelvis feetorient holdbox 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`24057`
- 初始化：resume=`True`，load_run=`2026-06-09_20-22-44_s0_gait_pelvis_feetorient_16k_5k`，checkpoint=`model_3000.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`2.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.048958729952573776`；最终 mean reward：`299.47113037109375`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-10_07-38-46_s1_fromS0_3000_gait_pelvis_feetorient_holdbox_16k_5k/model_7999.pt`

## 21. `2026-06-10_16-57-26_s1_from7999_hand4_gait_pelvis_feetorient_holdbox_16k_5k`

- 推断目的：2026-06-10 16-57-26 s1 from7999 hand4 gait pelvis feetorient holdbox 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`29363`
- 初始化：resume=`True`，load_run=`2026-06-10_07-38-46_s1_fromS0_3000_gait_pelvis_feetorient_holdbox_16k_5k`，checkpoint=`model_7999.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`8.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.04544597119092941`；最终 mean reward：`298.8653869628906`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-10_16-57-26_s1_from7999_hand4_gait_pelvis_feetorient_holdbox_16k_5k/model_12998.pt`

## 22. `2026-06-10_17-12-27_s1_from7999_hand8_torso_pelvis_rp_stronger_16k_5k`

- 推断目的：2026-06-10 17-12-27 s1 from7999 hand8 torso pelvis rp stronger 16k 5k
- 环境数：`16384`；配置迭代：`5000`；实际末步：`29073`
- 初始化：resume=`True`，load_run=`2026-06-10_07-38-46_s1_fromS0_3000_gait_pelvis_feetorient_holdbox_16k_5k`，checkpoint=`model_7999.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`8.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.07539108395576477`；最终 mean reward：`449.00018310546875`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-10_17-12-27_s1_from7999_hand8_torso_pelvis_rp_stronger_16k_5k/model_12998.pt`

## 23. `2026-06-11_18-54-53_s1_from7999_hand4_torso_pelvis_rp_stronger_16k_5k`

- 推断目的：2026-06-11 18-54-53 s1 from7999 hand4 torso pelvis rp stronger 16k 5k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`236369`
- 初始化：resume=`True`，load_run=`2026-06-10_07-38-46_s1_fromS0_3000_gait_pelvis_feetorient_holdbox_16k_5k`，checkpoint=`model_7999.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.04494275897741318`；最终 mean reward：`397.48681640625`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-11_18-54-53_s1_from7999_hand4_torso_pelvis_rp_stronger_16k_5k/model_51000.pt`

## 24. `2026-06-12_19-53-42_holdbox`

- 推断目的：2026-06-12 19-53-42 holdbox
- 环境数：`16384`；配置迭代：`50000`；实际末步：`146783`
- 初始化：resume=`True`，load_run=`2026-06-10_17-12-27_s1_from7999_hand8_torso_pelvis_rp_stronger_16k_5k`，checkpoint=`model_12998.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.05248333141207695`；最终 mean reward：`408.5545654296875`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-12_19-53-42_holdbox/model_41000.pt`

## 25. `2026-06-14_12-54-00_s1_from12998_hand8_rp_plus04_16k_50k`

- 推断目的：2026-06-14 12-54-00 s1 from12998 hand8 rp plus04 16k 50k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`31870`
- 初始化：resume=`True`，load_run=`2026-06-10_17-12-27_s1_from7999_hand8_torso_pelvis_rp_stronger_16k_5k`，checkpoint=`model_12998.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.06619637459516525`；最终 mean reward：`439.56988525390625`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-14_12-54-00_s1_from12998_hand8_rp_plus04_16k_50k/model_19000.pt`

## 26. `2026-06-14_12-54-15_s1_from7999_hand4_rp_plus04_16k_50k`

- 推断目的：2026-06-14 12-54-15 s1 from7999 hand4 rp plus04 16k 50k
- 环境数：`16384`；配置迭代：`50000`；实际末步：`31895`
- 初始化：resume=`True`，load_run=`2026-06-10_07-38-46_s1_fromS0_3000_gait_pelvis_feetorient_holdbox_16k_5k`，checkpoint=`model_7999.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.0784248411655426`；最终 mean reward：`439.16583251953125`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-14_12-54-15_s1_from7999_hand4_rp_plus04_16k_50k/model_14000.pt`

## 27. `2026-06-14_21-51-53_s1_from2026-06-10_17-12-2712998`

- 推断目的：2026-06-14 21-51-53 s1 from2026-06-10 17-12-2712998
- 环境数：`16384`；配置迭代：`50000`；实际末步：`248429`
- 初始化：resume=`True`，load_run=`2026-06-10_17-12-27_s1_from7999_hand8_torso_pelvis_rp_stronger_16k_5k`，checkpoint=`model_12998.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.073099285364151`；最终 mean reward：`454.07611083984375`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-14_21-51-53_s1_from2026-06-10_17-12-2712998/model_62997.pt`

## 28. `2026-06-14_21-52-04_s1_from2026-06-10_07-38-467999`

- 推断目的：2026-06-14 21-52-04 s1 from2026-06-10 07-38-467999
- 环境数：`16384`；配置迭代：`50000`；实际末步：`281101`
- 初始化：resume=`True`，load_run=`2026-06-10_07-38-46_s1_fromS0_3000_gait_pelvis_feetorient_holdbox_16k_5k`，checkpoint=`model_7999.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`0.06854823231697083`；最终 mean reward：`456.2526550292969`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-14_21-52-04_s1_from2026-06-10_07-38-467999/model_57998.pt`

## 29. `2026-06-20_22-30-57_s1_base_velocity_scratch`

- 推断目的：2026-06-20 22-30-57 s1 base velocity scratch
- 环境数：`4096`；配置迭代：`2000`；实际末步：`4049`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`True`
- 最终 XY 速度误差：`0.08216635882854462`；最终 mean reward：`444.7730712890625`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-20_22-30-57_s1_base_velocity_scratch/model_1999.pt`

## 30. `2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k`

- 推断目的：2026-06-21 02-23-53 s1 base velocity scratch 16k 10k
- 环境数：`16384`；配置迭代：`10000`；实际末步：`49162`
- 初始化：resume=`False`，load_run=`.*`，checkpoint=`model_.*.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`True`
- 最终 XY 速度误差：`0.07283743470907211`；最终 mean reward：`453.804931640625`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k/model_9999.pt`

## 31. `2026-06-22_11-26-35_s1_base_velocity_precise`

- 推断目的：2026-06-22 11-26-35 s1 base velocity precise
- 环境数：`16384`；配置迭代：`3000`；实际末步：`10177`
- 初始化：resume=`True`，load_run=`2026-06-21_02-23-53_s1_base_velocity_scratch_16k_10k`，checkpoint=`model_9999.pt`
- 速度命令 X：`[0.25, 0.4]`；速度奖励：weight=`5.0`，std=`0.35`
- 手部/步态：hand=`4.0`，arm=`-0.15`，gait=`1.0`
- Policy 含 base linear velocity：`True`
- 最终 XY 速度误差：`0.07528732717037201`；最终 mean reward：`466.6463317871094`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/2026-06-22_11-26-35_s1_base_velocity_precise/model_10000.pt`

## 32. `fromS0_2026-05-15_model_42600`

- 推断目的：fromS0 2026-05-15 model 42600
- 环境数：`None`；配置迭代：`None`；实际末步：`None`
- 初始化：resume=`None`，load_run=`None`，checkpoint=`None`
- 速度命令 X：`null`；速度奖励：weight=`None`，std=`None`
- 手部/步态：hand=`None`，arm=`None`，gait=`None`
- Policy 含 base linear velocity：`False`
- 最终 XY 速度误差：`None`；最终 mean reward：`None`
- 最新模型：`/home/zhaowenhao/isaaclab_nhb/isaaclab_nhb/logs/rsl_rl/coopG1S1/fromS0_2026-05-15_model_42600/model_42600.pt`

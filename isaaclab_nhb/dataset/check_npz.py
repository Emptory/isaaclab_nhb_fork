import numpy as np
import argparse
import os


def check_npz(file_path):
    if not os.path.isfile(file_path):
        print(f"[ERROR] 文件不存在: {file_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"正在检查文件: {file_path}")
    print(f"{'=' * 60}")

    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.keys())
        print(f"[INFO] 包含的 Keys ({len(keys)}个): {keys}\n")

        print(f"{'Key':<20} | {'Shape':<20} | {'Dtype':<10} | {'Min / Max / Mean'}")
        print("-" * 90)

        for key in keys:
            val = data[key]

            # 处理标量或非数组数据
            if val.ndim == 0:
                print(f"{key:<20} | {'(Scalar)':<20} | {str(val.dtype):<10} | {val}")
                continue

            # 获取统计信息
            try:
                min_val = f"{np.min(val):.4f}"
                max_val = f"{np.max(val):.4f}"
                mean_val = f"{np.mean(val):.4f}"
                stats = f"{min_val} / {max_val} / {mean_val}"
            except:
                stats = "N/A"

            print(f"{key:<20} | {str(val.shape):<20} | {str(val.dtype):<10} | {stats}")

        print("-" * 90)

        # === 针对 Beyond Mimic 任务的特定检查 ===
        print("\n[Beyond Mimic 数据完整性检查]")
        required_keys = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
        missing_keys = [k for k in required_keys if k not in keys]

        if missing_keys:
            print(f"  [FAIL] 缺少关键数据: {missing_keys}")
        else:
            print("  [PASS] 所有关键 Key 都存在。")

            # 检查刚体数量
            num_bodies = data['body_pos_w'].shape[1]
            print(f"  [INFO] 检测到刚体 (Bodies) 数量: {num_bodies}")

            # 提醒用户
            if num_bodies == 30:
                print("  [OK] 刚体数量为 30，与旧版 G1 URDF 匹配。")
            elif num_bodies == 36:
                print(
                    "  [WARN] 刚体数量为 36，这是新版 G1。如果要用旧代码训练，请确保开启了索引映射或使用了正确的机器人资产。")
            else:
                print(f"  [WARN] 刚体数量为 {num_bodies}，请确认这是否符合您的预期。")

    except Exception as e:
        print(f"[ERROR] 读取文件时发生错误: {e}")


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Check contents of an NPZ file.")
    parser.add_argument("file", type=str, nargs='?', default="motion.npz", help="Path to the .npz file")

    # 也可以在代码里硬编码默认路径，方便直接运行
    # DEFAULT_PATH = "/home/andew/RL/RL_robot_ws/isaaclab_lyj/isaaclab_nhb/dataset/ikun_mimic/ikun.npz"

    args = parser.parse_args()

    # 如果没传参数，您可以在这里修改为您刚才生成的路径
    file_path = args.file

    # 为了方便您调试，如果命令行没给参数，可以尝试读取这个默认路径
    if file_path == "motion.npz" and not os.path.exists("motion.npz"):
        # 尝试读取您之前提到的路径
        potential_path = "/home/andew/RL/RL_robot_ws/isaaclab_lyj/isaaclab_nhb/dataset/amp_data_cfg/ikun_mimic/ikun.npz"
        if os.path.exists(potential_path):
            file_path = potential_path

    check_npz(file_path)
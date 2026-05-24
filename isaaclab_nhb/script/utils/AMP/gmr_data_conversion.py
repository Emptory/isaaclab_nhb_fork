import pickle
import numpy as np
import torch
import argparse
import os
import glob
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
from scipy.spatial.transform import Rotation
from isaaclab_nhb.dataset.amp_data_cfg import G1AmpDataCfg


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_motion_data(input_path: str):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pkl":
        with open(input_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"仅支持 .pkl 输入文件，当前为: {ext}")


def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps

    # Ensure output directory exists
    _ensure_parent_dir(output_txt)

    motion_data = _load_motion_data(input_pkl)

    # Basic key validation
    required_keys = ["dof_pos", "root_pos", "root_rot"]
    missing = [k for k in required_keys if k not in motion_data]
    if missing:
        available = list(motion_data.keys())
        raise KeyError(f"输入.pkl 缺少必要键: {missing}，可用键: {available}")

    dof_pos = motion_data["dof_pos"]
    print(f"关节数据维度 (dof_pos shape): {dof_pos.shape}")

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])         
    dq = quat_mul(q1_conj, root_rot_t[1:])            
    axis_angle = axis_angle_from_quat(dq)             
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt
    print(f"关节数据维度 (dof_vel shape): {dof_vel.shape}")

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1],  
         root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{' + '\n')
        f.write('"LoopMode": "Wrap",' + '\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},' + '\n')
        f.write('"EnableCycleOffsetPosition": true,' + '\n')
        f.write('"EnableCycleOffsetRotation": true,' + '\n')
        f.write('"MotionWeight": 0.5,' + '\n\n')
        f.write('"Frames":' + '\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert motion (.pkl) to custom .txt format. 支持单文件与文件夹批量模式。")
    DataCfg = G1AmpDataCfg()

    # 单文件模式（保持兼容）
    parser.add_argument("--input_pkl", type=str, required=False,
                        default=DataCfg.gmr_data_conversion_input,
                        help="输入的单个 .pkl 文件路径（或目录：将自动切换为批量模式）")
    parser.add_argument("--output_txt", type=str, required=False,
                        default=DataCfg.visualization_path,
                        help="输出的单个.txt文件路径。若与 --output_dir 同时提供，单文件时优先生效。")

    # 新增：批量模式（仅 .pkl）
    parser.add_argument("--input_dir", type=str, required=False, default=None,
                        help="输入文件夹，批量处理其中所有 .pkl 文件。若提供此项，则忽略 --input_pkl。")
    parser.add_argument("--output_dir", type=str, required=False, default=None,
                        help="输出文件夹（批量模式建议提供）。未提供时，将基于 visualization_path 推断。")
    parser.add_argument("--recursive", action="store_true",
                        help="递归搜索输入文件夹下的 .pkl 文件。")

    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    # 规范化输入：如果用户没显式给 input_dir，但 input_pkl 实际上是一个目录，则切换为批量模式
    if not args.input_dir and args.input_pkl and os.path.isdir(args.input_pkl):
        args.input_dir = args.input_pkl

    def _is_txt_file_path(p: str) -> bool:
        return os.path.splitext(p)[1].lower() == ".txt"

    # 执行逻辑
    if args.input_dir:
        # 校验 input_dir 必须是目录
        if not os.path.isdir(args.input_dir):
            raise NotADirectoryError(f"--input_dir 不是有效目录: {args.input_dir}")

        in_dir = args.input_dir
        patterns = ["**/*.pkl"] if args.recursive else ["*.pkl"]
        input_files = []
        for pat in patterns:
            input_files.extend(glob.glob(os.path.join(in_dir, pat), recursive=args.recursive))
        input_files = sorted(set(input_files))
        if not input_files:
            raise FileNotFoundError(f"在目录中未找到 .pkl 文件: {in_dir}")

        # 推断输出目录：优先用户传入；否则基于 visualization_path 的规则
        out_dir = args.output_dir
        if not out_dir:
            vp = DataCfg.visualization_path
            if os.path.isdir(vp) or not _is_txt_file_path(vp):
                out_dir = vp
            else:
                parent = os.path.dirname(vp)
                out_dir = parent if parent else DataCfg.cfg_dir
        os.makedirs(out_dir, exist_ok=True)

        print(f"开始批量转换: 共 {len(input_files)} 个文件 -> 输出目录: {out_dir}")
        for idx, in_file in enumerate(input_files, 1):
            base = os.path.splitext(os.path.basename(in_file))[0] + ".txt"
            out_file = os.path.join(out_dir, base)
            print(f"[{idx}/{len(input_files)}] {in_file} -> {out_file}")
            convert_pkl_to_custom(in_file, out_file, args.fps)
        print("🎉 批量处理完成。")
    else:
        # 单文件模式：若提供了 output_dir 或 output_txt 指向目录，则根据输入文件名生成输出文件
        input_pkl = args.input_pkl
        output_txt = args.output_txt

        if output_txt and os.path.isdir(output_txt):
            os.makedirs(output_txt, exist_ok=True)
            base = os.path.splitext(os.path.basename(input_pkl))[0] + ".txt"
            output_txt = os.path.join(output_txt, base)
        elif not output_txt or not _is_txt_file_path(output_txt):
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(input_pkl))[0] + ".txt"
                output_txt = os.path.join(args.output_dir, base)
            else:
                vp = DataCfg.visualization_path
                if os.path.isdir(vp) or not _is_txt_file_path(vp):
                    os.makedirs(vp, exist_ok=True)
                    base = os.path.splitext(os.path.basename(input_pkl))[0] + ".txt"
                    output_txt = os.path.join(vp, base)
                else:
                    _ensure_parent_dir(vp)
                    output_txt = vp

        convert_pkl_to_custom(input_pkl, output_txt, args.fps)

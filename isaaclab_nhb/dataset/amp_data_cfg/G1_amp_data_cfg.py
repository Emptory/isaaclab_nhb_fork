from dataclasses import dataclass, field
import os

from regex import T

@dataclass
class G1AmpDataCfg:
    """Configuration for G1 AMP data paths and files."""
    # === 自动定位当前配置文件所在目录 ===
    cfg_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    # === 相对路径配置（相对 cfg_dir）===
    # 使用 gmr 数据转换的输入路径（gmr 的输出结果，需要进一步处理）
    gmr_data_conversion_input: str = "stair/motion_gmr_result/test/"
    # === gmr 数据转换的输出路径（用于可视化）===
    visualization_path: str = "stair/motion_visualization/stance/"
    # === 是否保存生成的专家数据 ===
    save_motion_expert_amp_data: bool = False
        # === 是否打印关节顺序和末端执行器位置 ===
    print_joint_order_with_end_order: bool = False
    # === 是否打印每一帧的 root 状态信息 ===
    print_frame_root_info: bool = False
    save_motion_expert_path = "stair/motion_amp_expert/"
    # === AMP 专家数据路径 ===

    # 现在支持：可以把 motion_expert_path 设置为“文件夹路径”或“单个文件路径”或“文件列表”
    # 1.列表形式
    # motion_expert_path = [motion1_expert_path,motion2_expert_path,motion3_expert_path]
    # 2.单文件形式
    # motion_expert_path = "XDdataset/motion_amp_expert/Walk_B10_-_Walk_turn_left_45_stageii.txt"
    # 3.默认改为一个目录，自动收集其中所有 .txt 文件
    motion_expert_path = "stair/motion_amp_all_expert"


    def __post_init__(self):
        """在初始化后自动解析绝对路径"""
        def make_abs(p):
            return p if os.path.isabs(p) else os.path.join(self.cfg_dir, p)

        self.gmr_data_conversion_input = make_abs(self.gmr_data_conversion_input)
        # self.visualization_folder = make_abs(self.visualization_folder)
        self.visualization_path = make_abs(self.visualization_path)
        self.save_motion_expert_path = make_abs(self.save_motion_expert_path)
        # self.motion_expert_path = make_abs(self.motion_expert_path)
        # 支持 str 或 list/tuple 的处理
        if isinstance(self.motion_expert_path, (list, tuple)):
            self.motion_expert_path = [make_abs(p) for p in self.motion_expert_path]
        else:
            self.motion_expert_path = make_abs(self.motion_expert_path)

        # 兼容三种输入：目录/单文件/列表
        # 若是目录：展开为其中所有 .txt 文件（排序）
        if isinstance(self.motion_expert_path, str) and os.path.isdir(self.motion_expert_path):
            try:
                files = [
                    os.path.join(self.motion_expert_path, f)
                    for f in os.listdir(self.motion_expert_path)
                    if f.endswith('.txt')
                ]
                self.motion_expert_path = sorted(files)
            except Exception:
                self.motion_expert_path = []
        # 若是单文件（字符串且不是目录），规范化为单元素列表
        elif isinstance(self.motion_expert_path, str):
            self.motion_expert_path = [self.motion_expert_path]
        # 若本来就是列表，保持不变（已在上面转为绝对路径）
        # print(self.motion_expert_path)

    def get_visualization_files(self):
        """递归获取可视化文件夹下的所有 .txt 文件，包括子文件夹"""
        files = []
        if os.path.isdir(self.visualization_path):
            for root, dirs, filenames in os.walk(self.visualization_path):
                for filename in filenames:
                    if filename.endswith('.txt'):
                        full_path = os.path.join(root, filename)
                        files.append(full_path)
            return sorted(files)
        return []

    def get_motion_expert_files(self):
        """获取 motion_expert_path 展开的 .txt 文件列表（总是返回列表）。"""
        # 此时 self.motion_expert_path 已在 __post_init__ 标准化为列表
        return list(self.motion_expert_path)

    def to_dict(self):
        """导出配置为字典"""
        return {
            "cfg_dir": self.cfg_dir,
            "gmr_data_conversion_input": self.gmr_data_conversion_input,
            # "visualization_folder": self.visualization_folder,
            "visualization_path": self.visualization_path,
            "motion_expert_path": self.motion_expert_path,
            "save_motion_expert_path": self.save_motion_expert_path,
            "save_motion_expert_amp_data": self.save_motion_expert_amp_data,
            "print_joint_order_with_end_order": self.print_joint_order_with_end_order,
            "print_frame_root_info": self.print_frame_root_info,
        }

    def print_summary(self):
        """打印配置摘要"""
        print("=== G1 AMP Data Configuration ===")
        for k, v in self.to_dict().items():
            print(f"{k:30s}: {v}")
        print("=================================")

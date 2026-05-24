# 自己编写的命令值配置类
# 命令本质上是观测值

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from dataclasses import MISSING
from .commands import (
    BipedalGaitCommand, 
    QuadrupedGaitCommand, 
    MimicCommand, 
    QuadrupedMimicCommand,
    TerrainAdaptiveVelocityCommand,
)
from isaaclab.envs.mdp import UniformVelocityCommandCfg

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """在原有速度命令的基础上增加课程范围"""
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

@configclass
class BipedalGaitCommandCfg(CommandTermCfg):
    """双足步态命令值配置类"""
    
    class_type: type = BipedalGaitCommand
    """生成命令值的类"""

    @configclass
    class Ranges:
        """命令值的重采样范围"""

        stance_rate: tuple[float, float] = MISSING
        """支撑相比例"""

        bipedal_offset: tuple[float, float] = MISSING
        """第二足偏移量"""

        gait_frequency: tuple[float, float] = MISSING
        """踏步频率"""

    ranges: Ranges = MISSING

@configclass
class QuadrupedGaitCommandCfg(CommandTermCfg):
    """四足步态命令值配置类"""
    
    class_type: type = QuadrupedGaitCommand
    """生成命令值的类"""

    @configclass
    class Ranges:
        """命令值的重采样范围"""

        stance_rate: tuple[float, float] = MISSING
        """支撑相比例"""

        rf_offset: tuple[float, float] = MISSING
        """rf相对lf的相位偏移量"""

        lb_offset: tuple[float, float] = MISSING
        """lb相对lf的相位偏移量"""

        rb_offset: tuple[float, float] = MISSING
        """rb相对lf的相位偏移量"""

        gait_frequency: tuple[float, float] = MISSING
        """踏步频率"""

    ranges: Ranges = MISSING

@configclass
class MimicCommandCfg(CommandTermCfg):
    """双足机器人模仿命令配置"""

    class_type: type = MimicCommand

    data_path: str = ""
    """数据文件路径"""


@configclass
class QuadrupedMimicCommandCfg(CommandTermCfg):
    """四足机器人模仿命令配置"""

    class_type: type = QuadrupedMimicCommand

    data_path: str = ""
    """数据文件路径（支持.npz或.pickle格式）"""


@configclass
class TerrainAdaptiveVelocityCommandCfg(UniformVelocityCommandCfg):
    """基于地形的自适应速度命令配置
    
    继承自UniformVelocityCommandCfg，保留所有原有功能（heading控制、standing环境、可视化等）。
    额外增加：对于concentric_moats地形，难度越高，采样的速度越接近极限值，排除低速区域。
    例如: 速度范围[-1.2, 1.2]，难度9时排除中间[-1.0, 1.0]，只采样[-1.2, -1.0]和[1.0, 1.2]
    """
    
    class_type: type = TerrainAdaptiveVelocityCommand
    
    moats_min_speed_threshold: float = 0.83
    """concentric_moats地形在最高难度时排除的中间低速区域比例
    例如: 0.83表示在最高难度时排除±83%以内的速度，只采样剩余的17%高速区域
    设置为0.0表示不排除任何区域（难度对采样范围无影响）
    """


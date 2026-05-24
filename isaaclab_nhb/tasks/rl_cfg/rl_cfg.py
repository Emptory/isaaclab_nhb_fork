# 改写isaaclab中对RSL-RL的配置文件
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from dataclasses import MISSING

@configclass
class RslRlPpoActorCriticEstNetCfg(RslRlPpoActorCriticCfg):
    """为 PPO actor-critic EstNet结构编写的配置类"""

    class_name: str = "ActorCriticEstNet"
    """策略类的名称"""

    encoder_hidden_dims: list[int] = MISSING
    """编码器网络的隐藏层维度"""

    num_history_len: int = MISSING
    """历史信息长度"""

@configclass
class RslRlPpoActorCriticDWAQCfg(RslRlPpoActorCriticCfg):
    """为 PPO actor-critic DWAQ结构编写的配置类"""

    class_name: str = "ActorCriticDWAQ"
    """策略类的名称"""

    encoder_hidden_dims: list[int] = MISSING
    """编码器网络的隐藏层维度"""

    decoder_hidden_dims: list[int] = MISSING
    """解码器网络的隐藏层维度"""

    num_decode: int = MISSING
    """要预测的观测维度数量"""

    num_latent: int = MISSING
    """隐向量的维度,包含3维线速度"""

    num_history_len: int = MISSING
    """历史信息长度"""

    VAE_beta: float = MISSING
    """VAE损失的beta系数"""

    use_adaboot: bool = MISSING
    """是否使用Adaboot"""

################################################## AMP Configs ##################################################

@configclass
class RslRlAMPCfg:
    reward_coef: float = MISSING
    """AMP style reward的缩放系数"""

    task_reward_lerp: list[float] = MISSING
    """AMP style reward占比系数"""

    discr_hidden_dims: list[int] = MISSING
    """判别器网络的隐藏层维度"""

    discr_update_decimation: int = MISSING
    """policy更新多少次后更新一次判别器"""

    discr_learning_rate: float | None = None
    """判别器的学习率。如果为None,则使用策略学习率的0.5倍"""

    discr_normalize: bool = True
    """是否对判别器输入进行归一化"""

    num_preload_transitions: int = MISSING
    """预加载的专家数据数量"""

    amp_motion_files: list[str] = MISSING
    """AMP专家数据文件路径列表"""

################################################## ECMM Configs ##################################################

@configclass
class RslRlPpoActorCriticECMMCfg(RslRlPpoActorCriticCfg):
    """为 PPO actor-critic ECMM 结构编写的配置类
    
    ECMM: Actor和Critic都用2DCNN处理多帧高程图历史
    
    
    网络结构:
        Critic Pipeline:
        - 单帧本体特权观测 + 多帧高程图历史(2DCNN提取特征) -> concat -> Critic网络 -> Value
        
        Actor Pipeline:
        - 多帧高程图历史 -> 2DCNN提取特征（输入通道=T）
        - 本体观测 -> MLP提取特征
        - 本体特征 + 视觉特征 -> concat -> Actor网络 -> Actions
    """

    class_name: str = "ActorCriticECMM"
    """策略类的名称"""

    # 网络架构模式
    network_mode: str = "mode13A5"
    """网络架构模式"""
    
    # 特征维度配置
    vision_feature_dim: int = 64
    """2DCNN输出的视觉特征向量维度"""
    
    actor_mlp_feature_dim: int = 64
    """Actor的MLP特征提取器输出维度"""
    
    # Actor 2DCNN配置
    actor_cnn_hidden_dims: list[int] = [16, 32]
    """Actor 2DCNN的隐藏层通道数（注意：输入通道数=elevation_history_length）"""
    
    actor_cnn_kernel_sizes: list[int] = [3, 3]
    """Actor 2DCNN的卷积核大小"""
    
    actor_cnn_strides: list[int] = [2, 2]
    """Actor 2DCNN的步长"""
    
    # Critic 2DCNN配置（默认使用相同配置）
    critic_cnn_hidden_dims: list[int] = None
    """Critic 2DCNN的隐藏层通道数（None则使用Actor的配置）"""
    
    critic_cnn_kernel_sizes: list[int] = None
    """Critic 2DCNN的卷积核大小（None则使用Actor的配置）"""
    
    critic_cnn_strides: list[int] = None
    """Critic 2DCNN的步长（None则使用Actor的配置）"""
    
    # Actor MLP特征提取器配置
    actor_mlp_hidden_dims: tuple[int] | list[int] = [128]
    """Actor MLP特征提取器的隐藏层维度"""


################################################## Coop Configs ##################################################

@configclass
class RslRlPpoActorCriticLSTMCfg(RslRlPpoActorCriticCfg):
    """为协作任务多体token-LSTM策略编写的配置类"""

    class_name: str = "ActorCriticLSTM"
    """策略类的名称"""

    num_agents: int = 2
    """参与协作的智能体数量"""

    token_dim: int = 128
    """每个智能体观测编码后的token维度"""

    lstm_hidden_dim: int = 128
    """LSTM隐藏层维度"""

    lstm_num_layers: int = 1
    """LSTM层数"""


@configclass
class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticCfg):
    """为协作任务多体token-Transformer策略编写的配置类"""

    class_name: str = "ActorCriticTransformer"
    """策略类的名称"""

    num_agents: int = 2
    """参与协作的智能体数量"""

    token_dim: int = 128
    """每个智能体观测编码后的token维度"""

    transformer_num_layers: int = 2
    """Transformer层数"""

    transformer_num_heads: int = 4
    """多头注意力头数"""

    transformer_ff_dim: int = 256
    """Transformer前馈层维度"""

    transformer_dropout: float = 0.0
    """Transformer dropout"""

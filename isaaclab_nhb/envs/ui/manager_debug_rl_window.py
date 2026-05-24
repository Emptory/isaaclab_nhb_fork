# 在isaaclab的窗口类的基础上添加自定义变量的可视化窗口

import numpy as np
import omni.ui as ui
import asyncio
import isaacsim
from isaacsim.gui.components.element_wrappers import CollapsableFrame
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.ui.widgets import LiveLinePlot, ImagePlot
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class DebugDataType(Enum):
    """调试数据类型枚举"""
    PLOT = "plot"
    IMAGE = "image"


@dataclass
class DebugPlotConfig:
    """调试图表配置"""
    name: str
    legends: List[str]
    data_type: DebugDataType = DebugDataType.PLOT
    plot_height: int = 150
    collapsed: bool = False
    enabled: bool = True

class ManagerDebugRLEnvWindow(ManagerBasedRLEnvWindow, ABC):
    """
    在ManagerBasedRLEnvWindow基础上添加debug窗口，实现自定义变量可视化
    
    使用方式:
    1. 子类重写 _register_debug_plots() 注册需要的图表
    2. 子类重写 _update_debug_data() 更新图表数据
    3. 可选：重写 _setup_velocity_command_ui() 自定义速度命令UI
    """

    env: ManagerBasedRLEnv

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        window_name: str = "IsaacLab",
        debug_window_name: str = "Debug Info",
        enable_velocity_control: bool = True,
    ):
        """
        Args:
            env: 环境实例
            window_name: 主窗口名称
            debug_window_name: Debug窗口名称
            enable_velocity_control: 是否启用速度命令控制UI
        """
        super().__init__(env, window_name)
        
        self._enable_velocity_control = enable_velocity_control
        self._debug_plots: Dict[str, DebugPlotConfig] = {}
        self._plot_widgets: Dict[str, LiveLinePlot] = {}
        self._image_widgets: Dict[str, ImagePlot] = {}
        self._debug_data: Dict[str, torch.Tensor] = {}
        self._controls_built = False
        
        self._close_unnecessary_windows()
        self._create_debug_window(debug_window_name)
    
    def _close_unnecessary_windows(self):
        """关闭不需要的默认窗口"""
        close_list = [
            "Stage", "Render Settings", "Layer", "Content", 
            "Console", "Property", "Semantics Schema Editor", "Simulation Settings"
        ]
        for window in ui.Workspace.get_windows():
            if window.title in close_list:
                window.visible = False
    
    def _create_debug_window(self, title: str):
        """创建debug窗口容器"""
        self.debug_window = ui.Window(
            title=title,
            width=300,
            height=300,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM,
            visible=True,
        )
        asyncio.ensure_future(self._dock_window(window_title=title))
        
        with self.debug_window.frame:
            self.debug_vstack = ui.VStack(spacing=5, height=0)
    
    def _build_debug_controls(self):
        """延迟创建所有UI控件"""
        if self._controls_built:
            return
        
        # 子类注册需要的图表
        self._register_debug_plots()
        
        with self.debug_vstack:
            # 速度命令UI
            if self._enable_velocity_control:
                self._setup_velocity_command_ui()
            
            # 创建所有已注册的图表
            for plot_cfg in self._debug_plots.values():
                if plot_cfg.enabled:
                    self._create_plot_widget(plot_cfg)
        
        self._controls_built = True
    
    @abstractmethod
    def _register_debug_plots(self):
        """
        子类实现：注册需要的图表
        
        示例:
            self.register_plot("velocity", ["cmd", "est", "real"])
            self.register_plot("error", ["MSE", "MAE"], plot_height=100)
        """
        pass
    
    @abstractmethod
    def _update_debug_data(self) -> Dict[str, torch.Tensor]:
        """
        子类实现：返回最新的调试数据
        
        Returns:
            字典，键为图表名称，值为对应的数据张量
            
        示例:
            return {
                "velocity": torch.tensor([cmd, est, real]),
                "error": torch.tensor([mse, mae])
            }
        """
        pass
    
    def register_plot(
        self,
        name: str,
        legends: List[str],
        data_type: DebugDataType = DebugDataType.PLOT,
        plot_height: int = 150,
        collapsed: bool = False,
        enabled: bool = True,
    ):
        """
        注册一个调试图表
        
        Args:
            name: 图表唯一标识
            legends: 图例列表
            data_type: 数据类型（曲线或图片）
            plot_height: 图表高度
            collapsed: 是否默认折叠
            enabled: 是否启用
        """
        self._debug_plots[name] = DebugPlotConfig(
            name=name,
            legends=legends,
            data_type=data_type,
            plot_height=plot_height,
            collapsed=collapsed,
            enabled=enabled,
        )
    
    def _create_plot_widget(self, cfg: DebugPlotConfig):
        """根据配置创建图表控件"""
        frame = CollapsableFrame(cfg.name, collapsed=cfg.collapsed)
        
        with frame:
            if cfg.data_type == DebugDataType.PLOT.value:
                # 创建占位数据（全零）
                placeholder = [[0.0] for _ in cfg.legends]
                widget = LiveLinePlot(
                    y_data=placeholder,
                    plot_height=cfg.plot_height,
                    show_legend=True,
                    legends=cfg.legends,
                )
                self._plot_widgets[cfg.name] = widget
                
            elif cfg.data_type == DebugDataType.IMAGE.value:
                widget = ImagePlot(
                    image=np.zeros((10, 10)),
                    label=cfg.legends[0] if cfg.legends else cfg.name,
                )
                self._image_widgets[cfg.name] = widget
    
    def _setup_velocity_command_ui(self):
        """设置速度命令控制UI（可被子类重写）"""
        self.ui_window_elements["vel_cmd_setting"] = isaacsim.gui.components.ui_utils.xyz_builder(
            label="Velocity Command",
            tooltip="Modify current velocity command.",
            default_val=[0.0, 0.0, 0.0],
            step=0.1,
            on_value_changed_fn=[self._set_velocity_command] * 3,
        )
        
        with ui.HStack(spacing=5):
            ui.Label("Current Velocity:", width=120)
            self.ui_window_elements["vel_x"] = ui.Label("X: 0.0000", width=60)
            self.ui_window_elements["vel_y"] = ui.Label("Y: 0.0000", width=60)
            self.ui_window_elements["vel_yaw"] = ui.Label("Yaw: 0.0000", width=60)
    
    def _set_velocity_command(self, model: ui.SimpleFloatModel):
        """速度命令回调"""
        vel_cmd = [
            self.ui_window_elements["vel_cmd_setting"][i].get_value_as_float() 
            for i in range(3)
        ]
        vel_term = self.env.command_manager._terms.get("base_velocity")
        if vel_term:
            vel_term.command[0, :2] = torch.tensor(vel_cmd[:2], device=self.env.device)
            vel_term.heading_target[0] = vel_cmd[2]
        
        # 重新采样步态命令（如果存在）
        gait_term = self.env.command_manager._terms.get("gait_command")
        if gait_term is not None:
            gait_term._resample_command([0])
    
    def _update_velocity_display(self):
        """更新速度显示"""
        if not self._enable_velocity_control:
            return
            
        vel_term = self.env.command_manager._terms.get("base_velocity")
        if vel_term and all(k in self.ui_window_elements for k in ["vel_x", "vel_y", "vel_yaw"]):
            cmd = vel_term.command[0]
            self.ui_window_elements["vel_x"].text = f"X: {cmd[0]:.4f}"
            self.ui_window_elements["vel_y"].text = f"Y: {cmd[1]:.4f}"
            self.ui_window_elements["vel_yaw"].text = f"Yaw: {cmd[2]:.4f}"
    
    def fresh_debug_info_frame(self):
        """刷新窗口内容（在环境step中调用）"""
        # 首次调用时创建控件
        self._build_debug_controls()
        
        # 获取最新数据
        try:
            self._debug_data = self._update_debug_data()
        except Exception as e:
            # 数据更新失败时不中断，继续运行
            print(f"[ManagerDebugRLEnvWindow] Failed to update debug data: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 更新速度显示
        self._update_velocity_display()
        
        # 更新所有图表
        for name, data in self._debug_data.items():
            if data is None:
                continue
            try:
                if name in self._plot_widgets:
                    self._plot_widgets[name].add_datapoint(data.tolist())
                elif name in self._image_widgets:
                    self._image_widgets[name].update_image(data.cpu().numpy())
            except Exception as e:
                print(f"[ManagerDebugRLEnvWindow] Failed to update plot '{name}': {e}")

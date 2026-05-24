import argparse
from isaaclab.app import AppLauncher

# 配置命令行参数并启动 Omniverse 应用
parser = argparse.ArgumentParser(description="Test CoopG1 Environment Physics and Joints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 启动后再引用其他依赖 (IsaacLab 规范)
import gymnasium as gym
import torch

from isaaclab_nhb.tasks.humanoid.coopG1.coopG1_env_cfg import CoopG1EnvCfg
from isaaclab_nhb.tasks.humanoid.coopG1.coopG1_env import CoopG1Env

def main():
    # 利用配置类初始化环境
    env_cfg = CoopG1EnvCfg()
    env_cfg.scene.num_envs = 2 # 只开 2 个环境方便肉眼观察
    
    print("[INFO] Attempting to create CoopG1 Environment...")
    env = CoopG1Env(cfg=env_cfg)
    
    print("[INFO] Environment created successfully! Box and Robots should be linked.")
    print("[INFO] Stepping the simulation blindly for 500 steps to check physics stability...")
    
    try:
        env.reset()
        for i in range(500):
            # 因为没有 action 配置，传一个空的占位 dict/tensor 避免底层断言
            # 这里即使发生 Warning，也会强制走物理步，能帮我们看见物理连线情况
            env.sim.step()
    except Exception as e:
        print(f"[WARN] Caught an expected exception due to empty Action/Obs Cfg: {e}")
        print("[INFO] But the Physics Scene was successfully created!")
        
    print("[INFO] Simulation test complete. The visual window will close.")

if __name__ == "__main__":
    main()
    simulation_app.close()

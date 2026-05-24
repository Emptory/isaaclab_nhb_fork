#!/usr/bin/env python3
"""
读取并查看 motion.npz 文件的内容
"""

import numpy as np
import sys
import os

def read_npz_file(filepath):
    """读取并显示 npz 文件的内容"""
    
    if not os.path.exists(filepath):
        print(f"错误：文件 '{filepath}' 不存在")
        return
    
    print(f"正在读取文件: {filepath}")
    print("=" * 80)
    
    # 加载 npz 文件
    data = np.load(filepath, allow_pickle=True)
    
    print(f"\n文件类型: {type(data)}")
    print(f"包含的数组/键: {list(data.keys())}")
    print(f"数组数量: {len(data.keys())}")
    print("\n" + "=" * 80)
    
    # 遍历所有的数组并显示信息
    for i, key in enumerate(data.keys(), 1):
        print(f"\n[{i}] 键名: '{key}'")
        print("-" * 80)
        
        arr = data[key]
        print(f"  数据类型: {type(arr)}")
        
        # 如果是 numpy 数组，显示详细信息
        if isinstance(arr, np.ndarray):
            print(f"  形状 (shape): {arr.shape}")
            print(f"  数据类型 (dtype): {arr.dtype}")
            print(f"  元素总数: {arr.size}")
            print(f"  维度数: {arr.ndim}")
            
            # 显示数值统计信息（如果是数值类型）
            if np.issubdtype(arr.dtype, np.number):
                print(f"  最小值: {np.min(arr)}")
                print(f"  最大值: {np.max(arr)}")
                print(f"  平均值: {np.mean(arr)}")
                print(f"  标准差: {np.std(arr)}")
            
            # 显示部分数据内容
            print(f"\n  数据预览:")
            if arr.size <= 10:
                print(f"    {arr}")
            else:
                # 只显示前几个元素
                if arr.ndim == 1:
                    print(f"    前5个元素: {arr[:5]}")
                    if arr.size > 5:
                        print(f"    后5个元素: {arr[-5:]}")
                elif arr.ndim == 2:
                    print(f"    前3行:")
                    print(f"{arr[:3]}")
                    if arr.shape[0] > 3:
                        print(f"    ...")
                else:
                    print(f"    {arr.flatten()[:10]}...")
        else:
            # 不是 numpy 数组的情况
            print(f"  内容: {arr}")
    
    print("\n" + "=" * 80)
    print("读取完成!")
    
    # 关闭文件
    data.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 如果没有提供文件路径，尝试使用默认的 motion.npz
        default_file = "motion.npz"
        if os.path.exists(default_file):
            filepath = default_file
        else:
            print("用法: python read_motion_npz.py <npz文件路径>")
            print("\n示例:")
            print("  python read_motion_npz.py motion.npz")
            print("  python read_motion_npz.py /path/to/your/motion.npz")
            sys.exit(1)
    else:
        filepath = sys.argv[1]
    
    read_npz_file(filepath)

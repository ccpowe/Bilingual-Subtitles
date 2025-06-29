#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超简单配置测试 - 无阻塞版本
"""

import os
import psutil
from audio_to_srt_optimized import get_recommended_config

def main():
    print("🚀 快速系统配置检测")
    print("=" * 40)
    
    # 检查系统资源
    memory = psutil.virtual_memory()
    cpu_count = os.cpu_count() or 4
    
    print(f"💾 内存: {memory.total//1024//1024//1024:.1f}GB")
    print(f"📊 内存使用: {memory.percent:.1f}%")
    print(f"🖥️ CPU核心: {cpu_count}")
    
    # 获取推荐配置
    config = get_recommended_config()
    
    print(f"\n💡 推荐配置:")
    print(f"   模型: {config['model_size']}")
    print(f"   计算类型: {config['compute_type']}")
    print(f"   CPU线程: {config['cpu_threads']}")
    
    # 检查测试文件
    audio_file = r"D:\Code_vs\cut\srt_translate\str_file\Building_Ambient_Agents_with_LangGraph_-_LangChain_Academy (1).mp4"
    
    if os.path.exists(audio_file):
        print(f"\n✅ 测试文件: 存在")
        
        print(f"\n🎯 推荐运行命令:")
        cmd = f'python audio_to_srt_optimized.py "{audio_file}" -l en -m {config["model_size"]} --compute-type {config["compute_type"]} --cpu-threads {config["cpu_threads"]}'
        print(cmd)
        
        print(f"\n⚡ 或使用自动配置:")
        cmd_auto = f'python audio_to_srt_optimized.py "{audio_file}" -l en --auto-config'
        print(cmd_auto)
    else:
        print(f"\n⚠️ 测试文件: 不存在")
    
    # 内存建议
    if memory.percent > 80:
        print(f"\n⚠️ 内存使用率较高({memory.percent:.1f}%)，建议:")
        print("   - 关闭其他应用程序")
        print("   - 使用tiny模型")
    elif memory.percent > 70:
        print(f"\n💡 内存使用率中等({memory.percent:.1f}%)，建议:")
        print("   - 使用base模型(推荐)")
    else:
        print(f"\n✅ 内存状态良好({memory.percent:.1f}%)，可以:")
        print("   - 使用small模型获得更好质量")

if __name__ == "__main__":
    main() 
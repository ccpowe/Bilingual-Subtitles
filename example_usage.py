#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT字幕优化器使用示例
展示如何使用srt_optimizer.py优化字幕文件
"""

from srt_optimizer import SRTOptimizer
import os

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建优化器实例
    optimizer = SRTOptimizer()
    
    # 输入文件路径
    input_file = "str_file/无标题视频 (1).srt"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    try:
        # 执行优化
        stats = optimizer.optimize(
            input_path=input_file,
            output_path="str_file/无标题视频_优化版.srt",
            create_template=True  # 同时创建双语模板
        )
        
        # 输出结果
        print(f"✅ 优化完成!")
        print(f"原始字幕数量: {stats['original_count']}")
        print(f"优化后字幕数量: {stats['optimized_count']}")
        print(f"减少比例: {stats['reduction_rate']:.1f}%")
        print(f"输出文件: {stats['output_path']}")
        
        if stats['warnings']:
            print(f"⚠️ 质量检查警告: {len(stats['warnings'])}个")
            
    except Exception as e:
        print(f"❌ 优化失败: {e}")

def example_custom_config():
    """自定义配置示例"""
    print("\n=== 自定义配置示例 ===")
    
    # 自定义配置
    custom_config = {
        'min_duration': 2500,      # 更短的最小时长
        'max_duration': 6000,      # 更短的最大时长
        'max_words': 15,           # 更少的最大单词数
        'merge_gap_threshold': 300, # 更严格的合并阈值
    }
    
    # 创建优化器实例
    optimizer = SRTOptimizer(custom_config)
    
    input_file = "str_file/无标题视频 (1).srt"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    try:
        stats = optimizer.optimize(
            input_path=input_file,
            output_path="str_file/无标题视频_自定义优化版.srt"
        )
        
        print(f"✅ 自定义配置优化完成!")
        print(f"减少比例: {stats['reduction_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ 优化失败: {e}")

def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建优化器
    optimizer = SRTOptimizer()
    
    # 假设有多个SRT文件需要处理
    srt_files = [
        "str_file/无标题视频 (1).srt",
        # "str_file/其他视频.srt",  # 添加更多文件
    ]
    
    successful = 0
    failed = 0
    
    for srt_file in srt_files:
        if not os.path.exists(srt_file):
            print(f"⚠️ 跳过不存在的文件: {srt_file}")
            continue
            
        try:
            print(f"正在处理: {srt_file}")
            stats = optimizer.optimize(
                input_path=srt_file,
                create_template=True
            )
            print(f"  ✅ 成功 - 减少{stats['reduction_rate']:.1f}%的字幕数量")
            successful += 1
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed += 1
    
    print(f"\n批量处理完成: 成功{successful}个，失败{failed}个")

def analyze_original_file():
    """分析原始文件的统计信息"""
    print("\n=== 原始文件分析 ===")
    
    input_file = "str_file/无标题视频 (1).srt"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    optimizer = SRTOptimizer()
    
    try:
        # 解析原始文件
        subtitles = optimizer.parse_srt(input_file)
        
        if not subtitles:
            print("❌ 无法解析字幕文件")
            return
        
        # 统计信息
        total_duration = sum(sub.duration() for sub in subtitles) / 1000  # 转换为秒
        avg_duration = total_duration / len(subtitles)
        avg_words = sum(sub.word_count() for sub in subtitles) / len(subtitles)
        
        short_subtitles = len([sub for sub in subtitles if sub.duration() < 3000])
        long_subtitles = len([sub for sub in subtitles if sub.duration() > 8000])
        
        print(f"字幕总数: {len(subtitles)}")
        print(f"总时长: {total_duration:.1f}秒")
        print(f"平均时长: {avg_duration:.1f}秒")
        print(f"平均单词数: {avg_words:.1f}")
        print(f"过短字幕数(<3秒): {short_subtitles} ({short_subtitles/len(subtitles)*100:.1f}%)")
        print(f"过长字幕数(>8秒): {long_subtitles} ({long_subtitles/len(subtitles)*100:.1f}%)")
        
        # 显示前几个字幕的示例
        print(f"\n前5个字幕示例:")
        for i, sub in enumerate(subtitles[:5]):
            print(f"  {i+1}. [{sub.duration()/1000:.1f}s] {sub.text}")
    
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def main():
    """主函数"""
    print("SRT字幕优化器使用示例")
    print("=" * 50)
    
    # 首先分析原始文件
    analyze_original_file()
    
    # 基本使用示例
    example_basic_usage()
    
    # 自定义配置示例
    example_custom_config()
    
    # 批量处理示例
    example_batch_processing()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成!")
    print("\n使用提示:")
    print("1. 检查生成的文件: *_optimized.srt")
    print("2. 检查双语模板: *_bilingual_template.srt")
    print("3. 可以用视频播放器测试优化后的字幕效果")

if __name__ == "__main__":
    main() 
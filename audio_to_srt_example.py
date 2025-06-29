#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频转SRT字幕生成器使用示例
展示如何使用audio_to_srt.py生成字幕文件
"""

from audio_to_srt import AudioToSRT
import os
from pathlib import Path

def example_single_file():
    """单文件转换示例"""
    print("=== 单文件转换示例 ===")
    
    # 创建转换器（使用base模型）
    converter = AudioToSRT(
        model_size="small",     # 你之前选择的模型
        device="cpu",          # 如果有GPU可以改为"cuda"
        compute_type="float32"
    )
    
    # 加载模型
    if not converter.load_model():
        print("❌ 模型加载失败")
        return
    
    # 示例音频文件（请替换为你的音频文件路径）
    audio_file = r"D:\Code_vs\cut\srt_translate\str_file\Building_Ambient_Agents_with_LangGraph_-_LangChain_Academy (1).mp4"  # 替换为实际文件
    
    if os.path.exists(audio_file):
        # 转换为SRT
        result = converter.transcribe_to_srt(
            audio_file=audio_file,
            language="en",  # 指定中文，如果是英文用"en"
            initial_prompt="This is a technical lecture recording:"  # 可选的提示词
        )
        
        if result:
            print(f"✅ 字幕文件已生成: {result}")
            
            # 使用SRT优化器进一步优化（可选）
            print("\n是否要使用SRT优化器优化字幕？")
            print("可以运行: python srt_optimizer.py \"{}\" -t".format(result))
        else:
            print("❌ 转换失败")
    else:
        print(f"❌ 音频文件不存在: {audio_file}")

def example_with_different_models():
    """不同模型对比示例"""
    print("\n=== 不同模型对比示例 ===")
    
    audio_file = "your_audio.mp4"  # 替换为实际文件
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    # 测试不同大小的模型
    models = ["tiny", "base", "small"]  # 根据需要添加更大的模型
    
    for model_size in models:
        print(f"\n使用 {model_size} 模型...")
        
        converter = AudioToSRT(model_size=model_size)
        
        if converter.load_model():
            output_file = f"output_{model_size}.srt"
            result = converter.transcribe_to_srt(
                audio_file=audio_file,
                output_file=output_file,
                language="zh"
            )
            
            if result:
                print(f"  ✅ {model_size} 模型完成: {result}")
            else:
                print(f"  ❌ {model_size} 模型失败")
        else:
            print(f"  ❌ {model_size} 模型加载失败")

def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建转换器
    converter = AudioToSRT(model_size="base")
    
    if not converter.load_model():
        print("❌ 模型加载失败")
        return
    
    # 批量处理目录（请替换为你的音频文件目录）
    input_dir = "audio_files"  # 包含音频文件的目录
    output_dir = "srt_files"   # 输出SRT文件的目录
    
    if os.path.exists(input_dir):
        converter.batch_convert(
            input_dir=input_dir,
            output_dir=output_dir,
            language="zh"  # 指定语言
        )
    else:
        print(f"❌ 输入目录不存在: {input_dir}")
        print("请创建目录并放入音频文件，或修改路径")

def example_with_optimization():
    """转换+优化一体化示例"""
    print("\n=== 转换+优化一体化示例 ===")
    
    audio_file = "your_audio.mp4"  # 替换为实际文件
    
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    # 第一步：音频转SRT
    print("第一步：音频转SRT...")
    converter = AudioToSRT(model_size="base")
    
    if not converter.load_model():
        print("❌ 模型加载失败")
        return
    
    raw_srt = converter.transcribe_to_srt(
        audio_file=audio_file,
        language="zh"
    )
    
    if not raw_srt:
        print("❌ 音频转换失败")
        return
    
    print(f"✅ 原始SRT已生成: {raw_srt}")
    
    # 第二步：SRT优化
    print("\n第二步：SRT优化...")
    try:
        from srt_optimizer import SRTOptimizer
        
        optimizer = SRTOptimizer()
        
        # 生成优化后的文件名
        raw_path = Path(raw_srt)
        optimized_srt = raw_path.parent / f"{raw_path.stem}_optimized.srt"
        
        stats = optimizer.optimize(
            input_path=raw_srt,
            output_path=str(optimized_srt),
            create_template=True  # 同时创建双语模板
        )
        
        print(f"✅ 优化完成:")
        print(f"  原始字幕数: {stats['original_count']}")
        print(f"  优化后数: {stats['optimized_count']}")
        print(f"  减少比例: {stats['reduction_rate']:.1f}%")
        print(f"  优化文件: {stats['output_path']}")
        
        # 第三步：建议下一步操作
        print(f"\n🎯 建议下一步操作:")
        print(f"1. 检查优化后的字幕文件")
        print(f"2. 使用视频播放器测试字幕效果")
        print(f"3. 如果需要翻译，编辑双语模板文件")
        
    except ImportError:
        print("⚠️ SRT优化器不可用，请确保srt_optimizer.py在同一目录")

def main():
    """主函数"""
    print("音频转SRT字幕生成器使用示例")
    print("=" * 50)
    
    print("⚠️ 注意：请先安装必要的依赖:")
    print("pip install faster-whisper soundfile")
    print()
    
    print("📝 使用前请修改示例中的文件路径:")
    print("- 将 'your_audio.mp4' 替换为实际的音频文件路径")
    print("- 根据需要调整语言设置和模型大小")
    print()
    
    # 检查依赖
    try:
        import faster_whisper
        print("✅ faster_whisper 已安装")
    except ImportError:
        print("❌ 请安装 faster_whisper: pip install faster-whisper")
        return
    
    try:
        import soundfile
        print("✅ soundfile 已安装")
    except ImportError:
        print("❌ 请安装 soundfile: pip install soundfile")
        return
    
    print("\n" + "=" * 50)
    
    # 运行示例（注释掉不需要的）
    example_single_file()
    # example_with_different_models()
    # example_batch_processing()
    # example_with_optimization()
    
    print("\n" + "=" * 50)
    print("示例运行完成!")
    print("\n💡 命令行使用方法:")
    print("python audio_to_srt.py your_audio.mp4 -l zh -m base")
    print("python audio_to_srt.py audio_folder --batch -o srt_folder")

if __name__ == "__main__":
    main() 
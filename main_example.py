#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py 使用示例
演示如何使用主控制脚本的各种模式
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"📝 命令: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 执行成功")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ 执行失败")
            if result.stderr:
                print(result.stderr)
    except Exception as e:
        print(f"❌ 执行异常: {e}")

def check_environment():
    """检查环境配置"""
    print("🔍 环境检查")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("MODEL_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    
    print(f"OPENAI_API_KEY: {'✅ 已设置' if api_key else '❌ 未设置'}")
    print(f"MODEL_BASE_URL: {'✅ 已设置' if base_url else '⚪ 使用默认'}")
    print(f"MODEL_NAME: {model_name or 'gpt-4o-mini (默认)'}")
    
    # 检查依赖
    try:
        import faster_whisper
        print("✅ faster-whisper: 已安装")
    except ImportError:
        print("❌ faster-whisper: 未安装")
    
    try:
        import langgraph
        import langchain_openai
        print("✅ 翻译依赖: 已安装")
    except ImportError:
        print("❌ 翻译依赖: 未安装")
    
    # 检查示例文件
    test_audio = "str_file/Building_Ambient_Agents_with_LangGraph_-_LangChain_Academy (1).mp4"
    test_srt = "str_file/Building_Ambient_Agents_20250629_155924.srt"
    
    print(f"\n📁 示例文件:")
    print(f"测试音频: {'✅ 存在' if os.path.exists(test_audio) else '❌ 不存在'}")
    print(f"测试SRT: {'✅ 存在' if os.path.exists(test_srt) else '❌ 不存在'}")

def example_dry_run():
    """空运行示例 - 验证配置"""
    print("\n" + "=" * 60)
    print("🔍 示例1: 空运行模式 (验证配置)")
    print("=" * 60)
    
    # 测试完整工作流配置
    cmd = 'uv run python main.py "test_video.mp4" --mode full -l en -s 英文 -t 中文 --dry-run'
    run_command(cmd, "验证完整工作流配置")
    
    # 测试仅转录配置
    cmd = 'uv run python main.py "test_video.mp4" --mode transcribe -l en --auto-config --dry-run'
    run_command(cmd, "验证转录配置")
    
    # 测试仅翻译配置
    cmd = 'uv run python main.py "test.srt" --mode translate -s 英文 -t 中文 --dry-run'
    run_command(cmd, "验证翻译配置")

def example_transcribe_only():
    """仅转录示例"""
    print("\n" + "=" * 60)
    print("🎵 示例2: 仅音频转录模式")
    print("=" * 60)
    
    # 假设的音频文件路径
    test_audio = "str_file/Building_Ambient_Agents_with_LangGraph_-_LangChain_Academy (1).mp4"
    
    if not os.path.exists(test_audio):
        print(f"⚠️ 测试音频文件不存在: {test_audio}")
        print("💡 请替换为实际的音频文件路径")
        test_audio = "your_audio_file.mp4"
    
    commands = [
        # 基础转录
        f'uv run python main.py "{test_audio}" --mode transcribe -l en',
        
        # 自动配置转录
        f'uv run python main.py "{test_audio}" --mode transcribe -l en --auto-config',
        
        # 高质量转录
        f'uv run python main.py "{test_audio}" --mode transcribe -l en -m small --compute-type float16',
        
        # 快速转录
        f'uv run python main.py "{test_audio}" --mode transcribe -l en -m tiny --compute-type int8'
    ]
    
    descriptions = [
        "基础英文转录",
        "自动配置转录",
        "高质量转录 (small模型)",
        "快速转录 (tiny模型)"
    ]
    
    for cmd, desc in zip(commands, descriptions):
        print(f"\n📝 {desc}:")
        print(f"命令: {cmd}")

def example_translate_only():
    """仅翻译示例"""
    print("\n" + "=" * 60)
    print("🌐 示例3: 仅字幕翻译模式")
    print("=" * 60)
    
    # 假设的SRT文件路径
    test_srt = "str_file/Building_Ambient_Agents_20250629_155924.srt"
    
    if not os.path.exists(test_srt):
        print(f"⚠️ 测试SRT文件不存在: {test_srt}")
        print("💡 请先生成SRT文件或替换为实际路径")
        test_srt = "your_subtitle.srt"
    
    commands = [
        # 英译中
        f'uv run python main.py "{test_srt}" --mode translate -s 英文 -t 中文',
        
        # 英译日
        f'uv run python main.py "{test_srt}" --mode translate -s 英文 -t 日语',
        
        # 小批量高质量翻译
        f'uv run python main.py "{test_srt}" --mode translate -s 英文 -t 中文 --llm-model gpt-4 -b 3',
        
        # 大批量快速翻译
        f'uv run python main.py "{test_srt}" --mode translate -s 英文 -t 中文 -b 10'
    ]
    
    descriptions = [
        "英文→中文翻译",
        "英文→日语翻译",
        "高质量翻译 (GPT-4, 小批量)",
        "快速翻译 (大批量)"
    ]
    
    for cmd, desc in zip(commands, descriptions):
        print(f"\n📝 {desc}:")
        print(f"命令: {cmd}")

def example_full_workflow():
    """完整工作流示例"""
    print("\n" + "=" * 60)
    print("🚀 示例4: 完整工作流模式")
    print("=" * 60)
    
    test_audio = "str_file/Building_Ambient_Agents_with_LangGraph_-_LangChain_Academy (1).mp4"
    
    if not os.path.exists(test_audio):
        print(f"⚠️ 测试音频文件不存在: {test_audio}")
        print("💡 请替换为实际的音频文件路径")
        test_audio = "your_audio_file.mp4"
    
    commands = [
        # 标准完整流程
        f'uv run python main.py "{test_audio}" --mode full -l en -s 英文 -t 中文',
        
        # 自动配置完整流程
        f'uv run python main.py "{test_audio}" --mode full -l en -s 英文 -t 中文 --auto-config',
        
        # 高质量完整流程
        f'uv run python main.py "{test_audio}" --mode full -l en -s 英文 -t 中文 -m small --llm-model gpt-4 -b 3',
        
        # 快速完整流程
        f'uv run python main.py "{test_audio}" --mode full -l en -s 英文 -t 中文 -m tiny --compute-type int8 -b 8'
    ]
    
    descriptions = [
        "标准完整流程",
        "自动配置完整流程",
        "高质量完整流程",
        "快速完整流程"
    ]
    
    for cmd, desc in zip(commands, descriptions):
        print(f"\n📝 {desc}:")
        print(f"命令: {cmd}")

def example_advanced_usage():
    """高级用法示例"""
    print("\n" + "=" * 60)
    print("⚙️ 示例5: 高级用法")
    print("=" * 60)
    
    print("📝 详细日志模式:")
    print('uv run python main.py "video.mp4" --mode full -l en -s 英文 -t 中文 --verbose')
    
    print("\n📝 安静模式:")
    print('uv run python main.py "video.mp4" --mode full -l en -s 英文 -t 中文 --quiet')
    
    print("\n📝 覆盖环境变量:")
    print('uv run python main.py "srt_file.srt" --mode translate --api-key your_key --base-url https://your.proxy.com/v1 --llm-model gpt-4')
    
    print("\n📝 批量处理脚本:")
    batch_script = '''
# 批量处理多个音频文件
for file in *.mp4; do
    echo "处理文件: $file"
    uv run python main.py "$file" --mode full -l en -s 英文 -t 中文 --auto-config
done
'''
    print(batch_script)

def example_environment_setup():
    """环境配置示例"""
    print("\n" + "=" * 60)
    print("🔧 示例6: 环境配置")
    print("=" * 60)
    
    print("📝 代理提供商配置 (Linux/macOS):")
    env_linux = '''
export OPENAI_API_KEY=your_api_key
export MODEL_BASE_URL=https://your.proxy.com/v1
export MODEL_NAME=gpt-4o-mini
'''
    print(env_linux)
    
    print("📝 代理提供商配置 (Windows PowerShell):")
    env_windows = '''
$env:OPENAI_API_KEY="your_api_key"
$env:MODEL_BASE_URL="https://your.proxy.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
'''
    print(env_windows)
    
    print("📝 验证配置:")
    print('uv run python main.py "test.mp4" --dry-run')

def example_troubleshooting():
    """故障排除示例"""
    print("\n" + "=" * 60)
    print("🔍 示例7: 故障排除")
    print("=" * 60)
    
    print("📝 检查依赖:")
    print('uv add faster-whisper soundfile psutil langgraph langchain-openai')
    
    print("\n📝 内存不足时使用tiny模型:")
    print('uv run python main.py "video.mp4" --mode transcribe -m tiny --compute-type int8 --cpu-threads 2')
    
    print("\n📝 翻译失败时减少批量大小:")
    print('uv run python main.py "subtitle.srt" --mode translate -s 英文 -t 中文 -b 2')
    
    print("\n📝 网络问题时的重试:")
    print('# 如果翻译中断，可以单独运行翻译部分')
    print('uv run python main.py "existing_subtitle.srt" --mode translate -s 英文 -t 中文')

def main():
    """主函数"""
    print("🎬 main.py 使用示例")
    print("=" * 60)
    
    while True:
        print("\n🔧 选择示例:")
        print("0. 环境检查")
        print("1. 空运行模式 (验证配置)")
        print("2. 仅音频转录")
        print("3. 仅字幕翻译")  
        print("4. 完整工作流")
        print("5. 高级用法")
        print("6. 环境配置")
        print("7. 故障排除")
        print("q. 退出")
        
        choice = input("\n请选择 (0-7, q): ").strip()
        
        if choice == 'q':
            print("👋 再见!")
            break
        elif choice == '0':
            check_environment()
        elif choice == '1':
            example_dry_run()
        elif choice == '2':
            example_transcribe_only()
        elif choice == '3':
            example_translate_only()
        elif choice == '4':
            example_full_workflow()
        elif choice == '5':
            example_advanced_usage()
        elif choice == '6':
            example_environment_setup()
        elif choice == '7':
            example_troubleshooting()
        else:
            print("❌ 无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main() 
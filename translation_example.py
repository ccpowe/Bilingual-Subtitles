#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT翻译Agent使用示例
演示如何使用LangGraph翻译Agent制作双语字幕
"""

import os
from srt_translator_agent import SRTTranslatorAgent
from dotenv import load_dotenv

load_dotenv()

def example_basic_translation():
    """基础翻译示例"""
    print("🎬 基础翻译示例")
    print("=" * 50)
    
    # 配置
    input_file = r"D:\Code_vs\cut\srt_translate\srt_file\testone.srt"  # 请替换为实际文件
    
    # 检查环境变量配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("MODEL_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    
    if not api_key:
        print("❌ 请设置OPENAI_API_KEY环境变量")
        return
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        print("💡 请先运行音频转字幕生成SRT文件")
        return
    
    print(f"🔧 配置信息:")
    print(f"   模型: {model_name or 'gpt-4o-mini (默认)'}")
    print(f"   基础URL: {base_url or '官方API (默认)'}")
    print(f"   API密钥: {'已设置' if api_key else '未设置'}")
    
    try:
        # 创建翻译Agent (使用环境变量配置)
        print(f"🔧 批量配置: batch_size=3 (小批量翻译)")
        agent = SRTTranslatorAgent(
            batch_size=3  # 小批量，减少API调用成本和避免超时
        )
        
        # 执行翻译（英文→中文）
        output_file = agent.translate_srt(
            input_file=input_file,
            source_lang="英文",
            target_lang="中文"
        )
        
        print(f"🎉 翻译完成!")
        print(f"📁 双语字幕文件: {output_file}")
        
    except Exception as e:
        print(f"❌ 翻译失败: {e}")

def example_advanced_translation():
    """高级翻译示例"""
    print("\n🚀 高级翻译示例")
    print("=" * 50)
    
    # 配置
    input_file = "str_file/Building_Ambient_Agents_20250629_155924.srt"
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or not os.path.exists(input_file):
        print("⏭️ 跳过高级示例（需要API密钥和SRT文件）")
        return
    
    try:
        # 创建高质量翻译Agent (使用环境变量配置)
        print(f"🔧 高级配置: llm_model=gpt-4, batch_size=2 (高质量翻译)")
        agent = SRTTranslatorAgent(
            llm_model="gpt-4",  # 覆盖环境变量，使用高质量模型
            batch_size=2  # 小批量获得更好的翻译质量，避免超时
        )
        
        # 执行翻译（英文→中文）
        output_file = agent.translate_srt(
            input_file=input_file,
            source_lang="英文",
            target_lang="中文"
        )
        
        print(f"🎉 高质量翻译完成!")
        print(f"📁 双语字幕文件: {output_file}")
        
    except Exception as e:
        print(f"❌ 高级翻译失败: {e}")

def example_multilingual():
    """多语言翻译示例"""
    print("\n🌍 多语言翻译示例")
    print("=" * 50)
    
    # 模拟不同语言对的翻译
    language_pairs = [
        ("英文", "中文"),
        ("英文", "日语"),
        ("英文", "法语"),
        ("中文", "英文")
    ]
    
    input_file = "str_file/Building_Ambient_Agents_20250629_155924.srt"
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or not os.path.exists(input_file):
        print("⏭️ 跳过多语言示例（需要API密钥和SRT文件）")
        return
    
    # 创建翻译Agent (使用环境变量配置)
    print(f"🔧 多语言配置: batch_size=3 (平衡速度和质量)")
    agent = SRTTranslatorAgent(
        batch_size=3  # 中等批量大小，平衡速度和质量
    )
    
    for source_lang, target_lang in language_pairs:
        try:
            print(f"\n🔄 翻译: {source_lang} → {target_lang}")
            
            output_file = agent.translate_srt(
                input_file=input_file,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            print(f"✅ 完成: {output_file}")
            
        except Exception as e:
            print(f"❌ {source_lang}→{target_lang} 翻译失败: {e}")

def example_batch_processing():
    """批量处理示例"""
    print("\n📦 批量处理示例")
    print("=" * 50)
    
    # 模拟多个SRT文件
    srt_files = [
        "str_file/video1_20250629_155924.srt",
        "str_file/video2_20250629_160235.srt",
        "str_file/video3_20250629_160512.srt"
    ]
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⏭️ 跳过批量处理示例（需要API密钥）")
        return
    
    # 创建翻译Agent (使用环境变量配置)
    print(f"🔧 批量处理配置: batch_size=5 (批量处理优化)")
    agent = SRTTranslatorAgent(
        batch_size=5  # 较大批量，提高批量处理效率
    )
    
    successful_translations = 0
    failed_translations = 0
    
    for srt_file in srt_files:
        if not os.path.exists(srt_file):
            print(f"⏭️ 跳过不存在的文件: {srt_file}")
            continue
            
        try:
            print(f"\n🔄 处理文件: {srt_file}")
            
            output_file = agent.translate_srt(
                input_file=srt_file,
                source_lang="英文",
                target_lang="中文"
            )
            
            print(f"✅ 成功: {output_file}")
            successful_translations += 1
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            failed_translations += 1
    
    print(f"\n📊 批量处理结果:")
    print(f"   成功: {successful_translations}")
    print(f"   失败: {failed_translations}")

def example_workflow_integration():
    """工作流集成示例"""
    print("\n🔗 完整工作流示例")
    print("=" * 50)
    
    # 演示从音频到双语字幕的完整流程
    audio_file = r"D:\Code_vs\cut\srt_translate\str_file\Building_Ambient_Agents_with_LangGraph_-_LangChain_Academy (1).mp4"
    
    print("完整工作流程:")
    print("1. 音频转录 → SRT字幕")
    print("2. SRT翻译 → 双语字幕")
    print("3. 质量检查和优化")
    
    if os.path.exists(audio_file):
        print(f"📁 找到音频文件: {os.path.basename(audio_file)}")
        
        print("\n建议的命令序列:")
        print("# 步骤1: 生成字幕")
        print(f'uv run python audio_to_srt_optimized.py "{audio_file}" -l en --auto-config')
        
        print("\n# 步骤2: 翻译字幕")
        print("uv run python srt_translator_agent.py \"生成的字幕文件.srt\" -s 英文 -t 中文")
        
        print("\n# 步骤3: 检查结果")
        print("在视频播放器中测试双语字幕效果")
        
    else:
        print("⚠️ 未找到音频文件，请先准备音频素材")

def main():
    """主函数"""
    print("🤖 SRT翻译Agent使用示例")
    print("基于LangGraph的智能字幕翻译系统")
    print("=" * 60)
    
    # 检查环境
    print("🔍 环境检查:")
    
    # 检查环境变量配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("MODEL_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    
    print(f"   OPENAI_API_KEY: {'✅ 已设置' if api_key else '❌ 未设置'}")
    print(f"   MODEL_BASE_URL: {'✅ 已设置' if base_url else '⚪ 使用默认'}")
    print(f"   MODEL_NAME: {model_name or 'gpt-4o-mini (默认)'}")
    
    if not api_key:
        print("\n💡 代理提供商配置示例:")
        print("   export OPENAI_API_KEY=your_api_key")
        print("   export MODEL_BASE_URL=https://your.proxy.com/v1")
        print("   export MODEL_NAME=gpt-4o-mini")
        print("💡 官方OpenAI API只需要设置OPENAI_API_KEY")
    
    # 检查依赖
    try:
        import langgraph
        print("✅ langgraph: 已安装")
    except ImportError:
        print("❌ langgraph: 未安装")
        print("💡 运行: uv add langgraph")
    
    try:
        import langchain_openai
        print("✅ langchain-openai: 已安装")
    except ImportError:
        print("❌ langchain-openai: 未安装")
        print("💡 运行: uv add langchain-openai")
    
    # 运行示例
    try:
        example_basic_translation()
        example_advanced_translation()
        example_multilingual()
        example_batch_processing()
        example_workflow_integration()
        
        print("\n🎉 所有示例演示完成!")
        print("💡 现在可以开始使用SRT翻译Agent了")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断示例")
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")

if __name__ == "__main__":
    main() 
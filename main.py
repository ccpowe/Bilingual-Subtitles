#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频转双语字幕主控制脚本
整合音频转录和翻译功能，支持完整工作流或单独运行
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import time
from dotenv import load_dotenv
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioToSubtitlePipeline:
    """音频转双语字幕流水线"""
    
    def __init__(self, config: dict):
        """
        初始化流水线
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志级别"""
        if self.config.get('verbose'):
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.config.get('quiet'):
            logging.getLogger().setLevel(logging.WARNING)
    
    def validate_config(self) -> bool:
        """验证配置"""
        logger.info("🔍 验证配置...")
        
        # 检查输入文件
        if self.config['mode'] in ['full', 'transcribe', 'full-embed']:
            input_file = self.config['input']
            if not os.path.exists(input_file):
                logger.error(f"❌ 输入文件不存在: {input_file}")
                return False
            
            # 检查文件格式
            valid_extensions = ['.mp3', '.mp4', '.wav', '.flac', '.m4a', '.avi', '.mov', '.mkv']
            if not any(input_file.lower().endswith(ext) for ext in valid_extensions):
                logger.warning(f"⚠️ 文件格式可能不支持: {input_file}")
        
        # 检查视频文件（用于字幕嵌入）
        if self.config['mode'] in ['embed', 'full-embed']:
            input_file = self.config['input']
            if not os.path.exists(input_file):
                logger.error(f"❌ 输入视频文件不存在: {input_file}")
                return False
                
            # 检查是否为视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            if not any(input_file.lower().endswith(ext) for ext in video_extensions):
                logger.warning(f"⚠️ 输入文件可能不是视频格式: {input_file}")
            
            # 检查字幕嵌入器依赖
            try:
                from video_subtitle_embedder import VideoSubtitleEmbedder
                logger.debug("✅ 字幕嵌入器模块可用")
            except ImportError:
                logger.error("❌ 字幕嵌入模块导入失败")
                return False
        
        # 检查翻译配置
        if self.config['mode'] in ['full', 'translate', 'full-embed']:
            api_key = self.config.get('api_key') or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("❌ 翻译功能需要设置OPENAI_API_KEY环境变量")
                return False
        
        # 检查依赖
        try:
            if self.config['mode'] in ['full', 'transcribe', 'full-embed']:
                import faster_whisper
                logger.debug("✅ faster-whisper 已安装")
            
            if self.config['mode'] in ['full', 'translate', 'full-embed']:
                import langgraph
                import langchain_openai
                logger.debug("✅ 翻译依赖已安装")
                
        except ImportError as e:
            logger.error(f"❌ 依赖缺失: {e}")
            return False
        
        logger.info("✅ 配置验证通过")
        return True
    
    def transcribe_audio(self, input_file: str) -> Optional[str]:
        """
        音频转录为SRT
        
        Args:
            input_file: 输入音频文件
            
        Returns:
            str: 输出SRT文件路径，失败返回None
        """
        logger.info(f"🎵 开始音频转录: {os.path.basename(input_file)}")
        
        try:
            # 动态导入避免不必要的依赖
            import faster_whisper
            import soundfile as sf
            import psutil
            import gc
            
            # 创建输出目录
            output_dir = Path("srt_file")
            output_dir.mkdir(exist_ok=True)
            
            # 生成输出文件名
            input_path = Path(input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{input_path.stem}_{timestamp}.srt"
            
            # 配置转录参数
            model_size = self.config.get('whisper_model', 'base')
            language = self.config.get('language')
            compute_type = self.config.get('compute_type', 'int8')
            cpu_threads = self.config.get('cpu_threads', 4)
            
            logger.info(f"🔧 转录配置:")
            logger.info(f"   模型: {model_size}")
            logger.info(f"   语言: {language or '自动检测'}")
            logger.info(f"   精度: {compute_type}")
            logger.info(f"   线程: {cpu_threads}")
            
            # 初始化模型
            logger.info("📥 加载Whisper模型...")
            model = faster_whisper.WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type=compute_type,
                cpu_threads=cpu_threads
            )
            
            # 智能提示词选择
            prompt = None
            if language == "zh":
                prompt = "以下是音频转录内容："
            elif language == "en":
                prompt = "The following is the audio transcription:"
            
            # 执行转录
            logger.info("🎯 开始转录...")
            segments, info = model.transcribe(
                input_file,
                language=language,
                initial_prompt=prompt,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            logger.info(f"📊 检测语言: {info.language}")
            logger.info(f"📊 语言概率: {info.language_probability:.2f}")
            
            # 生成SRT内容
            logger.info("📝 生成SRT字幕...")
            srt_content = []
            segment_count = 0
            
            for segment in segments:
                segment_count += 1
                start_time = self._format_timestamp(segment.start)
                end_time = self._format_timestamp(segment.end)
                text = segment.text.strip()
                
                srt_content.append(f"{segment_count}")
                srt_content.append(f"{start_time} --> {end_time}")
                srt_content.append(text)
                srt_content.append("")
                
                if segment_count % 10 == 0:
                    logger.debug(f"   已处理 {segment_count} 个片段")
            
            # 保存SRT文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            # 清理内存
            del model
            gc.collect()
            
            logger.info(f"✅ 转录完成!")
            logger.info(f"📁 字幕文件: {output_file}")
            logger.info(f"📊 总片段数: {segment_count}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ 转录失败: {e}")
            return None
    
    def translate_srt(self, srt_file: str) -> Optional[str]:
        """
        翻译SRT为双语字幕
        
        Args:
            srt_file: SRT文件路径
            
        Returns:
            str: 双语SRT文件路径，失败返回None
        """
        logger.info(f"🌐 开始翻译字幕: {os.path.basename(srt_file)}")
        
        try:
            # 动态导入翻译模块
            from srt_translator_agent import SRTTranslatorAgent
            
            # 配置翻译参数
            source_lang = self.config.get('source_lang', '英文')
            target_lang = self.config.get('target_lang', '中文')
            llm_model = self.config.get('llm_model')
            api_key = self.config.get('api_key')
            base_url = self.config.get('base_url')
            batch_size = self.config.get('batch_size', 5)
            
            logger.info(f"🔧 翻译配置:")
            logger.info(f"   源语言: {source_lang}")
            logger.info(f"   目标语言: {target_lang}")
            logger.info(f"   模型: {llm_model or os.getenv('MODEL_NAME') or 'gpt-4o-mini'}")
            logger.info(f"   批量大小: {batch_size}")
            
            # 创建翻译Agent
            agent = SRTTranslatorAgent(
                llm_model=llm_model,
                api_key=api_key,
                base_url=base_url,
                batch_size=batch_size
            )
            
            logger.info(f"🔧 LangGraph配置: recursion_limit=100 (解决递归限制问题)")
            
            # 执行翻译
            output_file = agent.translate_srt(
                input_file=srt_file,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            logger.info(f"✅ 翻译完成!")
            logger.info(f"📁 双语字幕: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ 翻译失败: {e}")
            return None
    
    def embed_subtitle_to_video(self, video_file: str, srt_file: str) -> Optional[str]:
        """
        嵌入字幕到视频
        
        Args:
            video_file: 视频文件路径
            srt_file: 字幕文件路径
            
        Returns:
            str: 输出视频文件路径，失败返回None
        """
        logger.info(f"🎬 开始嵌入字幕到视频: {os.path.basename(video_file)}")
        
        try:
            # 动态导入嵌入器模块
            from video_subtitle_embedder import VideoSubtitleEmbedder
            
            # 创建嵌入器
            embedder = VideoSubtitleEmbedder(processor=self.config.get('processor', 'auto'))
            
            # 嵌入字幕
            output_file = embedder.embed_subtitle(
                video_file=video_file,
                srt_file=srt_file,
                embed_type=self.config.get('embed_type', 'soft'),
                style_preset=self.config.get('style_preset', 'default')
            )
            
            logger.info(f"✅ 字幕嵌入完成!")
            logger.info(f"📁 输出视频: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ 字幕嵌入失败: {e}")
            return None
    
    def find_subtitle_file(self, video_file: str, subtitle_choice: str) -> Optional[str]:
        """
        查找对应的字幕文件
        
        Args:
            video_file: 视频文件路径
            subtitle_choice: 字幕类型选择 ('original', 'translation', 'bilingual')
            
        Returns:
            str: 字幕文件路径，未找到返回None
        """
        video_path = Path(video_file)
        srt_dir = Path("srt_file")
        
        if not srt_dir.exists():
            logger.warning("⚠️ srt_file 目录不存在")
            return None
        
        # 搜索模式：按优先级查找
        search_patterns = []
        
        if subtitle_choice == 'bilingual':
            search_patterns = [
                f"{video_path.stem}_bilingual_*.srt",
                f"{video_path.stem}_*.srt"
            ]
        elif subtitle_choice == 'translation':
            search_patterns = [
                f"{video_path.stem}_translation_*.srt",
                f"{video_path.stem}_bilingual_*.srt"
            ]
        elif subtitle_choice == 'original':
            search_patterns = [
                f"{video_path.stem}_[0-9]*.srt",  # 原始转录文件
                f"{video_path.stem}.srt"
            ]
        
        # 按优先级搜索文件
        for pattern in search_patterns:
            matching_files = list(srt_dir.glob(pattern))
            if matching_files:
                # 如果有多个文件，选择最新的
                latest_file = max(matching_files, key=os.path.getmtime)
                logger.info(f"📁 找到字幕文件: {latest_file.name}")
                return str(latest_file)
        
        logger.warning(f"⚠️ 未找到 {subtitle_choice} 类型的字幕文件")
        return None
    
    def run_full_pipeline(self) -> bool:
        """运行完整流水线"""
        logger.info("🚀 启动完整工作流: 音频 → SRT → 双语字幕")
        
        start_time = time.time()
        
        # 步骤1: 音频转录
        srt_file = self.transcribe_audio(self.config['input'])
        if not srt_file:
            return False
        
        # 步骤2: 翻译字幕
        bilingual_file = self.translate_srt(srt_file)
        if not bilingual_file:
            return False
        
        # 完成
        elapsed = time.time() - start_time
        logger.info(f"🎉 完整工作流完成! 耗时: {elapsed:.1f}秒")
        logger.info(f"📁 原始字幕: {srt_file}")
        logger.info(f"📁 双语字幕: {bilingual_file}")
        
        return True
    
    def run_transcribe_only(self) -> bool:
        """仅运行音频转录"""
        logger.info("🎵 启动转录模式: 音频 → SRT")
        
        srt_file = self.transcribe_audio(self.config['input'])
        return srt_file is not None
    
    def run_translate_only(self) -> bool:
        """仅运行字幕翻译"""
        logger.info("🌐 启动翻译模式: SRT → 双语字幕")
        
        srt_file = self.config['input']
        bilingual_file = self.translate_srt(srt_file)
        return bilingual_file is not None
    
    def run_embed_only(self) -> bool:
        """仅运行字幕嵌入"""
        logger.info("🎬 启动字幕嵌入模式: 视频 + SRT → 带字幕视频")
        
        video_file = self.config['input']
        
        # 获取字幕文件
        if self.config.get('srt_file'):
            # 用户指定了字幕文件
            srt_file = self.config['srt_file']
            if not os.path.exists(srt_file):
                logger.error(f"❌ 指定的字幕文件不存在: {srt_file}")
                return False
        else:
            # 自动查找字幕文件
            srt_file = self.find_subtitle_file(video_file, self.config.get('subtitle_choice', 'bilingual'))
            if not srt_file:
                logger.error("❌ 未找到对应的字幕文件")
                return False
        
        # 嵌入字幕
        output_file = self.embed_subtitle_to_video(video_file, srt_file)
        return output_file is not None
    
    def run_full_embed_pipeline(self) -> bool:
        """运行完整工作流+字幕嵌入"""
        logger.info("🚀 启动完整工作流+字幕嵌入: 音频 → SRT → 双语字幕 → 带字幕视频")
        
        start_time = time.time()
        video_file = self.config['input']
        
        # 步骤1: 音频转录
        srt_file = self.transcribe_audio(video_file)
        if not srt_file:
            return False
        
        # 步骤2: 翻译字幕
        bilingual_file = self.translate_srt(srt_file)
        if not bilingual_file:
            logger.warning("⚠️ 翻译失败，将使用原始字幕")
            bilingual_file = srt_file
        
        # 步骤3: 选择要嵌入的字幕
        subtitle_choice = self.config.get('subtitle_choice', 'bilingual')
        if subtitle_choice == 'bilingual' and bilingual_file != srt_file:
            embed_srt = bilingual_file
        elif subtitle_choice == 'original':
            embed_srt = srt_file
        else:
            # 尝试查找指定类型的字幕
            found_srt = self.find_subtitle_file(video_file, subtitle_choice)
            embed_srt = found_srt if found_srt else bilingual_file
        
        # 步骤4: 嵌入字幕到视频
        output_video = self.embed_subtitle_to_video(video_file, embed_srt)
        if not output_video:
            return False
        
        # 完成
        elapsed = time.time() - start_time
        logger.info(f"🎉 完整工作流+字幕嵌入完成! 耗时: {elapsed:.1f}秒")
        logger.info(f"📁 原始字幕: {srt_file}")
        logger.info(f"📁 双语字幕: {bilingual_file}")
        logger.info(f"📁 输出视频: {output_video}")
        
        return True
    
    def run(self) -> bool:
        """运行流水线"""
        if not self.validate_config():
            return False
        
        logger.info(f"🎬 音频转双语字幕流水线启动")
        logger.info(f"🔧 运行模式: {self.config['mode']}")
        
        try:
            if self.config['mode'] == 'full':
                return self.run_full_pipeline()
            elif self.config['mode'] == 'transcribe':
                return self.run_transcribe_only()
            elif self.config['mode'] == 'translate':
                return self.run_translate_only()
            elif self.config['mode'] == 'embed':
                return self.run_embed_only()
            elif self.config['mode'] == 'full-embed':
                return self.run_full_embed_pipeline()
            else:
                logger.error(f"❌ 未知运行模式: {self.config['mode']}")
                return False
                
        except KeyboardInterrupt:
            logger.warning("⏹️ 用户中断操作")
            return False
        except Exception as e:
            logger.error(f"❌ 流水线执行失败: {e}")
            return False
    
    def _format_timestamp(self, seconds: float) -> str:
        """格式化时间戳为SRT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="音频转双语字幕主控制脚本",
        epilog="""
运行模式:
  full        完整工作流 (音频 → SRT → 双语字幕)
  transcribe  仅音频转录 (音频 → SRT)
  translate   仅字幕翻译 (SRT → 双语字幕)
  embed       仅字幕嵌入 (视频 + SRT → 带字幕视频)
  full-embed  完整工作流+嵌入 (音频 → SRT → 双语字幕 → 带字幕视频)

示例:
  # 完整工作流
  uv run main.py video.mp4 --mode full -l en -s 英文 -t 中文
  
  # 完整工作流+字幕嵌入
  uv run main.py video.mp4 --mode full-embed -l en -s 英文 -t 中文 --subtitle-choice bilingual
  
  # 仅嵌入字幕
  uv run main.py video.mp4 --mode embed --subtitle-choice bilingual --srt-file subtitle.srt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 基础参数
    parser.add_argument('input', help='输入文件路径 (音频/视频/SRT)')
    parser.add_argument('--mode', choices=['full', 'transcribe', 'translate', 'embed', 'full-embed'], 
                       default='full', help='运行模式 (默认: full)')
    
    # 音频转录参数
    transcribe_group = parser.add_argument_group('音频转录参数')
    transcribe_group.add_argument('-m', '--whisper-model', default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                       help='Whisper模型大小 (默认: base)')
    transcribe_group.add_argument('-l', '--language', help='语言代码 (如: en, zh)')
    transcribe_group.add_argument('--compute-type', default='int8',
                       choices=['int8', 'float16', 'float32'],
                       help='计算精度 (默认: int8)')
    transcribe_group.add_argument('--cpu-threads', type=int, default=4,
                       help='CPU线程数 (默认: 4)')
    transcribe_group.add_argument('--auto-config', action='store_true',
                       help='自动配置转录参数')
    
    # 翻译参数
    translate_group = parser.add_argument_group('翻译参数')
    translate_group.add_argument('-s', '--source-lang', default='英文',
                       help='源语言 (默认: 英文)')
    translate_group.add_argument('-t', '--target-lang', default='中文',
                       help='目标语言 (默认: 中文)')
    translate_group.add_argument('--llm-model', help='LLM模型 (覆盖环境变量)')
    translate_group.add_argument('--api-key', help='API密钥 (覆盖环境变量)')
    translate_group.add_argument('--base-url', help='API基础URL (覆盖环境变量)')
    translate_group.add_argument('-b', '--batch-size', type=int, default=5,
                       help='翻译批量大小 (默认: 5)')
    
    # 字幕嵌入参数
    embed_group = parser.add_argument_group('字幕嵌入参数')
    embed_group.add_argument('--subtitle-choice', 
                            choices=['original', 'translation', 'bilingual'],
                            default='bilingual',
                            help='选择嵌入的字幕类型 (默认: bilingual)')
    embed_group.add_argument('--embed-type',
                            choices=['soft', 'hard'],
                            default='soft',
                            help='字幕嵌入方式 (默认: soft)')
    embed_group.add_argument('--style-preset',
                            default='default',
                            help='字幕样式预设 (默认: default)')
    embed_group.add_argument('--processor',
                            choices=['auto', 'ffmpeg', 'moviepy'],
                            default='auto',
                            help='视频处理器选择 (默认: auto)')
    embed_group.add_argument('--srt-file',
                            help='指定字幕文件路径 (embed模式使用)')
    
    # 控制参数
    control_group = parser.add_argument_group('控制参数')
    control_group.add_argument('-v', '--verbose', action='store_true',
                       help='详细输出')
    control_group.add_argument('-q', '--quiet', action='store_true',
                       help='安静模式')
    control_group.add_argument('--dry-run', action='store_true',
                       help='空运行 (仅验证配置)')
    
    args = parser.parse_args()
    
    # 构建配置字典
    config = {
        'input': args.input,
        'mode': args.mode,
        'whisper_model': args.whisper_model,
        'language': args.language,
        'compute_type': args.compute_type,
        'cpu_threads': args.cpu_threads,
        'auto_config': args.auto_config,
        'source_lang': args.source_lang,
        'target_lang': args.target_lang,
        'llm_model': args.llm_model,
        'api_key': args.api_key,
        'base_url': args.base_url,
        'batch_size': args.batch_size,
        'subtitle_choice': args.subtitle_choice,
        'embed_type': args.embed_type,
        'style_preset': args.style_preset,
        'processor': args.processor,
        'srt_file': args.srt_file,
        'verbose': args.verbose,
        'quiet': args.quiet,
        'dry_run': args.dry_run
    }
    
    # 自动配置处理
    if args.auto_config and args.mode in ['full', 'transcribe', 'full-embed']:
        logger.info("🔧 启用自动配置...")
        try:
            import psutil
            
            # 获取系统信息
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # 自动推荐配置
            if memory_gb < 8:
                config['whisper_model'] = 'tiny'
                config['cpu_threads'] = 2
                logger.info(f"   自动配置: {memory_gb:.1f}GB内存 → 使用tiny模型")
            elif memory_gb < 16:
                config['whisper_model'] = 'base'
                config['cpu_threads'] = 4
                logger.info(f"   自动配置: {memory_gb:.1f}GB内存 → 使用base模型")
            else:
                config['whisper_model'] = 'small'
                config['cpu_threads'] = 6
                logger.info(f"   自动配置: {memory_gb:.1f}GB内存 → 使用small模型")
                
        except ImportError:
            logger.warning("⚠️ 自动配置需要psutil库")
    
    # 显示配置信息
    print("\n" + "="*60)
    print("🎬 音频转双语字幕主控制脚本")
    print("="*60)
    print(f"📁 输入文件: {config['input']}")
    print(f"🔧 运行模式: {config['mode']}")
    
    if config['mode'] in ['full', 'transcribe', 'full-embed']:
        print(f"\n🎵 转录配置:")
        print(f"   模型: {config['whisper_model']}")
        print(f"   语言: {config['language'] or '自动检测'}")
        print(f"   精度: {config['compute_type']}")
        print(f"   线程: {config['cpu_threads']}")
    
    if config['mode'] in ['full', 'translate', 'full-embed']:
        print(f"\n🌐 翻译配置:")
        print(f"   {config['source_lang']} → {config['target_lang']}")
        print(f"   模型: {config['llm_model'] or os.getenv('MODEL_NAME') or 'gpt-4o-mini'}")
        print(f"   批量: {config['batch_size']}")
        
        # 环境变量检查
        api_key = config['api_key'] or os.getenv("OPENAI_API_KEY")
        base_url = config['base_url'] or os.getenv("MODEL_BASE_URL")
        print(f"   API: {'✅' if api_key else '❌'}")
        print(f"   代理: {'✅' if base_url else '⚪'}")
    
    if config['mode'] in ['embed', 'full-embed']:
        print(f"\n🎬 字幕嵌入配置:")
        print(f"   字幕类型: {config['subtitle_choice']}")
        print(f"   嵌入方式: {config['embed_type']}")
        print(f"   样式预设: {config['style_preset']}")
        print(f"   处理器: {config['processor']}")
        if config['srt_file']:
            print(f"   指定字幕: {config['srt_file']}")
    
    print("="*60)
    
    # 空运行模式
    if config['dry_run']:
        print("\n🔍 空运行模式 - 仅验证配置")
        pipeline = AudioToSubtitlePipeline(config)
        success = pipeline.validate_config()
        print(f"\n✅ 配置验证{'通过' if success else '失败'}")
        return 0 if success else 1
    
    # 创建并运行流水线
    pipeline = AudioToSubtitlePipeline(config)
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 
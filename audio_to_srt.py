#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频转SRT字幕生成器
直接使用Whisper模型将音频文件转换为SRT字幕文件
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioToSRT:
    """音频转SRT字幕生成器"""
    
    def __init__(self, model_size="small", device="cpu", compute_type="float16"):
        """
        初始化转换器
        
        Args:
            model_size: 模型大小 (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: 设备 (cpu, cuda)
            compute_type: 计算类型 (float32, float16, int8)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        logger.info(f"初始化音频转SRT转换器:")
        logger.info(f"  模型: {model_size}")
        logger.info(f"  设备: {device}")
        logger.info(f"  计算类型: {compute_type}")
    
    def load_model(self):
        """加载Whisper模型"""
        try:
            logger.info("正在加载Whisper模型...")
            start_time = time.time()
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ 模型加载成功 ({load_time:.2f}秒)")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def transcribe_to_srt(self, audio_file, output_file=None, language=None, initial_prompt=None):
        """
        将音频文件转录为SRT字幕
        
        Args:
            audio_file: 输入音频文件路径
            output_file: 输出SRT文件路径（可选）
            language: 指定语言（可选，如 'zh', 'en'）
            initial_prompt: 初始提示词（可选）
        
        Returns:
            str: 输出文件路径
        """
        if self.model is None:
            logger.error("❌ 模型未加载，请先调用 load_model()")
            return None
        
        # 检查输入文件
        if not os.path.exists(audio_file):
            logger.error(f"❌ 音频文件不存在: {audio_file}")
            return None
        
        # 生成输出文件名（带时间戳避免覆盖）
        if output_file is None:
            # 创建输出目录
            output_dir = Path("srt_file")
            output_dir.mkdir(exist_ok=True)
            
            # 生成输出文件名
            audio_path = Path(audio_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = audio_path.stem
            output_file = output_dir / f"{base_name}_{timestamp}.srt"
        
        try:
            logger.info(f"开始转录音频文件: {audio_file}")
            start_time = time.time()
            
            # 执行转录
            # 根据语言选择合适的提示词
            if initial_prompt is None:
                if language == "zh":
                    initial_prompt = "以下是音频转录内容："
                elif language == "en":
                    initial_prompt = "The following is an English audio transcription:"
                else:
                    initial_prompt = None  # 让模型自动判断
            
            segments, info = self.model.transcribe(
                audio_file,
                language=language,
                initial_prompt=initial_prompt
            )
            
            # 将segments转为列表（因为是生成器）
            segment_list = list(segments)
            
            # 生成SRT内容
            srt_content = self._generate_srt_content(segment_list)
            
            # 写入SRT文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            process_time = time.time() - start_time
            
            logger.info(f"✅ 转录完成!")
            logger.info(f"  输出文件: {output_file}")
            logger.info(f"  处理时间: {process_time:.2f}秒")
            logger.info(f"  检测语言: {info.language} (置信度: {info.language_probability:.2f})")
            logger.info(f"  字幕段数: {len(segment_list)}")
            logger.info(f"  音频时长: {info.duration:.2f}秒" if hasattr(info, 'duration') else "")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ 转录失败: {e}")
            return None
    
    def _generate_srt_content(self, segments):
        """生成SRT格式内容"""
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            # 时间格式转换
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            
            # SRT格式
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment.text.strip())
            srt_lines.append("")  # 空行分隔
        
        return "\n".join(srt_lines)
    
    def _seconds_to_srt_time(self, seconds):
        """将秒数转换为SRT时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def batch_convert(self, input_dir, output_dir=None, language=None, 
                     audio_extensions=None):
        """
        批量转换目录中的音频文件
        
        Args:
            input_dir: 输入音频文件目录
            output_dir: 输出SRT文件目录（可选）
            language: 指定语言（可选）
            audio_extensions: 音频文件扩展名列表（可选）
        """
        if audio_extensions is None:
            audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.mp4', '.mkv', '.avi']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"❌ 输入目录不存在: {input_dir}")
            return
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path
        
        # 查找音频文件
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"*{ext}"))
            audio_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            logger.warning(f"⚠️ 在目录中未找到音频文件: {input_dir}")
            return
        
        logger.info(f"找到 {len(audio_files)} 个音频文件，开始批量转换...")
        
        successful = 0
        failed = 0
        
        for audio_file in audio_files:
            try:
                logger.info(f"\n正在处理: {audio_file.name}")
                
                # 生成输出文件路径
                output_file = output_path / f"{audio_file.stem}.srt"
                
                # 执行转换
                result = self.transcribe_to_srt(
                    str(audio_file),
                    str(output_file),
                    language=language
                )
                
                if result:
                    successful += 1
                    logger.info(f"  ✅ 成功: {output_file.name}")
                else:
                    failed += 1
                    logger.error(f"  ❌ 失败: {audio_file.name}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"  ❌ 处理失败: {e}")
        
        logger.info(f"\n📊 批量转换完成:")
        logger.info(f"  成功: {successful} 个")
        logger.info(f"  失败: {failed} 个")


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='音频转SRT字幕生成器')
    parser.add_argument('input', help='输入音频文件或目录')
    parser.add_argument('-o', '--output', help='输出SRT文件或目录')
    parser.add_argument('-m', '--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'],
                       help='Whisper模型大小 (默认: base)')
    parser.add_argument('-l', '--language', help='指定语言代码 (如: zh, en)')
    parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda'],
                       help='计算设备 (默认: cpu)')
    parser.add_argument('--compute-type', default='float32', 
                       choices=['float32', 'float16', 'int8'],
                       help='计算类型 (默认: float32)')
    parser.add_argument('--batch', action='store_true', help='批量处理目录')
    parser.add_argument('--prompt', help='初始提示词')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = AudioToSRT(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type
    )
    
    # 加载模型
    if not converter.load_model():
        logger.error("❌ 模型加载失败，退出程序")
        return 1
    
    try:
        if args.batch or os.path.isdir(args.input):
            # 批量处理
            converter.batch_convert(
                input_dir=args.input,
                output_dir=args.output,
                language=args.language
            )
        else:
            # 单文件处理
            result = converter.transcribe_to_srt(
                audio_file=args.input,
                output_file=args.output,
                language=args.language,
                initial_prompt=args.prompt
            )
            
            if result:
                logger.info(f"🎉 字幕文件已生成: {result}")
            else:
                logger.error("❌ 字幕生成失败")
                return 1
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ 用户中断操作")
        return 0
    except Exception as e:
        logger.error(f"❌ 程序执行出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
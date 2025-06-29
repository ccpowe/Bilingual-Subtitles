#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频字幕嵌入器
支持将SRT字幕嵌入到视频文件中，提供软字幕和硬字幕两种模式
优先使用系统 FFmpeg，降级到 MoviePy
"""

import os
import sys
import json
import subprocess
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoSubtitleEmbedder:
    """视频字幕嵌入器"""
    
    def __init__(self, processor: str = "auto"):
        """
        初始化嵌入器
        
        Args:
            processor: 处理器选择 ("auto", "ffmpeg", "moviepy")
        """
        self.processor = self._detect_processor(processor)
        self.styles = self._load_styles()
        
        logger.info(f"🎬 视频字幕嵌入器初始化")
        logger.info(f"🔧 使用处理器: {self.processor}")
    
    def _detect_processor(self, preference: str) -> str:
        """检测可用的处理器"""
        if preference == "moviepy":
            return self._check_moviepy()
        elif preference == "ffmpeg":
            return self._check_ffmpeg()
        else:  # auto
            # 优先 FFmpeg，降级到 MoviePy
            ffmpeg_processor = self._check_ffmpeg()
            if ffmpeg_processor == "ffmpeg":
                return ffmpeg_processor
            else:
                return self._check_moviepy()
    
    def _check_ffmpeg(self) -> str:
        """检查系统 FFmpeg 是否可用"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                logger.debug("✅ 系统 FFmpeg 可用")
                return "ffmpeg"
            else:
                logger.warning("⚠️ FFmpeg 命令执行失败")
                return self._suggest_ffmpeg_installation()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            logger.warning("⚠️ 系统未安装 FFmpeg")
            return self._suggest_ffmpeg_installation()
    
    def _check_moviepy(self) -> str:
        """检查 MoviePy 是否可用"""
        try:
            import moviepy.editor as mp
            logger.debug("✅ MoviePy 可用")
            return "moviepy"
        except ImportError:
            logger.error("❌ MoviePy 未安装，请安装: uv add moviepy")
            raise ImportError("需要安装 MoviePy: uv add moviepy")
    
    def _suggest_ffmpeg_installation(self) -> str:
        """提供 FFmpeg 安装建议并降级到 MoviePy"""
        logger.warning("💡 建议安装 FFmpeg 以获得最佳性能:")
        
        # 检测操作系统
        import platform
        system = platform.system().lower()
        
        if system == "windows":
            logger.warning("   Windows: winget install ffmpeg")
        elif system == "darwin":  # macOS
            logger.warning("   macOS: brew install ffmpeg")
        else:  # Linux
            logger.warning("   Linux: apt install ffmpeg 或 yum install ffmpeg")
        
        logger.warning("📦 正在降级使用 MoviePy...")
        return self._check_moviepy()
    
    def _load_styles(self) -> Dict[str, Any]:
        """加载字幕样式配置"""
        styles_file = Path(__file__).parent / "subtitle_styles.json"
        
        # 默认样式配置
        default_styles = {
            "default": {
                "font_size": 20,
                "font_color": "white",
                "outline_color": "black",
                "position": "bottom_center",
                "margin_v": 20,
                "description": "默认白色字幕，黑色边框"
            },
            "cinema": {
                "font_size": 24,
                "font_color": "yellow",
                "outline_color": "black", 
                "position": "bottom_center",
                "margin_v": 25,
                "description": "电影院风格黄色字幕"
            },
            "simple": {
                "font_size": 18,
                "font_color": "white",
                "outline_color": "none",
                "position": "bottom_center", 
                "margin_v": 15,
                "description": "简洁白色字幕，无边框"
            }
        }
        
        try:
            if styles_file.exists():
                with open(styles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 创建默认样式文件
                with open(styles_file, 'w', encoding='utf-8') as f:
                    json.dump(default_styles, f, indent=2, ensure_ascii=False)
                logger.info(f"📝 创建默认样式配置: {styles_file}")
                return default_styles
                
        except Exception as e:
            logger.warning(f"⚠️ 样式配置加载失败，使用默认样式: {e}")
            return default_styles
    

    
    def embed_subtitle(self, 
                      video_file: str,
                      srt_file: str,
                      output_file: Optional[str] = None,
                      embed_type: str = "soft",
                      style_preset: str = "default") -> str:
        """
        嵌入字幕到视频
        
        Args:
            video_file: 输入视频文件路径
            srt_file: SRT字幕文件路径
            output_file: 输出视频文件路径（可选）
            embed_type: 嵌入类型 ("soft" 或 "hard")
            style_preset: 样式预设名称
        
        Returns:
            str: 输出视频文件路径
        """
        # 验证输入文件
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"视频文件不存在: {video_file}")
        
        if not os.path.exists(srt_file):
            raise FileNotFoundError(f"字幕文件不存在: {srt_file}")
        
        # 生成输出文件名
        if output_file is None:
            output_file = self._generate_output_filename(video_file, embed_type)
        
        # 创建输出目录
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎬 开始嵌入字幕:")
        logger.info(f"   视频: {os.path.basename(video_file)}")
        logger.info(f"   字幕: {os.path.basename(srt_file)}")
        logger.info(f"   类型: {embed_type}")
        logger.info(f"   样式: {style_preset}")
        logger.info(f"   处理器: {self.processor}")
        
        # 根据处理器选择实现方式
        if self.processor == "ffmpeg":
            return self._embed_with_ffmpeg(video_file, srt_file, output_file, embed_type, style_preset)
        else:  # moviepy
            return self._embed_with_moviepy(video_file, srt_file, output_file, embed_type, style_preset)
    
    def _generate_output_filename(self, video_file: str, embed_type: str) -> str:
        """生成输出文件名"""
        video_path = Path(video_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        output_dir = Path("video_output")
        output_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        suffix = "embedded" if embed_type == "soft" else "hardcoded"
        output_file = output_dir / f"{video_path.stem}_{suffix}_{timestamp}{video_path.suffix}"
        
        return str(output_file)
    
    def _embed_with_ffmpeg(self, video_file: str, srt_file: str, 
                          output_file: str, embed_type: str, style_preset: str) -> str:
        """使用系统 FFmpeg 嵌入字幕"""
        try:
            if embed_type == "soft":
                # 软字幕：快速复制，无需重编码
                cmd = [
                    "ffmpeg", "-y",  # -y 覆盖输出文件
                    "-i", video_file,
                    "-i", srt_file,
                    "-c:v", "copy",         # 复制视频流
                    "-c:a", "copy",         # 复制音频流
                    "-c:s", "mov_text",     # 字幕编码格式 (MP4容器)
                    "-map", "0:v",          # 映射视频流
                    "-map", "0:a?",         # 映射音频流（如果存在）
                    "-map", "1:s",          # 映射字幕流
                    "-metadata:s:s:0", f"title={style_preset}",
                    "-disposition:s:0", "default",  # 设置字幕为默认显示
                    output_file
                ]
                logger.info("🚀 使用 FFmpeg 嵌入软字幕（快速模式）")
                
            else:  # hard
                # 硬字幕：需要重编码，使用字幕滤镜
                style = self.styles.get(style_preset, self.styles["default"])
                
                # Windows路径需要特殊处理：转换为正斜杠或转义反斜杠
                srt_path_for_ffmpeg = srt_file.replace('\\', '/').replace(':', '\\:')
                subtitle_filter = f"subtitles='{srt_path_for_ffmpeg}'"
                
                # 构建样式参数
                force_style_params = []
                
                if style.get("font_size"):
                    force_style_params.append(f"FontSize={style['font_size']}")
                
                if style.get("font_color"):
                    force_style_params.append(f"PrimaryColour=&H{self._color_to_hex(style['font_color'])}")
                
                if style.get("outline_color") and style["outline_color"] != "none":
                    force_style_params.append(f"OutlineColour=&H{self._color_to_hex(style['outline_color'])}")
                
                if style.get("margin_v"):
                    force_style_params.append(f"MarginV={style['margin_v']}")
                
                # 如果有背景设置
                if style.get("background") == "semi_transparent":
                    force_style_params.append(f"BackColour=&H80000000")  # 半透明黑色背景
                    force_style_params.append(f"BorderStyle=3")  # 背景框样式
                
                # 组装完整的force_style参数
                if force_style_params:
                    subtitle_filter += f":force_style='{','.join(force_style_params)}'"
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_file,
                    "-vf", subtitle_filter,
                    "-c:a", "copy",  # 复制音频流
                    output_file
                ]
                logger.info("🎞️ 使用 FFmpeg 嵌入硬字幕（重编码模式）")
            
            # 执行 FFmpeg 命令
            logger.debug(f"执行命令: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if process.returncode == 0:
                logger.info(f"✅ FFmpeg 处理完成: {os.path.basename(output_file)}")
                return output_file
            else:
                error_msg = f"FFmpeg 执行失败: {process.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"❌ FFmpeg 嵌入失败: {e}")
            raise
    
    def _embed_with_moviepy(self, video_file: str, srt_file: str,
                           output_file: str, embed_type: str, style_preset: str) -> str:
        """使用 MoviePy 嵌入字幕"""
        try:
            import moviepy.editor as mp
            from moviepy.video.tools.subtitles import SubtitlesClip
            
            logger.info("📦 使用 MoviePy 处理视频字幕")
            
            # 加载视频
            video = mp.VideoFileClip(video_file)
            
            if embed_type == "soft":
                # MoviePy 不直接支持软字幕，转为硬字幕处理
                logger.warning("⚠️ MoviePy 不支持软字幕，自动转为硬字幕模式")
                embed_type = "hard"
            
            # 硬字幕处理
            style = self.styles.get(style_preset, self.styles["default"])
            
            # 创建字幕剪辑
            def generator(txt):
                return mp.TextClip(
                    txt,
                    fontsize=style.get("font_size", 20),
                    color=style.get("font_color", "white"),
                    stroke_color=style.get("outline_color", "black") if style.get("outline_color") != "none" else None,
                    stroke_width=1 if style.get("outline_color") != "none" else 0
                ).set_position(('center', 'bottom')).set_duration(1)
            
            # 加载字幕
            subtitles = SubtitlesClip(srt_file, generator)
            
            # 合成视频
            final_video = mp.CompositeVideoClip([video, subtitles])
            
            # 导出视频
            logger.info("🎬 开始视频导出...")
            final_video.write_videofile(
                output_file,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # 清理资源
            video.close()
            final_video.close()
            subtitles.close()
            
            logger.info(f"✅ MoviePy 处理完成: {os.path.basename(output_file)}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ MoviePy 嵌入失败: {e}")
            raise
    
    def _color_to_hex(self, color: str) -> str:
        """将颜色名称转换为 FFmpeg 支持的十六进制格式"""
        color_map = {
            "white": "FFFFFF",
            "black": "000000", 
            "red": "FF0000",
            "green": "00FF00",
            "blue": "0000FF",
            "yellow": "FFFF00",
            "cyan": "00FFFF",
            "magenta": "FF00FF"
        }
        
        color_lower = color.lower()
        if color_lower in color_map:
            return color_map[color_lower]
        elif color.startswith("#"):
            return color[1:]  # 移除 # 号
        else:
            return "FFFFFF"  # 默认白色


if __name__ == "__main__":
    # 简单的测试代码
    embedder = VideoSubtitleEmbedder()
    print("视频字幕嵌入器模块创建成功！") 
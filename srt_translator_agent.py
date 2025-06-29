#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT翻译Agent - 基于LangGraph
将SRT字幕文件翻译为双语字幕
"""

import os
import re
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass

# LangGraph相关导入
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain相关导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SRTEntry:
    """SRT字幕条目"""
    index: int
    start_time: str
    end_time: str
    text: str
    translation: str = ""

class TranslationState(TypedDict):
    """翻译Agent状态"""
    input_file: str
    output_file: str
    source_lang: str
    target_lang: str
    entries: List[SRTEntry]
    current_index: int
    total_entries: int
    translation_progress: float
    errors: List[str]
    llm_model: str
    batch_size: int
    max_retries: int
    bilingual_content: str

class SRTTranslatorAgent:
    """基于LangGraph的SRT翻译Agent"""
    
    def __init__(self, 
                 llm_model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 batch_size: int = 5,
                 max_retries: int = 3):
        """
        初始化翻译Agent
        
        Args:
            llm_model: 使用的语言模型（可通过MODEL_NAME环境变量设置）
            api_key: API密钥（可通过OPENAI_API_KEY环境变量设置）
            base_url: API基础URL（可通过MODEL_BASE_URL环境变量设置）
            batch_size: 批量翻译大小
            max_retries: 最大重试次数
        """
        # 从环境变量获取配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("MODEL_BASE_URL")
        self.llm_model = llm_model or os.getenv("MODEL_NAME") or "gpt-4o-mini"
        
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # 初始化LLM参数
        llm_kwargs = {
            "model": self.llm_model,
            "api_key": self.api_key,
            "temperature": 0.3
        }
        
        # 如果有base_url配置，添加到参数中
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        
        # 初始化LLM
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # 初始化LangGraph
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        logger.info(f"🤖 SRT翻译Agent已初始化:")
        logger.info(f"   模型: {llm_model}")
        logger.info(f"   批量大小: {batch_size}")
        logger.info(f"   最大重试: {max_retries}")
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(TranslationState)
        
        # 添加节点
        workflow.add_node("parse_srt", self.parse_srt_node)
        workflow.add_node("translate_batch", self.translate_batch_node)
        workflow.add_node("validate_translation", self.validate_translation_node)
        workflow.add_node("generate_bilingual_srt", self.generate_bilingual_srt_node)
        workflow.add_node("save_result", self.save_result_node)
        
        # 设置入口点
        workflow.set_entry_point("parse_srt")
        
        # 添加边
        workflow.add_edge("parse_srt", "translate_batch")
        workflow.add_conditional_edges(
            "translate_batch",
            self.should_continue_translation,
            {
                "continue": "translate_batch",
                "validate": "validate_translation"
            }
        )
        workflow.add_edge("validate_translation", "generate_bilingual_srt")
        workflow.add_edge("generate_bilingual_srt", "save_result")
        workflow.add_edge("save_result", END)
        
        return workflow
    
    def parse_srt_node(self, state: TranslationState) -> TranslationState:
        """解析SRT文件节点"""
        logger.info(f"📖 开始解析SRT文件: {state['input_file']}")
        
        try:
            entries = self._parse_srt_file(state['input_file'])
            state['entries'] = entries
            state['total_entries'] = len(entries)
            state['current_index'] = 0
            state['translation_progress'] = 0.0
            
            logger.info(f"✅ SRT解析完成，共 {len(entries)} 条字幕")
            
        except Exception as e:
            error_msg = f"SRT文件解析失败: {e}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
        
        return state
    
    def translate_batch_node(self, state: TranslationState) -> TranslationState:
        """批量翻译节点"""
        start_idx = state['current_index']
        end_idx = min(start_idx + state['batch_size'], state['total_entries'])
        
        logger.info(f"🔄 翻译批次 {start_idx + 1}-{end_idx} / {state['total_entries']}")
        
        # 准备批量翻译
        batch_entries = state['entries'][start_idx:end_idx]
        texts_to_translate = [entry.text for entry in batch_entries]
        
        try:
            # 执行批量翻译
            translations = self._batch_translate(
                texts_to_translate,
                state['source_lang'],
                state['target_lang']
            )
            
            # 更新翻译结果
            for i, translation in enumerate(translations):
                state['entries'][start_idx + i].translation = translation
            
            # 更新进度
            state['current_index'] = end_idx
            state['translation_progress'] = end_idx / state['total_entries']
            
            logger.info(f"✅ 批次翻译完成，进度: {state['translation_progress']:.1%}")
            
        except Exception as e:
            error_msg = f"批量翻译失败: {e}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
        
        return state
    
    def should_continue_translation(self, state: TranslationState) -> str:
        """判断是否继续翻译"""
        if state['current_index'] < state['total_entries']:
            return "continue"
        else:
            return "validate"
    
    def validate_translation_node(self, state: TranslationState) -> TranslationState:
        """验证翻译质量节点"""
        logger.info("🔍 验证翻译质量...")
        
        empty_translations = 0
        for entry in state['entries']:
            if not entry.translation.strip():
                empty_translations += 1
        
        if empty_translations > 0:
            logger.warning(f"⚠️ 发现 {empty_translations} 条空翻译")
            state['errors'].append(f"存在 {empty_translations} 条空翻译")
        else:
            logger.info("✅ 翻译质量验证通过")
        
        return state
    
    def generate_bilingual_srt_node(self, state: TranslationState) -> TranslationState:
        """生成双语SRT节点"""
        logger.info("📝 生成双语SRT内容...")
        
        try:
            bilingual_content = self._generate_bilingual_content(state['entries'])
            state['bilingual_content'] = bilingual_content
            logger.info("✅ 双语SRT内容生成完成")
            
        except Exception as e:
            error_msg = f"双语SRT生成失败: {e}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
        
        return state
    
    def save_result_node(self, state: TranslationState) -> TranslationState:
        """保存结果节点"""
        logger.info(f"💾 保存双语SRT文件: {state['output_file']}")
        
        try:
            with open(state['output_file'], 'w', encoding='utf-8') as f:
                f.write(state['bilingual_content'])
            
            logger.info(f"✅ 双语SRT文件已保存: {state['output_file']}")
            
        except Exception as e:
            error_msg = f"文件保存失败: {e}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
        
        return state
    
    def _parse_srt_file(self, file_path: str) -> List[SRTEntry]:
        """解析SRT文件"""
        entries = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # SRT格式正则表达式
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\n*$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            index = int(match[0])
            start_time = match[1]
            end_time = match[2]
            text = match[3].strip().replace('\n', ' ')
            
            entries.append(SRTEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text
            ))
        
        return entries
    
    def _batch_translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """批量翻译文本"""
        # 创建翻译提示模板
        system_prompt = f"""你是一个专业的字幕翻译师。请将以下{source_lang}字幕翻译为{target_lang}。

要求：
1. 保持原意准确，语言自然流畅
2. 适合口语化表达
3. 保持简洁，适合字幕显示
4. 专业术语请准确翻译
5. 每行翻译对应一行原文

请按照以下格式输出，每行一个翻译结果：
1. [第一行翻译]
2. [第二行翻译]
...
"""
        
        # 格式化输入文本
        numbered_texts = [f"{i+1}. {text}" for i, text in enumerate(texts)]
        user_input = "\n".join(numbered_texts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # 执行翻译
        chain = prompt | self.llm
        response = chain.invoke({"input": user_input})
        
        # 解析翻译结果
        response_content = str(response.content if hasattr(response, 'content') else response)
        translations = self._parse_translation_response(response_content, len(texts))
        
        return translations
    
    def _parse_translation_response(self, response: str, expected_count: int) -> List[str]:
        """解析翻译响应"""
        lines = response.strip().split('\n')
        translations = []
        
        for line in lines:
            # 移除序号
            cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
            if cleaned:
                translations.append(cleaned)
        
        # 确保翻译数量正确
        while len(translations) < expected_count:
            translations.append("[翻译失败]")
        
        return translations[:expected_count]
    
    def _generate_bilingual_content(self, entries: List[SRTEntry]) -> str:
        """生成双语SRT内容"""
        content_lines = []
        
        for entry in entries:
            content_lines.append(str(entry.index))
            content_lines.append(f"{entry.start_time} --> {entry.end_time}")
            content_lines.append(entry.text)
            content_lines.append(entry.translation)
            content_lines.append("")  # 空行分隔
        
        return "\n".join(content_lines)
    
    def translate_srt(self, 
                      input_file: str,
                      output_file: Optional[str] = None,
                      source_lang: str = "英文",
                      target_lang: str = "中文") -> str:
        """
        翻译SRT文件
        
        Args:
            input_file: 输入SRT文件路径
            output_file: 输出文件路径（可选）
            source_lang: 源语言
            target_lang: 目标语言
        
        Returns:
            str: 输出文件路径
        """
        # 生成输出文件名
        if output_file is None:
            input_path = Path(input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(input_path.parent / f"{input_path.stem}_bilingual_{timestamp}.srt")
        
        # 初始化状态
        initial_state = TranslationState(
            input_file=input_file,
            output_file=str(output_file),
            source_lang=source_lang,
            target_lang=target_lang,
            entries=[],
            current_index=0,
            total_entries=0,
            translation_progress=0.0,
            errors=[],
            llm_model=self.llm_model,
            batch_size=self.batch_size,
            max_retries=self.max_retries,
            bilingual_content=""
        )
        
        # 运行工作流
        logger.info(f"🚀 开始翻译SRT文件: {input_file}")
        logger.info(f"📝 源语言: {source_lang} → 目标语言: {target_lang}")
        
        try:
            # 执行翻译工作流 - 增加递归限制解决LangGraph限制问题
            result = self.app.invoke(
                initial_state,
                config={
                    "recursion_limit": 100,  # 增加递归限制，解决LangGraph限制问题
                    "configurable": {"thread_id": f"translation_{int(time.time())}"}
                }
            )
            
            if result['errors']:
                logger.warning("⚠️ 翻译过程中出现错误:")
                for error in result['errors']:
                    logger.warning(f"   - {error}")
            else:
                logger.info("🎉 翻译完成!")
            
            logger.info(f"📁 双语字幕文件: {result['output_file']}")
            return result['output_file']
            
        except Exception as e:
            logger.error(f"❌ 翻译失败: {e}")
            raise


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='SRT字幕翻译Agent')
    parser.add_argument('input', help='输入SRT文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('-s', '--source-lang', default='英文', help='源语言')
    parser.add_argument('-t', '--target-lang', default='中文', help='目标语言')
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo', 
                       help='LLM模型')
    parser.add_argument('-b', '--batch-size', type=int, default=5,
                       help='批量翻译大小')
    parser.add_argument('--api-key', help='OpenAI API密钥')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"❌ 输入文件不存在: {args.input}")
        return 1
    
    # 检查API密钥
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ 请设置OPENAI_API_KEY环境变量或使用--api-key参数")
        return 1
    
    try:
        # 创建翻译Agent
        agent = SRTTranslatorAgent(
            llm_model=args.model,
            api_key=api_key,
            batch_size=args.batch_size
        )
        
        # 执行翻译
        output_file = agent.translate_srt(
            input_file=args.input,
            output_file=args.output,
            source_lang=args.source_lang,
            target_lang=args.target_lang
        )
        
        logger.info(f"🎉 翻译完成! 输出文件: {output_file}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 
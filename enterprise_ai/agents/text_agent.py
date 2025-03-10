"""
文本处理Agent - 专门处理文本相关任务的智能体
集成RAG检索能力和LLM推理，可以回答问题、总结文档和执行文本分析
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import json

# 导入基类
from .base_agent import BaseAgent

# 导入MCP协议组件
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.controller import MessageType, TaskStatus, AgentCapability

# 导入RAG检索器
from rag.retriever import RAGRetriever

# 导入LangChain组件
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextAgent(BaseAgent):
    """专门处理文本任务的Agent"""
    
    def __init__(
        self,
        name: str,
        controller_reference,
        rag_retriever: Optional[RAGRetriever] = None,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        rag_k: int = 5,
        hybrid_search: bool = True,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # 初始化基类
        capabilities = [
            AgentCapability.TEXT_PROCESSING,
            AgentCapability.REASONING
        ]
        
        super().__init__(
            name=name,
            capabilities=capabilities,
            controller_reference=controller_reference,
            agent_id=agent_id,
            metadata=metadata or {"description": "Specialized in text processing with RAG capabilities"}
        )
        
        # 初始化RAG检索器
        self.rag_retriever = rag_retriever
        self.rag_k = rag_k
        self.hybrid_search = hybrid_search
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=temperature
        )
        
        # 初始化对话内存
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 创建基础提示模板
        self.qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "你是一个专业、准确的AI助手。你将基于提供的上下文信息来回答问题。"
                "如果问题无法从上下文中得到答案，请坦率地说明无法找到相关信息，不要编造信息。"
                "回答要简洁、专业，并引用上下文的相关部分来支持你的回答。"
            )),
            HumanMessagePromptTemplate.from_template(
                "基于以下上下文信息回答问题:\n\n"
                "上下文: {context}\n\n"
                "问题: {question}\n"
            )
        ])
        
        # 创建LLM链
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=self.qa_prompt
        )
        
        # 注册其他消息处理器
        self._register_additional_handlers()
        
        logger.info(f"TextAgent '{name}' initialized with RAG capabilities")
    
    def _register_additional_handlers(self):
        """注册其他消息处理器"""
        pass  # 可以根据需要添加其他消息处理器
    
    def execute_task(self, task_info: Dict[str, Any]) -> Any:
        """执行分配的任务"""
        task_type = task_info.get("metadata", {}).get("task_type", "qa")
        
        logger.info(f"TextAgent '{self.name}' executing task of type: {task_type}")
        
        if task_type == "qa":
            return self._handle_qa_task(task_info)
        elif task_type == "summarization":
            return self._handle_summarization_task(task_info)
        elif task_type == "text_analysis":
            return self._handle_analysis_task(task_info)
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {"error": f"Unsupported task type: {task_type}"}
    
    def _handle_qa_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理问答任务"""
        question = task_info.get("description", "")
        
        # 提取其他参数
        rag_k = task_info.get("metadata", {}).get("rag_k", self.rag_k)
        hybrid = task_info.get("metadata", {}).get("hybrid_search", self.hybrid_search)
        filters = task_info.get("metadata", {}).get("filters", None)
        
        # 检查是否有RAG检索器
        if not self.rag_retriever:
            logger.warning("No RAG retriever available for QA task")
            return {
                "answer": "I'm unable to retrieve information as the RAG system is not available.",
                "sources": []
            }
        
        try:
            # 执行RAG检索
            if hybrid:
                search_results = self.rag_retriever.hybrid_search(
                    query=question,
                    k=rag_k,
                    filter=filters,
                    include_metadata=True
                )
            else:
                search_results = self.rag_retriever.search(
                    query=question,
                    k=rag_k,
                    filter=filters,
                    include_metadata=True
                )
            
            # 准备上下文
            if search_results:
                context = "\n\n".join([f"[{i+1}] {result['content']}" for i, result in enumerate(search_results)])
                sources = [result.get("metadata", {}) for result in search_results]
            else:
                context = "没有找到相关信息。"
                sources = []
            
            # 执行QA链
            response = self.qa_chain.run(
                context=context,
                question=question
            )
            
            logger.info(f"QA task completed with {len(search_results)} retrieved documents")
            
            return {
                "answer": response,
                "sources": sources
            }
        
        except Exception as e:
            logger.error(f"Error processing QA task: {str(e)}")
            return {
                "error": f"处理问题时出错: {str(e)}",
                "answer": "很抱歉，处理您的问题时发生错误。请稍后再试。",
                "sources": []
            }
    
    def _handle_summarization_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本摘要任务"""
        text = task_info.get("metadata", {}).get("text", "")
        doc_id = task_info.get("metadata", {}).get("document_id")
        max_length = task_info.get("metadata", {}).get("max_length", 200)
        
        # 如果提供了文档ID而不是直接提供文本
        if not text and doc_id and self.rag_retriever:
            # 尝试从RAG系统检索文档
            search_results = self.rag_retriever.search(
                query="",  # 空查询，使用ID过滤器
                k=1,
                filter={"document_id": doc_id}
            )
            
            if search_results:
                text = search_results[0]["content"]
        
        if not text:
            return {
                "error": "No text provided for summarization",
                "summary": ""
            }
        
        try:
            # 创建摘要提示
            summarization_prompt = PromptTemplate(
                input_variables=["text", "max_length"],
                template=(
                    "请为以下文本创建一个简明扼要的摘要，最多{max_length}个字：\n\n"
                    "{text}\n\n"
                    "摘要："
                )
            )
            
            # 创建摘要链
            summarization_chain = LLMChain(
                llm=self.llm,
                prompt=summarization_prompt
            )
            
            # 执行摘要
            summary = summarization_chain.run(
                text=text[:8000],  # 限制输入长度
                max_length=max_length
            )
            
            logger.info(f"Summarization task completed, generated {len(summary)} chars summary")
            
            return {
                "summary": summary
            }
        
        except Exception as e:
            logger.error(f"Error processing summarization task: {str(e)}")
            return {
                "error": f"生成摘要时出错: {str(e)}",
                "summary": ""
            }
    
    def _handle_analysis_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本分析任务"""
        text = task_info.get("metadata", {}).get("text", "")
        analysis_type = task_info.get("metadata", {}).get("analysis_type", "sentiment")
        
        if not text:
            return {
                "error": "No text provided for analysis",
                "result": {}
            }
        
        try:
            if analysis_type == "sentiment":
                # 情感分析
                analysis_prompt = PromptTemplate(
                    input_variables=["text"],
                    template=(
                        "请对以下文本进行情感分析，并给出以下信息：\n"
                        "1. 总体情感（积极、中性或消极）\n"
                        "2. 情感强度（1-5，其中5表示最强烈）\n"
                        "3. 突出文本中的关键情感词汇\n"
                        "4. 简短的情感分析结论\n\n"
                        "文本：{text}\n\n"
                        "请以JSON格式返回结果，包含以下字段：sentiment, intensity, key_words, conclusion"
                    )
                )
            
            elif analysis_type == "key_points":
                # 关键点提取
                analysis_prompt = PromptTemplate(
                    input_variables=["text"],
                    template=(
                        "请从以下文本中提取最重要的要点：\n\n"
                        "{text}\n\n"
                        "列出5-7个最重要的要点，并为每个要点提供简短说明。"
                        "请以JSON格式返回结果，包含一个名为'key_points'的数组，每个数组项应有'point'和'explanation'字段。"
                    )
                )
            
            elif analysis_type == "entity":
                # 实体识别
                analysis_prompt = PromptTemplate(
                    input_variables=["text"],
                    template=(
                        "请从以下文本中识别所有重要实体（人物、组织、地点、日期等）：\n\n"
                        "{text}\n\n"
                        "对于每个实体，提供其类型和在文本中的重要性。"
                        "请以JSON格式返回结果，包含一个名为'entities'的数组，每个数组项应有'entity'、'type'和'importance'字段。"
                    )
                )
            
            else:
                return {
                    "error": f"Unsupported analysis type: {analysis_type}",
                    "result": {}
                }
            
            # 创建分析链
            analysis_chain = LLMChain(
                llm=self.llm,
                prompt=analysis_prompt
            )
            
            # 执行分析
            analysis_result_text = analysis_chain.run(text=text[:4000])  # 限制输入长度
            
            # 尝试解析JSON结果
            try:
                # 从文本中提取JSON部分
                json_text = analysis_result_text
                # 如果结果包含多余的文本，尝试提取JSON部分
                if not json_text.strip().startswith('{'):
                    start_idx = json_text.find('{')
                    end_idx = json_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_text = json_text[start_idx:end_idx]
                
                result = json.loads(json_text)
            except json.JSONDecodeError:
                # 如果无法解析JSON，返回原始文本
                result = {"raw_result": analysis_result_text}
            
            logger.info(f"Text analysis task completed with type: {analysis_type}")
            
            return {
                "analysis_type": analysis_type,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error processing text analysis task: {str(e)}")
            return {
                "error": f"执行文本分析时出错: {str(e)}",
                "analysis_type": analysis_type,
                "result": {}
            }
    
    def answer_question(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """回答问题（便捷方法）"""
        if use_rag and self.rag_retriever:
            task_info = {
                "description": question,
                "metadata": {
                    "task_type": "qa",
                    "rag_k": self.rag_k,
                    "hybrid_search": self.hybrid_search
                }
            }
            return self._handle_qa_task(task_info)
        else:
            # 直接使用LLM回答
            try:
                messages = [
                    SystemMessage(content="你是一个有帮助的AI助手。请简明扼要地回答问题。"),
                    HumanMessage(content=question)
                ]
                
                response = self.llm(messages)
                
                return {
                    "answer": response.content,
                    "sources": []
                }
            
            except Exception as e:
                logger.error(f"Error answering question directly: {str(e)}")
                return {
                    "error": str(e),
                    "answer": "很抱歉，处理您的问题时发生错误。",
                    "sources": []
                }
    
    def add_to_knowledge_base(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加文档到知识库"""
        if not self.rag_retriever:
            logger.warning("No RAG retriever available to add document")
            return False
        
        try:
            return self.rag_retriever.add_documents([document], [metadata] if metadata else None)
        
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {str(e)}")
            return False 
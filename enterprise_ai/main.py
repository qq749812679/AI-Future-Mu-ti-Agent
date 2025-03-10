"""
企业级AI多Agent多模态系统 - 主程序入口
演示如何初始化和使用MCP协议的多Agent系统
"""

import os
import sys
import logging
import time
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# 导入MCP协议核心组件
from core.controller import Controller, MessageType, AgentCapability
from core.message_bus import MessageBus

# 导入Agent实现
from agents.text_agent import TextAgent

# 导入RAG系统
from rag.retriever import RAGRetriever

# 示例知识库文档
SAMPLE_DOCUMENTS = [
    {
        "content": """
        MCP协议是Master Control Protocol的缩写，是一种用于分布式系统中多Agent协作的协议标准。
        它定义了Agent之间通信、任务分配和状态同步的标准化方式，使得不同功能的智能体可以无缝协作。
        MCP协议的核心组件包括控制器、消息总线和各类专用Agent。
        """,
        "metadata": {
            "source": "MCP Protocol Documentation",
            "topic": "protocol_basics",
            "document_id": "doc001"
        }
    },
    {
        "content": """
        RAG (Retrieval-Augmented Generation) 是一种结合检索系统和生成模型的AI技术。
        它通过从知识库中检索相关信息来增强语言模型的回答能力，从而提供更准确、更有依据的回复。
        RAG系统通常包括文档处理、向量存储和相似度检索等组件。
        这种方法可以有效减少AI的幻觉问题，提高回答的可靠性。
        """,
        "metadata": {
            "source": "AI Technologies Overview",
            "topic": "rag_technology",
            "document_id": "doc002"
        }
    },
    {
        "content": """
        在多Agent系统中，Agent协作是关键挑战。每个Agent通常具有特定的能力和专长，系统需要协调这些Agent共同解决复杂问题。
        协作模式主要有三种：
        1. 主从式：由一个主控Agent分配任务给其他Agent
        2. 对等式：Agent之间平等协商任务分工
        3. 混合式：结合前两种模式的灵活协作机制
        有效的协作需要明确的通信协议、任务分配机制和冲突解决策略。
        """,
        "metadata": {
            "source": "Multi-Agent Systems Design",
            "topic": "agent_collaboration",
            "document_id": "doc003"
        }
    }
]

def initialize_system():
    """初始化MCP多Agent系统"""
    logger.info("Initializing MCP multi-agent system...")
    
    # 创建MCP控制器
    controller = Controller()
    logger.info("MCP Controller initialized")
    
    # 创建消息总线
    message_bus = MessageBus(controller_reference=controller)
    logger.info("Message Bus initialized")
    
    # 创建RAG检索器
    rag_retriever = RAGRetriever(
        vector_db_type="chroma",
        embedding_model="openai",
        persist_directory="./data/vector_db",
        collection_name="enterprise_knowledge"
    )
    logger.info("RAG Retriever initialized")
    
    # 添加示例文档到知识库
    for doc in SAMPLE_DOCUMENTS:
        rag_retriever.add_documents([doc["content"]], [doc["metadata"]])
    logger.info(f"Added {len(SAMPLE_DOCUMENTS)} sample documents to knowledge base")
    
    # 创建文本处理Agent
    text_agent = TextAgent(
        name="TextProcessor",
        controller_reference=controller,
        rag_retriever=rag_retriever,
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    )
    logger.info("Text Agent initialized")
    
    # 注册消息处理器
    def handle_system_notification(message):
        logger.info(f"System notification received: {message.content}")
    
    message_bus.subscribe("controller", handle_system_notification)
    
    # 构建系统对象集合
    system = {
        "controller": controller,
        "message_bus": message_bus,
        "agents": {
            "text_agent": text_agent
        },
        "rag_retriever": rag_retriever
    }
    
    logger.info("MCP system initialization completed")
    return system

def run_demo_tasks(system):
    """运行示例任务演示系统功能"""
    logger.info("Running demo tasks...")
    
    controller = system["controller"]
    text_agent = system["agents"]["text_agent"]
    
    # 示例1: 问答任务
    logger.info("Demo Task 1: Question Answering with RAG")
    qa_result = text_agent.answer_question("什么是MCP协议？它有哪些核心组件？")
    
    print("\n--- Question Answering Result ---")
    print(f"Question: 什么是MCP协议？它有哪些核心组件？")
    print(f"Answer: {qa_result['answer']}")
    print("Sources:")
    for src in qa_result.get("sources", []):
        print(f"  - {src.get('source', 'Unknown')}")
    print()
    
    # 示例2: 通过控制器创建任务
    logger.info("Demo Task 2: Creating a task through the Controller")
    task_id = controller.create_task(
        description="分析以下文本中的关键实体：MCP协议是由通信工程师John Smith于2010年提出的，现已被Microsoft和Google等公司采用。",
        creator_id="main_app",
        required_capabilities=[AgentCapability.TEXT_PROCESSING],
        metadata={
            "task_type": "text_analysis",
            "analysis_type": "entity",
            "text": "MCP协议是由通信工程师John Smith于2010年提出的，现已被Microsoft和Google等公司采用。"
        }
    )
    
    # 等待任务完成
    max_wait = 10  # 最多等待10秒
    for _ in range(max_wait):
        task_status = controller.get_task_status(task_id)
        if task_status and task_status.value in ["completed", "failed"]:
            break
        time.sleep(1)
    
    # 获取任务结果
    task_result = controller.tasks.get(task_id, {}).result
    
    print("\n--- Text Analysis Result ---")
    print(f"Task ID: {task_id}")
    print(f"Result: {task_result}")
    print()
    
    # 示例3: 摘要生成
    logger.info("Demo Task 3: Document Summarization")
    summary_task = {
        "description": "Summarize the text about agent collaboration",
        "metadata": {
            "task_type": "summarization",
            "document_id": "doc003",
            "max_length": 100
        }
    }
    
    summary_result = text_agent.execute_task(summary_task)
    
    print("\n--- Document Summarization Result ---")
    print(f"Original Document ID: doc003")
    print(f"Summary: {summary_result.get('summary', '')}")
    print()
    
    # 系统状态报告
    system_status = controller.get_system_status()
    
    print("\n--- System Status Report ---")
    print(f"Agents Count: {system_status['agents_count']}")
    print(f"Tasks Count: {system_status['tasks_count']}")
    print(f"Completed Tasks: {system_status['completed_tasks']}")
    print(f"Failed Tasks: {system_status['failed_tasks']}")
    print()
    
    logger.info("Demo tasks completed")

def main():
    """主程序入口"""
    try:
        print("\n===== 企业级AI多Agent多模态系统演示 =====\n")
        
        # 初始化系统
        system = initialize_system()
        
        # 运行示例任务
        run_demo_tasks(system)
        
        # 清理资源
        system["message_bus"].shutdown()
        
        print("\n===== 演示完成 =====\n")
        
    except Exception as e:
        logger.error(f"Error in main program: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 
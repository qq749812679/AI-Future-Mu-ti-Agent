"""
API路由 - 提供RESTful API接口，允许外部系统与多Agent系统交互
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MCP核心组件
from core.controller import Controller, MessageType, TaskStatus, AgentCapability
from agents.text_agent import TextAgent
from rag.retriever import RAGRetriever

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="企业级AI多Agent系统API",
    description="提供与MCP协议多Agent系统交互的RESTful API接口",
    version="1.0.0"
)

# 定义数据模型
class QuestionRequest(BaseModel):
    question: str = Field(..., description="用户提问")
    use_rag: bool = Field(True, description="是否使用RAG增强")
    rag_k: Optional[int] = Field(None, description="检索文档数量")
    hybrid_search: Optional[bool] = Field(None, description="是否使用混合检索")

class DocumentRequest(BaseModel):
    content: str = Field(..., description="文档内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")

class TaskRequest(BaseModel):
    description: str = Field(..., description="任务描述")
    task_type: str = Field(..., description="任务类型")
    params: Optional[Dict[str, Any]] = Field({}, description="任务参数")

class AnalysisRequest(BaseModel):
    text: str = Field(..., description="待分析文本")
    analysis_type: str = Field("sentiment", description="分析类型")

class SummarizationRequest(BaseModel):
    text: str = Field(..., description="待总结文本")
    max_length: int = Field(200, description="最大长度")

# 全局变量持有系统实例
controller = None
text_agent = None
rag_retriever = None

# 依赖项函数，用于获取控制器
def get_controller():
    if controller is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return controller

# 依赖项函数，用于获取文本Agent
def get_text_agent():
    if text_agent is None:
        raise HTTPException(status_code=503, detail="Text Agent not initialized")
    return text_agent

# 依赖项函数，用于获取RAG检索器
def get_rag_retriever():
    if rag_retriever is None:
        raise HTTPException(status_code=503, detail="RAG Retriever not initialized")
    return rag_retriever

# 初始化系统
def initialize_system():
    global controller, text_agent, rag_retriever
    
    # 创建MCP控制器
    controller = Controller()
    
    # 创建RAG检索器
    rag_retriever = RAGRetriever(
        vector_db_type="chroma",
        embedding_model="openai",
        persist_directory="./data/vector_db",
        collection_name="enterprise_knowledge"
    )
    
    # 创建文本处理Agent
    text_agent = TextAgent(
        name="API_TextAgent",
        controller_reference=controller,
        rag_retriever=rag_retriever,
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    logger.info("API system initialized")

# 在应用启动时初始化系统
@app.on_event("startup")
async def startup_event():
    initialize_system()

# 路由定义
@app.get("/")
async def root():
    """API根路径，返回系统状态"""
    return {
        "status": "online",
        "system_info": controller.get_system_status() if controller else {"error": "System not initialized"}
    }

@app.post("/question")
async def ask_question(
    question_request: QuestionRequest,
    text_agent: TextAgent = Depends(get_text_agent)
):
    """提交问题并获取回答"""
    try:
        result = text_agent.answer_question(
            question=question_request.question,
            use_rag=question_request.use_rag
        )
        return result
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/document")
async def add_document(
    document_request: DocumentRequest,
    rag_retriever: RAGRetriever = Depends(get_rag_retriever)
):
    """添加文档到知识库"""
    try:
        success = rag_retriever.add_documents(
            documents=[document_request.content],
            metadatas=[document_request.metadata] if document_request.metadata else None
        )
        
        if success:
            return {"status": "success", "message": "Document added to knowledge base"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add document to knowledge base")
    
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.post("/task")
async def create_task(
    task_request: TaskRequest,
    background_tasks: BackgroundTasks,
    controller: Controller = Depends(get_controller)
):
    """创建新任务"""
    try:
        # 确定需要的能力
        capabilities = []
        
        if task_request.task_type in ["qa", "summarization", "text_analysis"]:
            capabilities.append(AgentCapability.TEXT_PROCESSING)
        
        if task_request.task_type in ["image_analysis", "ocr"]:
            capabilities.append(AgentCapability.IMAGE_PROCESSING)
        
        if not capabilities:
            raise HTTPException(status_code=400, detail=f"Unsupported task type: {task_request.task_type}")
        
        # 准备任务元数据
        metadata = {
            "task_type": task_request.task_type,
            **task_request.params
        }
        
        # 创建任务
        task_id = controller.create_task(
            description=task_request.description,
            creator_id="api_user",
            required_capabilities=capabilities,
            metadata=metadata
        )
        
        return {
            "task_id": task_id,
            "status": "created",
            "message": "Task created successfully"
        }
    
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")

@app.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    controller: Controller = Depends(get_controller)
):
    """获取任务状态和结果"""
    try:
        if task_id not in controller.tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task = controller.tasks[task_id]
        
        return {
            "task_id": task_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "assigned_agent_id": task.assigned_agent_id,
            "result": task.result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting task status: {str(e)}")

@app.post("/analyze")
async def analyze_text(
    analysis_request: AnalysisRequest,
    text_agent: TextAgent = Depends(get_text_agent)
):
    """分析文本"""
    try:
        task_info = {
            "description": f"Analyze text with type: {analysis_request.analysis_type}",
            "metadata": {
                "task_type": "text_analysis",
                "analysis_type": analysis_request.analysis_type,
                "text": analysis_request.text
            }
        }
        
        result = text_agent.execute_task(task_info)
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.post("/summarize")
async def summarize_text(
    summarization_request: SummarizationRequest,
    text_agent: TextAgent = Depends(get_text_agent)
):
    """总结文本"""
    try:
        task_info = {
            "description": f"Summarize text with max length: {summarization_request.max_length}",
            "metadata": {
                "task_type": "summarization",
                "text": summarization_request.text,
                "max_length": summarization_request.max_length
            }
        }
        
        result = text_agent.execute_task(task_info)
        return result
    
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")

@app.get("/system/status")
async def system_status(
    controller: Controller = Depends(get_controller)
):
    """获取系统状态"""
    try:
        status = controller.get_system_status()
        
        # 添加RAG系统状态
        if rag_retriever:
            status["rag"] = rag_retriever.get_stats()
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@app.post("/system/reset")
async def reset_system(
    background_tasks: BackgroundTasks
):
    """重置系统"""
    try:
        # 在后台重新初始化系统
        background_tasks.add_task(initialize_system)
        
        return {
            "status": "success",
            "message": "System reset initiated"
        }
    
    except Exception as e:
        logger.error(f"Error resetting system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting system: {str(e)}") 
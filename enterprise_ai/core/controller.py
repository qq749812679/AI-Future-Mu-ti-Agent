"""
MCP协议控制器 - 多Agent系统的核心组件
负责协调各Agent之间的通信、任务分发与状态管理
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    TASK_REQUEST = "task_request"  # 任务请求
    TASK_ASSIGNMENT = "task_assignment"  # 任务分配
    TASK_UPDATE = "task_update"  # 任务状态更新
    TASK_RESULT = "task_result"  # 任务结果
    AGENT_REGISTRATION = "agent_registration"  # Agent注册
    AGENT_STATUS = "agent_status"  # Agent状态更新
    SYSTEM_NOTIFICATION = "system_notification"  # 系统通知

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"  # 等待中
    ASSIGNED = "assigned"  # 已分配
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败

class Message:
    """MCP消息类"""
    def __init__(
        self,
        sender_id: str,
        receiver_id: str,
        content: Any,
        message_type: MessageType,
        message_id: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        self.message_id = message_id or str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.type = message_type
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典表示"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "type": self.type.value,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息对象"""
        return cls(
            message_id=data.get("message_id"),
            sender_id=data.get("sender_id"),
            receiver_id=data.get("receiver_id"),
            content=data.get("content"),
            message_type=MessageType(data.get("type")),
            created_at=datetime.fromisoformat(data.get("created_at"))
        )

class Task:
    """任务类"""
    def __init__(
        self,
        description: str,
        creator_id: str,
        task_id: Optional[str] = None,
        status: TaskStatus = TaskStatus.PENDING,
        assigned_agent_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        deadline: Optional[datetime] = None,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.description = description
        self.creator_id = creator_id
        self.status = status
        self.assigned_agent_id = assigned_agent_id
        self.created_at = created_at or datetime.now()
        self.deadline = deadline
        self.priority = priority
        self.metadata = metadata or {}
        self.result = None

    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典表示"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "creator_id": self.creator_id,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "priority": self.priority,
            "metadata": self.metadata,
            "result": self.result
        }

class AgentCapability(Enum):
    """Agent能力枚举"""
    TEXT_PROCESSING = "text_processing"  # 文本处理
    IMAGE_PROCESSING = "image_processing"  # 图像处理
    AUDIO_PROCESSING = "audio_processing"  # 音频处理
    CODE_GENERATION = "code_generation"  # 代码生成
    DATA_ANALYSIS = "data_analysis"  # 数据分析
    REASONING = "reasoning"  # 推理能力

class Controller:
    """MCP控制器 - 系统的中央协调组件"""
    
    def __init__(self):
        # 注册的Agent字典 {agent_id: agent_info}
        self.agents: Dict[str, Dict[str, Any]] = {}
        # 任务字典 {task_id: Task}
        self.tasks: Dict[str, Task] = {}
        # 消息历史记录
        self.message_history: List[Message] = []
        # 消息处理器注册表 {message_type: [handler_functions]}
        self.message_handlers: Dict[MessageType, List[Callable]] = {msg_type: [] for msg_type in MessageType}
        # 能力路由表 {capability: [agent_ids]}
        self.capability_routing: Dict[AgentCapability, List[str]] = {cap: [] for cap in AgentCapability}
        
        logger.info("MCP Controller initialized")

    def register_agent(self, agent_id: str, name: str, capabilities: List[AgentCapability], 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """注册新Agent到系统"""
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered")
            return False
        
        self.agents[agent_id] = {
            "id": agent_id,
            "name": name,
            "capabilities": capabilities,
            "status": "active",
            "registered_at": datetime.now(),
            "last_active": datetime.now(),
            "metadata": metadata or {}
        }
        
        # 更新能力路由表
        for capability in capabilities:
            if capability in self.capability_routing:
                self.capability_routing[capability].append(agent_id)
        
        logger.info(f"Agent {agent_id} ({name}) registered with capabilities: {[c.value for c in capabilities]}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """从系统中注销Agent"""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        # 从能力路由表中移除
        for capability in self.agents[agent_id]["capabilities"]:
            if capability in self.capability_routing and agent_id in self.capability_routing[capability]:
                self.capability_routing[capability].remove(agent_id)
        
        # 移除Agent
        agent_name = self.agents[agent_id]["name"]
        del self.agents[agent_id]
        
        logger.info(f"Agent {agent_id} ({agent_name}) unregistered")
        return True
    
    def create_task(self, description: str, creator_id: str, required_capabilities: List[AgentCapability], 
                   priority: int = 1, deadline: Optional[datetime] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """创建新任务并返回任务ID"""
        task = Task(
            description=description,
            creator_id=creator_id,
            priority=priority,
            deadline=deadline,
            metadata=metadata or {"required_capabilities": [c.value for c in required_capabilities]}
        )
        
        self.tasks[task.task_id] = task
        logger.info(f"Task {task.task_id} created: {description[:50]}...")
        
        # 尝试分配任务
        self._assign_task(task.task_id, required_capabilities)
        
        return task.task_id
    
    def _assign_task(self, task_id: str, required_capabilities: List[AgentCapability]) -> bool:
        """根据能力要求分配任务给合适的Agent"""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        
        # 寻找满足所有能力要求的Agent
        suitable_agents = []
        for agent_id, agent_info in self.agents.items():
            if all(cap in agent_info["capabilities"] for cap in required_capabilities):
                suitable_agents.append(agent_id)
        
        if not suitable_agents:
            logger.warning(f"No suitable agent found for task {task_id}")
            return False
        
        # 简单策略：选择第一个合适的Agent
        # 实际系统中可以实现更复杂的分配策略（负载均衡、专业性匹配等）
        selected_agent_id = suitable_agents[0]
        
        # 更新任务状态
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = selected_agent_id
        
        # 发送任务分配消息
        self.send_message(
            sender_id="controller",
            receiver_id=selected_agent_id,
            content={
                "task_id": task_id,
                "description": task.description,
                "metadata": task.metadata
            },
            message_type=MessageType.TASK_ASSIGNMENT
        )
        
        logger.info(f"Task {task_id} assigned to agent {selected_agent_id}")
        return True
    
    def send_message(self, sender_id: str, receiver_id: str, content: Any, 
                    message_type: MessageType) -> str:
        """发送消息并触发相应的处理器"""
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type
        )
        
        # 记录消息历史
        self.message_history.append(message)
        
        # 触发消息处理器
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error handling message {message.message_id}: {str(e)}")
        
        logger.debug(f"Message {message.message_id} sent from {sender_id} to {receiver_id}")
        return message.message_id
    
    def register_message_handler(self, message_type: MessageType, handler: Callable[[Message], None]) -> None:
        """注册消息处理器"""
        if message_type in self.message_handlers:
            self.message_handlers[message_type].append(handler)
            logger.debug(f"Registered new handler for message type {message_type.value}")
    
    def update_task_status(self, task_id: str, status: TaskStatus, result: Any = None) -> bool:
        """更新任务状态"""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        task.status = status
        
        if result is not None:
            task.result = result
        
        # 通知任务创建者
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self.send_message(
                sender_id="controller",
                receiver_id=task.creator_id,
                content={
                    "task_id": task_id,
                    "status": status.value,
                    "result": task.result
                },
                message_type=MessageType.TASK_RESULT
            )
        
        logger.info(f"Task {task_id} status updated to {status.value}")
        return True
    
    def get_agent_by_capability(self, capability: AgentCapability) -> List[str]:
        """根据能力获取Agent列表"""
        return self.capability_routing.get(capability, [])
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态概览"""
        return {
            "agents_count": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a["status"] == "active"),
            "tasks_count": len(self.tasks),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "in_progress_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "messages_count": len(self.message_history)
        } 
"""
Agent基类 - 所有特定领域Agent的基础实现
提供Agent的通用功能和与MCP控制器的通信接口
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from datetime import datetime

# 导入MCP协议基础组件
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.controller import MessageType, TaskStatus, AgentCapability, Message

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """所有Agent的基类"""
    
    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        controller_reference,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.capabilities = capabilities
        self.controller = controller_reference
        self.metadata = metadata or {}
        self.state = "initialized"
        self.current_task = None
        self.message_handlers = {}
        
        # 自动注册到控制器
        self._register_with_controller()
        
        # 注册默认消息处理器
        self._register_default_handlers()
        
        logger.info(f"Agent '{name}' ({self.agent_id}) initialized with capabilities: {[c.value for c in capabilities]}")
    
    def _register_with_controller(self):
        """注册当前Agent到MCP控制器"""
        if self.controller:
            success = self.controller.register_agent(
                agent_id=self.agent_id,
                name=self.name,
                capabilities=self.capabilities,
                metadata=self.metadata
            )
            if success:
                self.state = "active"
                logger.info(f"Agent '{self.name}' successfully registered with controller")
            else:
                logger.warning(f"Failed to register agent '{self.name}' with controller")
    
    def _register_default_handlers(self):
        """注册默认的消息处理器"""
        self.register_message_handler(MessageType.TASK_ASSIGNMENT, self._handle_task_assignment)
        self.register_message_handler(MessageType.SYSTEM_NOTIFICATION, self._handle_system_notification)
    
    def register_message_handler(self, message_type: MessageType, handler: Callable[[Message], None]):
        """注册消息处理器"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Agent '{self.name}' registered handler for message type: {message_type.value}")
    
    def process_message(self, message: Message):
        """处理收到的消息"""
        if message.receiver_id != self.agent_id and message.receiver_id != "broadcast":
            logger.warning(f"Agent '{self.name}' received message intended for {message.receiver_id}")
            return False
        
        logger.info(f"Agent '{self.name}' processing message: {message.message_id} of type {message.type.value}")
        
        # 调用相应类型的消息处理器
        if message.type in self.message_handlers:
            for handler in self.message_handlers[message.type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error handling message {message.message_id}: {str(e)}")
        else:
            logger.warning(f"No handler registered for message type: {message.type.value}")
        
        return True
    
    def _handle_task_assignment(self, message: Message):
        """处理任务分配消息"""
        task_info = message.content
        logger.info(f"Agent '{self.name}' received task assignment: {task_info.get('task_id')}")
        
        # 更新当前任务
        self.current_task = task_info
        
        # 开始执行任务
        try:
            # 首先通知控制器任务状态已更改为进行中
            if self.controller:
                self.controller.update_task_status(
                    task_id=task_info.get('task_id'),
                    status=TaskStatus.IN_PROGRESS
                )
            
            # 执行具体任务逻辑
            task_result = self.execute_task(task_info)
            
            # 更新任务状态为已完成
            if self.controller:
                self.controller.update_task_status(
                    task_id=task_info.get('task_id'),
                    status=TaskStatus.COMPLETED,
                    result=task_result
                )
            
            logger.info(f"Agent '{self.name}' completed task: {task_info.get('task_id')}")
            
        except Exception as e:
            logger.error(f"Agent '{self.name}' failed to execute task {task_info.get('task_id')}: {str(e)}")
            # 更新任务状态为失败
            if self.controller:
                self.controller.update_task_status(
                    task_id=task_info.get('task_id'),
                    status=TaskStatus.FAILED,
                    result={"error": str(e)}
                )
        
        # 清除当前任务
        self.current_task = None
    
    def _handle_system_notification(self, message: Message):
        """处理系统通知消息"""
        notification = message.content
        logger.info(f"Agent '{self.name}' received system notification: {notification.get('type')}")
        
        # 根据通知类型执行相应操作
        if notification.get('type') == 'shutdown':
            self.state = "shutting_down"
            logger.info(f"Agent '{self.name}' is shutting down...")
            self._cleanup()
        
        elif notification.get('type') == 'pause':
            self.state = "paused"
            logger.info(f"Agent '{self.name}' is paused")
        
        elif notification.get('type') == 'resume':
            self.state = "active"
            logger.info(f"Agent '{self.name}' is resumed")
    
    def _cleanup(self):
        """执行Agent清理操作"""
        # 从控制器注销
        if self.controller:
            self.controller.unregister_agent(self.agent_id)
        
        self.state = "terminated"
        logger.info(f"Agent '{self.name}' cleanup completed")
    
    def send_message(self, receiver_id: str, content: Any, message_type: MessageType):
        """发送消息到其他Agent或控制器"""
        if not self.controller:
            logger.error(f"Agent '{self.name}' cannot send message: no controller reference")
            return None
        
        message_id = self.controller.send_message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type
        )
        
        logger.debug(f"Agent '{self.name}' sent message {message_id} to {receiver_id}")
        return message_id
    
    def create_task(self, description: str, required_capabilities: List[AgentCapability], 
                   priority: int = 1, metadata: Optional[Dict[str, Any]] = None):
        """创建新任务并提交给控制器"""
        if not self.controller:
            logger.error(f"Agent '{self.name}' cannot create task: no controller reference")
            return None
        
        task_id = self.controller.create_task(
            description=description,
            creator_id=self.agent_id,
            required_capabilities=required_capabilities,
            priority=priority,
            metadata=metadata
        )
        
        logger.info(f"Agent '{self.name}' created task: {task_id}")
        return task_id
    
    @abstractmethod
    def execute_task(self, task_info: Dict[str, Any]) -> Any:
        """执行分配的任务 - 需要由子类实现"""
        pass 
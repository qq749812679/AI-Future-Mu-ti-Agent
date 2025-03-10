"""
消息总线 - 负责MCP协议中的消息路由和分发
支持异步消息传递、消息队列管理和消息处理
"""

import uuid
import logging
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime

# 导入MCP协议基础组件
from .controller import Message, MessageType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageBus:
    """消息总线实现，支持异步消息分发和订阅"""
    
    def __init__(self, controller_reference, max_queue_size: int = 1000, workers: int = 3):
        self.controller = controller_reference
        self.message_queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = True
        self.subscribers = {}  # {agent_id/topic: [callback_functions]}
        self.topic_subscriptions = {}  # {topic: [agent_ids]}
        self.workers = []
        self.num_workers = workers
        
        # 创建并启动工作线程
        self._start_workers()
        
        logger.info(f"MessageBus initialized with {workers} workers")
    
    def _start_workers(self):
        """启动工作线程处理消息队列"""
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._process_queue, name=f"MessageBusWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            logger.debug(f"Started MessageBusWorker-{i}")
    
    def _process_queue(self):
        """工作线程函数：从队列中取出消息并处理"""
        while self.is_running:
            try:
                # 从队列获取消息，如无消息则阻塞
                message, callbacks = self.message_queue.get(block=True, timeout=1.0)
                
                # 处理消息（调用回调函数）
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in message callback: {str(e)}")
                
                # 标记任务完成
                self.message_queue.task_done()
                
            except queue.Empty:
                # 队列超时，继续循环
                continue
            except Exception as e:
                logger.error(f"Error in message worker: {str(e)}")
    
    def publish(self, message: Message) -> bool:
        """发布消息到总线"""
        if not self.is_running:
            logger.warning("MessageBus is not running, cannot publish message")
            return False
        
        # 确定消息的接收者和回调函数
        callbacks = []
        
        # 直接接收者的回调
        if message.receiver_id in self.subscribers:
            callbacks.extend(self.subscribers[message.receiver_id])
        
        # 广播消息
        if message.receiver_id == "broadcast":
            for agent_callbacks in self.subscribers.values():
                callbacks.extend(agent_callbacks)
        
        # 主题订阅
        if "topic" in message.content and message.content["topic"] in self.topic_subscriptions:
            topic = message.content["topic"]
            for agent_id in self.topic_subscriptions[topic]:
                if agent_id in self.subscribers:
                    callbacks.extend(self.subscribers[agent_id])
        
        # 如果没有接收者，记录警告
        if not callbacks:
            logger.warning(f"No subscribers found for message {message.message_id} to {message.receiver_id}")
            return False
        
        try:
            # 将消息和回调放入队列
            self.message_queue.put((message, callbacks), block=True, timeout=2.0)
            logger.debug(f"Message {message.message_id} queued for delivery to {len(callbacks)} subscribers")
            return True
        except queue.Full:
            logger.error(f"Message queue is full, cannot publish message {message.message_id}")
            return False
    
    def subscribe(self, agent_id: str, callback: Callable[[Message], None]) -> bool:
        """订阅某个Agent ID的消息"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        
        self.subscribers[agent_id].append(callback)
        logger.info(f"Agent {agent_id} subscribed to messages")
        return True
    
    def unsubscribe(self, agent_id: str, callback: Optional[Callable] = None) -> bool:
        """取消订阅某个Agent ID的消息"""
        if agent_id not in self.subscribers:
            logger.warning(f"Agent {agent_id} not found in subscribers")
            return False
        
        if callback:
            # 移除特定回调
            if callback in self.subscribers[agent_id]:
                self.subscribers[agent_id].remove(callback)
                logger.info(f"Callback removed from agent {agent_id} subscriptions")
            else:
                logger.warning(f"Callback not found in agent {agent_id} subscriptions")
                return False
        else:
            # 移除所有回调
            self.subscribers.pop(agent_id)
            logger.info(f"All subscriptions removed for agent {agent_id}")
        
        # 从主题订阅中移除
        for topic, agents in self.topic_subscriptions.items():
            if agent_id in agents:
                agents.remove(agent_id)
        
        return True
    
    def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """订阅特定主题的消息"""
        if topic not in self.topic_subscriptions:
            self.topic_subscriptions[topic] = []
        
        if agent_id not in self.topic_subscriptions[topic]:
            self.topic_subscriptions[topic].append(agent_id)
            logger.info(f"Agent {agent_id} subscribed to topic {topic}")
            return True
        
        return False
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        """取消订阅特定主题的消息"""
        if topic not in self.topic_subscriptions:
            logger.warning(f"Topic {topic} not found in subscriptions")
            return False
        
        if agent_id in self.topic_subscriptions[topic]:
            self.topic_subscriptions[topic].remove(agent_id)
            logger.info(f"Agent {agent_id} unsubscribed from topic {topic}")
            return True
        
        logger.warning(f"Agent {agent_id} not subscribed to topic {topic}")
        return False
    
    def shutdown(self):
        """关闭消息总线"""
        logger.info("Shutting down MessageBus...")
        self.is_running = False
        
        # 等待所有工作线程完成
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
        
        # 清空消息队列
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except queue.Empty:
                break
        
        logger.info("MessageBus shutdown complete")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态信息"""
        return {
            "queue_size": self.message_queue.qsize(),
            "queue_full": self.message_queue.full(),
            "subscribers_count": len(self.subscribers),
            "topics_count": len(self.topic_subscriptions),
            "workers_alive": sum(1 for w in self.workers if w.is_alive())
        }
    
    def create_direct_message(
        self,
        sender_id: str,
        receiver_id: str,
        content: Any,
        message_type: MessageType
    ) -> Message:
        """创建直接消息并发布"""
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type
        )
        
        self.publish(message)
        return message
    
    def create_topic_message(
        self,
        sender_id: str,
        topic: str,
        content: Any,
        message_type: MessageType
    ) -> Message:
        """创建主题消息并发布"""
        # 创建包含主题的内容
        topic_content = content.copy() if isinstance(content, dict) else {"data": content}
        topic_content["topic"] = topic
        
        message = Message(
            sender_id=sender_id,
            receiver_id="broadcast",  # 主题消息使用广播接收者
            content=topic_content,
            message_type=message_type
        )
        
        self.publish(message)
        return message 
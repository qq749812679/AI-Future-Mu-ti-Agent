# 企业级AI多Agent多模态系统

基于MCP协议和LangChain框架实现的企业级AI多Agent多模态系统，包含RAG技术增强的知识检索能力。
#技术栈

- **LangChain**: 用于构建LLM应用的框架
- **FastAPI**: 后端API服务
- **ReactJS**: 前端用户界面
- **Chroma/FAISS**: 向量数据库
- **Redis**: 消息队列和缓存
- **Docker**: 容器化部署

### 1.1 什么是MCP协议

MCP (Master Control Protocol) 是一种用于分布式系统通信的协议框架，主要用于Agent之间的消息传递、状态同步和任务协调。在多Agent系统中，MCP协议提供了一套标准化的通信机制，使不同的智能体能够高效协作

### 1.2 核心组件设计

#### 控制器(Controller)
- 负责系统整体协调
- Agent注册与发现
- 消息路由与任务分发
- 系统状态监控

#### Agent
- 能力注册
- 消息处理
- 任务执行
- 状态同步

#### 消息总线
- 高性能消息队列
- 消息过滤与转换
- 消息持久化
- 消息优先级控制
  
##  1.3 核心实体
<img width="639" alt="image" src="https://github.com/user-attachments/assets/1c35633e-175d-4905-a1d1-64abb9a638c3" />

##  1.4 MCP 通信模型
<img width="957" alt="image" src="https://github.com/user-attachments/assets/7aa51156-a79e-43ca-8fb2-cbac3cc23183" />

##  1.5 MCP协议高级架构设计

<img width="926" alt="image" src="https://github.com/user-attachments/assets/edf9f452-f0c3-4b89-9e6d-e1e5393b4954" />

## 1.6 系统架构图

<img width="1195" alt="image" src="https://github.com/user-attachments/assets/f156068f-72e7-4c6b-b202-dce008ef2559" />


## 1.7 工作流程图

<img width="352" alt="image" src="https://github.com/user-attachments/assets/6fb89756-e078-48dc-ac27-e2060a19c766" />




## 安装与设置

1. 克隆仓库并安装依赖:
bash
git clone https://github.com/yourusername/enterprise-ai.git
cd enterprise-ai
pip install -r requirements.txt


2. 配置环境变量:
bash
cp .env.example .env
# 编辑.env文件设置您的API密钥和其他配置


3. 启动系统:
bash
python main.py



💬 贡献和反馈
如果您有任何反馈、建议或想要做出贡献，请随时打开问题或拉取请求。我们欢迎新的想法和建议。帮助我们完善这个项目，让它对 Java 社区更有用。

🧑🏻‍💻 作者
“AIbot-hum“ @qq749812679




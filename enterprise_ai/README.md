# 企业级AI多Agent多模态系统

基于MCP协议和LangChain框架实现的企业级AI多Agent多模态系统，包含RAG技术增强的知识检索能力。

## 项目结构

```
enterprise_ai/
├── README.md                  # 项目说明
├── requirements.txt           # 项目依赖
├── config/                    # 配置文件目录
│   └── config.py              # 配置管理
├── core/                      # 核心组件
│   ├── controller.py          # 中央控制器
│   ├── message_bus.py         # 消息总线
│   └── agent_manager.py       # Agent管理器
├── agents/                    # 各类Agent实现
│   ├── base_agent.py          # Agent基类
│   ├── text_agent.py          # 文本处理Agent
│   ├── vision_agent.py        # 视觉处理Agent
│   ├── audio_agent.py         # 音频处理Agent
│   └── expert_agent.py        # 专家领域Agent
├── rag/                       # RAG系统实现
│   ├── document_loader.py     # 文档加载器
│   ├── document_processor.py  # 文档处理器
│   ├── vector_store.py        # 向量存储接口
│   └── retriever.py           # 检索器
├── api/                       # API接口
│   ├── routes.py              # 路由定义
│   └── middleware.py          # 中间件
└── ui/                        # 用户界面
    ├── pages/                 # 页面组件
    └── components/            # UI组件
```

## 技术栈

- **LangChain**: 用于构建LLM应用的框架
- **FastAPI**: 后端API服务
- **ReactJS**: 前端用户界面
- **Chroma/FAISS**: 向量数据库
- **Redis**: 消息队列和缓存
- **Docker**: 容器化部署

## 安装与设置

1. 克隆仓库并安装依赖:
```bash
git clone https://github.com/yourusername/enterprise-ai.git
cd enterprise-ai
pip install -r requirements.txt
```

2. 配置环境变量:
```bash
cp .env.example .env
# 编辑.env文件设置您的API密钥和其他配置
```

3. 启动系统:
```bash
python main.py
```

## 使用方法

详细的使用说明请参考[使用文档](docs/usage.md)。

## 贡献指南

如果您希望为项目贡献代码，请参考[贡献指南](CONTRIBUTING.md)。

## 许可证

本项目采用MIT许可证。详情见[LICENSE](LICENSE)文件。 
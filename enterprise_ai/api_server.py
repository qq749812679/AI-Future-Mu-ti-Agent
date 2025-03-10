"""
API服务器入口 - 启动FastAPI服务，提供多Agent系统的REST API接口
"""

import os
import logging
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_server.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """启动API服务器"""
    try:
        logger.info("Starting Enterprise AI API Server...")
        
        # 获取配置参数
        host = os.environ.get("API_HOST", "127.0.0.1")
        port = int(os.environ.get("API_PORT", "8000"))
        reload = os.environ.get("API_RELOAD", "false").lower() in ("true", "1", "yes")
        
        # 显示启动信息
        print("\n" + "="*50)
        print(" 企业级AI多Agent多模态系统 API服务 ")
        print("="*50)
        print(f" 服务地址: http://{host}:{port}")
        print(f" Swagger文档: http://{host}:{port}/docs")
        print(f" ReDoc文档: http://{host}:{port}/redoc")
        print("="*50 + "\n")
        
        # 启动服务器
        uvicorn.run(
            "api.routes:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 
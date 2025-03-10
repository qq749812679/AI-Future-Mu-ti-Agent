"""
RAG检索器 - 为Agent提供知识检索能力
整合向量数据库和文档检索功能，增强Agent的信息获取能力
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.docstore.document import Document

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGRetriever:
    """RAG检索器实现，支持向量检索和混合检索策略"""
    
    def __init__(
        self, 
        vector_db_type: str = "chroma",
        embedding_model: str = "openai",
        persist_directory: Optional[str] = None,
        collection_name: str = "enterprise_rag",
        embedding_model_name: str = "text-embedding-ada-002",
        search_type: str = "similarity",
        distance_metric: str = "cosine"
    ):
        self.vector_db_type = vector_db_type.lower()
        self.embedding_model_type = embedding_model.lower()
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.search_type = search_type
        self.distance_metric = distance_metric
        
        # 初始化嵌入模型
        self.embedding_model = self._init_embedding_model()
        
        # 初始化向量数据库
        self.vector_db = self._init_vector_db()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info(f"RAGRetriever initialized with {vector_db_type} and {embedding_model} embedding model")
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        if self.embedding_model_type == "openai":
            return OpenAIEmbeddings(model=self.embedding_model_name)
        elif self.embedding_model_type == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        else:
            raise ValueError(f"Unsupported embedding model type: {self.embedding_model_type}")
    
    def _init_vector_db(self):
        """初始化向量数据库"""
        if self.vector_db_type == "chroma":
            if self.persist_directory and os.path.exists(self.persist_directory):
                # 从持久化目录加载
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name
                )
            else:
                # 创建新的数据库
                return Chroma(
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
        elif self.vector_db_type == "faiss":
            if self.persist_directory and os.path.exists(os.path.join(self.persist_directory, "faiss_index")):
                # 从持久化目录加载
                return FAISS.load_local(
                    folder_path=self.persist_directory,
                    embeddings=self.embedding_model,
                    index_name="faiss_index"
                )
            else:
                # 创建新的数据库
                return FAISS.from_documents(
                    documents=[],  # 空文档初始化
                    embedding=self.embedding_model
                )
        else:
            raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")
    
    def add_documents(self, documents: List[Union[str, Document]], metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """添加文档到向量数据库"""
        try:
            # 将字符串转换为Document对象
            doc_objects = []
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    doc_objects.append(Document(page_content=doc, metadata=metadata))
                else:
                    doc_objects.append(doc)
            
            # 分割文档
            split_docs = self.text_splitter.split_documents(doc_objects)
            
            # 添加到向量数据库
            if self.vector_db_type == "chroma":
                self.vector_db.add_documents(split_docs)
                if self.persist_directory:
                    self.vector_db.persist()
            elif self.vector_db_type == "faiss":
                self.vector_db = FAISS.from_documents(
                    documents=split_docs,
                    embedding=self.embedding_model
                )
                if self.persist_directory:
                    self.vector_db.save_local(self.persist_directory, index_name="faiss_index")
            
            logger.info(f"Added {len(split_docs)} document chunks to vector database")
            return True
        
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            return False
    
    def load_from_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """从文件加载文档"""
        try:
            # 根据文件扩展名选择合适的加载器
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".csv":
                loader = CSVLoader(file_path)
            elif ext in [".txt", ".md", ".py", ".java", ".js", ".html", ".css"]:
                loader = TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file extension: {ext}")
                return False
            
            # 加载文档
            docs = loader.load()
            
            # 添加元数据
            if metadata:
                for doc in docs:
                    doc.metadata.update(metadata)
            
            # 添加到向量数据库
            return self.add_documents(docs)
        
        except Exception as e:
            logger.error(f"Error loading document from file {file_path}: {str(e)}")
            return False
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        search_type: Optional[str] = None,
        include_metadata: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """执行向量检索"""
        try:
            search_type = search_type or self.search_type
            
            # 执行检索
            if search_type == "similarity":
                docs = self.vector_db.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
                
                # 处理结果
                results = []
                for doc in docs:
                    result = {
                        "content": doc.page_content,
                        "score": None  # similarity_search不返回分数
                    }
                    if include_metadata:
                        result["metadata"] = doc.metadata
                    results.append(result)
                
            elif search_type == "similarity_score":
                docs_and_scores = self.vector_db.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter
                )
                
                # 处理结果
                results = []
                for doc, score in docs_and_scores:
                    # 过滤低分结果
                    if score_threshold is not None and score < score_threshold:
                        continue
                    
                    result = {
                        "content": doc.page_content,
                        "score": score
                    }
                    if include_metadata:
                        result["metadata"] = doc.metadata
                    results.append(result)
            
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            logger.info(f"Vector search for '{query}' returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        bm25_weight: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """执行混合检索（向量检索 + BM25）"""
        # 当前实现仅支持FAISS
        if self.vector_db_type != "faiss":
            logger.warning(f"Hybrid search is currently only supported with FAISS, falling back to regular search")
            return self.search(query, k, filter, include_metadata=include_metadata)
        
        try:
            # 使用FAISS的混合检索
            docs_and_scores = self.vector_db.hybrid_search(
                query=query,
                k=k,
                filter=filter,
                alpha=bm25_weight  # BM25权重
            )
            
            # 处理结果
            results = []
            for doc, score in docs_and_scores:
                result = {
                    "content": doc.page_content,
                    "score": score
                }
                if include_metadata:
                    result["metadata"] = doc.metadata
                results.append(result)
            
            logger.info(f"Hybrid search for '{query}' returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return []
    
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> bool:
        """从向量数据库中删除文档"""
        try:
            if self.vector_db_type == "chroma":
                if ids:
                    self.vector_db.delete(ids=ids)
                elif filter:
                    self.vector_db.delete(filter=filter)
                else:
                    logger.warning("No ids or filter provided for deletion")
                    return False
                
                if self.persist_directory:
                    self.vector_db.persist()
                
                logger.info(f"Deleted documents from vector database")
                return True
            
            else:
                logger.warning(f"Delete operation not fully supported for {self.vector_db_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量数据库统计信息"""
        try:
            if self.vector_db_type == "chroma":
                collection = self.vector_db._collection
                return {
                    "document_count": collection.count(),
                    "vector_dimension": self.vector_db._embedding_function.embedding_dimension,
                    "db_type": self.vector_db_type,
                    "collection_name": self.collection_name
                }
            elif self.vector_db_type == "faiss":
                return {
                    "document_count": len(self.vector_db.index_to_docstore_id),
                    "vector_dimension": self.vector_db.index.d,
                    "db_type": self.vector_db_type
                }
            else:
                return {"error": "Stats not available for this database type"}
        
        except Exception as e:
            logger.error(f"Error getting vector database stats: {str(e)}")
            return {"error": str(e)} 
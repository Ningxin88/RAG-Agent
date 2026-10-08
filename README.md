RAG 知识库问答系统

基于 LangChain + ChromaDB 搭建的 RAG（检索增强生成）知识库问答系统。

技术架构
- 文档处理：PyPDFLoader + RecursiveCharacterTextSplitter
- 向量化：中文 Embedding 模型
- 向量存储：ChromaDB
- 检索策略：余弦相似度 + Top-K 检索
- 生成模型：LLM API

核心链路
离线索引：文档加载 → 文本分割 → 向量化 → 存入向量库
在线检索：查询向量化 → Top-K 检索 → Prompt 拼接 → LLM 生成

快速开始
pip install -r requirements.txt
python rag_qa.py

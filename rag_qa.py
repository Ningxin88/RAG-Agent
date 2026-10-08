# -*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
# 修改：Document 文档对象
from langchain_core.documents import Document

load_dotenv()

 
class RAGConfig:
    """统一配置项"""
    chunk_size = 500
    chunk_overlap = 50
    k_retrieval = 3
    retrieval_score_threshold = 0.5
    persist_directory = "./chroma_db"
    embedding_model = "all-MiniLM-L6-v2"
    # llm_model = "gpt-3.5-turbo"
    llm_model = "claude-sonnet-4-6"

    temperature = 0
    encoding = "utf-8"
 
 
def load_documents(file_path: str) -> Optional[List[Document]]:
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return None
 
    try:
        if file_path.endswith('.pdf'):
            print("检测到 PDF 文件")
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            print("检测到 Word 文件")
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            print("检测到 TXT 文件")
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="gbk")
        else:
            print("不支持的文件格式")
            print("支持格式：.pdf, .docx, .txt")
            return None
 
        documents = loader.load()
        print(f"成功加载文档：{file_path}")
        total_len = sum(len(d.page_content) for d in documents)
        print(f"文档信息：共 {len(documents)} 页/段，总字符数：{total_len}")
        return documents
 
    except Exception as e:
        print(f"加载文档失败：{str(e)}")
        return None
 
 
def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.chunk_size,
        chunk_overlap=RAGConfig.chunk_overlap,
        # 中文文档
        # separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        # 英文文档
        separators=["\n\n", "\n", ". ", "! ", "? ", ";", ",", " ", ""],
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    print("文本拆分完成")
    print(f"拆分信息：共拆分成 {len(splits)} 个片段")
    return splits
 
 
def get_or_create_vector_db(splits: List[Document]) -> Tuple[Chroma, SentenceTransformerEmbeddings]:
    print(f"\n初始化向量化模型：{RAGConfig.embedding_model}")
 
    embeddings = SentenceTransformerEmbeddings(model_name=RAGConfig.embedding_model)
 
    if os.path.exists(RAGConfig.persist_directory) and os.listdir(RAGConfig.persist_directory):
        print(f"检测到已存在的向量库，直接加载：{RAGConfig.persist_directory}")
        vectordb = Chroma(
            persist_directory=RAGConfig.persist_directory,
            embedding_function=embeddings
        )
        print(f"向量库信息：包含 {vectordb._collection.count()} 个向量")
    else:
        print(f"未检测到向量库，新建并保存：{RAGConfig.persist_directory}")
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=RAGConfig.persist_directory
        )
        vectordb.persist()
        print(f"向量库信息：包含 {vectordb._collection.count()} 个向量")
 
    return vectordb, embeddings
 
 
def create_retriever(vectordb: Chroma):
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RAGConfig.k_retrieval,
            "fetch_k": RAGConfig.k_retrieval * 3,
            "lambda_mult": 0.7
        }
    )
    print(f"检索器创建完成，检索策略：MMR，召回数量：{RAGConfig.k_retrieval}")
    return retriever
 
 
def build_rag_chain(retriever):
    print("\n初始化大模型...")
    llm = ChatOpenAI(
        model_name=RAGConfig.llm_model,
        temperature=RAGConfig.temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        timeout=45  
    )
 
    prompt_template = """
You are a professional Q&A assistant. Please answer the user's question strictly based on the following [Context Content].

【Constraints】

1. If the answer exists in the context, respond concisely and accurately, and indicate where the information is located in the document.
2. If no relevant information can be found in the context, directly reply: Based on the existing knowledge base, this question cannot be answered.
3. Do not fabricate information or use external knowledge under any circumstances.
4. Keep responses objective and refrain from adding personal opinions.

【Context Content】
{context}
【User Question】
{question}
【Answer】:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("RAG问答链构建完成")
    return qa_chain
 
 
def rag_qa(qa_chain, question: str):
    print(f"\n正在检索：'{question}'")
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
 
    print("\n" + "="*60)
    print("回答：")
    print("-"*60)
    print(answer)
    print("\n答案来源：")
    print("-"*60)
    for i, doc in enumerate(sources, 1):
        source_info = f"{i}. {doc.metadata.get('source', '未知来源')}"
        if 'page' in doc.metadata:
            source_info += f" (第 {doc.metadata['page'] + 1} 页)"
        preview = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
        print(source_info)
        print(f"   预览：{preview}")
    print("="*60)
 
 
def load_folder_documents(folder_path: str) -> List[Document]:
    if not os.path.exists(folder_path):
        print(f"文件夹不存在：{folder_path}")
        return []
 
    all_docs = []
    supported = ('.pdf', '.docx', '.txt')
    print(f"\n扫描文件夹：{folder_path}")
    for file in os.listdir(folder_path):
        fp = os.path.join(folder_path, file)
        if os.path.isfile(fp) and fp.lower().endswith(supported):
            print(f"发现文档：{file}")
            docs = load_documents(fp)
            if docs:
                all_docs.extend(docs)
    print(f"\n文件夹加载完成，共加载 {len(all_docs)} 页/段")
    return all_docs
 
 
def main():
    print("个人私有知识库 RAG 系统 v2.0")
    print("="*60)
    print()
    # 单个文件
    # DOC_PATH = r"D:\cityu_course\6004_smartcity\course\Lecture-2 Dimensionality Reduction.pdf"
    # documents = load_documents(DOC_PATH)
    # 多个文件，即一个文件夹
    documents = load_folder_documents(r"D:\cityu_course\6004_smartcity\course")

 
    if not documents:
        print("文档加载失败，程序退出")
        return
 
    splits = split_documents(documents)
    vectordb, embeddings = get_or_create_vector_db(splits)
    retriever = create_retriever(vectordb)
    qa_chain = build_rag_chain(retriever)
 
    print("\n问答交互已启动（输入 q 退出）")
    while True:
        q = input("\n请输入你的问题：")
        if q.lower() in ['q', 'quit', 'exit']:
            print("感谢使用，再见！")
            break
        if not q.strip():
            print("问题不能为空，请重新输入")
            continue
        rag_qa(qa_chain, q)
 
 
if __name__ == "__main__":
    main()
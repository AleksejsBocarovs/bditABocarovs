from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
character = "Eorlund Gray-Mane"
knowledge_dir = "RAG//"+character
filenames = [f for f in os.listdir(knowledge_dir)]
documents = []
for filename in filenames:
    path = os.path.join(knowledge_dir, filename)
    loader = TextLoader(path, encoding="utf-8")
    documents.extend(loader.load())
splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=96)
docs = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("rag_db"+"\\"+character)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import (RunnablePassthrough,RunnableLambda)
from langchain_core.output_parsers import (StrOutputParser)
from prompts import prompt
from rag import llm
from vectorstore import load_db, format_docs
import os
def get_chunker(doc_type):
    if doc_type.lower() == "ncert":
        return RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=200)
    elif doc_type.lower() == "research":
        return RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
    elif doc_type.lower() == "code":
        return RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
    return RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
def pdf_loader(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )
    loader = PyPDFLoader(file_path)
    return loader.load()
pdf_path = r"C:\Users\KARTHICK\Downloads\Deep_Learning_Final_Assignment_Report.pdf"
documents = pdf_loader(pdf_path)
doc_type = input("Document Type (ncert/research/code): ")
splitter = get_chunker(doc_type)
chunks = splitter.split_documents(documents)
print(f"Total Chunks: {len(chunks)}")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
retriever = load_db( chunks,embedder)
chain = ({"context":retriever| RunnableLambda(format_docs),"question":RunnablePassthrough()} | prompt| llm| StrOutputParser())
print("PDF Chatbot Ready")
print("Type 'exit' to quit")
while True:
    question = input("Ask: ")
    if question.lower() == "exit":
        break
    try:
        answer = chain.invoke(question)
        print("\nAnswer:\n",answer)
    except Exception as e:
        print( f"\nError: {e}\n")
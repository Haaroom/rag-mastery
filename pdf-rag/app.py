from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
#loading the file 
def pdf_loader(file):
    if not os.path.exists(file):
        print("the specified file path does not exists")
    loader = PyPDFLoader(file)
    documents=loader.load(file)
    return documents
data = pdf_loader("C:\Users\KARTHICK\Downloads\Deep_Learning_Final_Assignment_Report.pdf")
print(data[0].page_content)
#chunking 
text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
chunks=text_splitter.split_documents(data)
#embedding 
embedder = HuggingFaceEmbeddings(model='"sentence-transformers/all-mpnet-base-v2"')
#db 
db =FAISS.from_documents(chunks,embedder)
#running the rag system 
while True :
    prompt = input("ask your question or enter exit to exit :")
    if prompt.lower()=="exit":
        break
    response = db.similarity_search(prompt,k=3)
    for i, doc in enumerate(response, start=1):
        print(f"{i}th match")
        print(doc.page_content)
        print(doc.metadata)

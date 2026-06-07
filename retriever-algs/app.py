import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BM25Retriever 

load_dotenv(find_dotenv())

def similarity_search(db):
    return db.as_retriever(search_kwargs={"k": 3})

def mmr_search(db):
    # Corrected: search_type handles MMR configuration
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

def get_bm25_retriever(chunks):
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 3
    return retriever 

prompt = PromptTemplate.from_template("""
You are a helpful cloud engineer.
Context:
{context}
Question:
{question}
Answer:
"""
)
loader = PyPDFLoader(r"C:\Users\KARTHICK\Downloads\Deep_Learning_Final_Assignment_Report.pdf")
data = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30).split_documents(data)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(chunks, embedder)
similarity = similarity_search(db)
mmr = mmr_search(db)
bm25_retriever = get_bm25_retriever(chunks)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)
chain1 = {"context": similarity, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
chain2 = {"context": mmr, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
chain3 = {"context": bm25_retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
print("Similarity Search Result")
print(chain1.invoke("what is deep learning"))
print("MMR Search Result")
print(chain2.invoke("what is deep learning"))
print("BM25 Search Result")
print(chain3.invoke("what is deep learning"))

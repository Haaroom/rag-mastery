from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.llms.groq import Groq
from dotenv import load_dotenv,find_dotenv
import os
load_dotenv(find_dotenv) 
data=SimpleDirectoryReader("C:\Users\KARTHICK\Downloads\Deep_Learning_Final_Assignment_Report.pdf").load_data()
index=VectorStoreIndex(data)
llm=Groq(model="llama3-70b-8192", api_key=os.environ.get("GROQ_API_KEY"))
query_engine=index.as_query_engine(llm=llm)
while True :
    prompt = input("enter your question or 1 to quit")
    if prompt == "1":
        break
    print(query_engine.query(prompt))
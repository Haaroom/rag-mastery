from langchain_groq import ChatGroq
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY")
)
res=llm.invoke("what is machine learning " )
print(res.content)
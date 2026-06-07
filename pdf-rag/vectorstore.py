from langchain_community.vectorstores import FAISS
import os
def load_db(chunks, embedder):
    if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index",embedder,allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(chunks,embedder)
        db.save_local("faiss_index")
    return db.as_retriever(
        search_kwargs={"k":3}
    )
def format_docs(data):
    return "/n/n".join(i.page_content for i in data)
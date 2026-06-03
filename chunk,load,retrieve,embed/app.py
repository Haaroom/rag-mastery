from loader import LoaderFactory
from retriever import Retriever
from chunker import  ChunkerFactory 
from langchain_text_splitters import RecursiveCharacterTextSplitter

while True:
    input = input("enter if you want to load-chunk-embed(enter 1) or search:(enter 2) ")
    if input == "1":
        input = input("enter the file path: ")
        loader = LoaderFactory("text").get_loader()
        text = loader.load(input)
        input = input("enter the chunker type (recursive, character, spacy, nltk, markdown, sentence): ")
        chunker_factory = ChunkerFactory(input, chunk_size=1000, chunk_overlap=200)
        chunker = chunker_factory.get_chunker()
        chunks = chunker.split_text(text)
        retriever = Retriever()
        retriever.fit(chunks)
    elif input == "2":
        input = input("enter your query: ")
        results = retriever.search(input)
        for result in results:
            print(f"Chunk: {result['chunk']}\nScore: {result['score']}\n")
    else:
        print("Invalid input. Please enter 1 or 2.")
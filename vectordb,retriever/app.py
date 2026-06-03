import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="restaurants")
collection.add(
    ids=[r["id"] for r in restaurants],

    documents=[
        r["description"]
        for r in restaurants
    ],

    metadatas=[
        {
            "name":r["name"],
            "city":r["city"],
            "rating":r["rating"]
        }
        for r in restaurants
    ]
)

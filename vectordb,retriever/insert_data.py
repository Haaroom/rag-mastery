from db import collection
from data import restaurants
collection.add(
    ids=[r['id'] for r in restaurants],
    documents=[r['description'] for r in restaurants],
    metadatas=[
        {
            "name": r['name'],
            "city": r['city'],
            "rating": r['rating']
        }
        for r in restaurants
    ]
)
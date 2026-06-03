from db import collection
results = collection.query(
    query_texts=["best restaurant in new york"],
    n_results=3
)
for metadata,description in zip(results["metadatas"][0],results["documents"][0]):
    print(f"Name: {metadata['name']}")
    print(f"City: {metadata['city']}")
    print(f"Rating: {metadata['rating']}")
    print(f"Description: {description}")
#meta data filtering 
results = collection.query(
    query_texts=["best restaurant in new york"],
    n_results=3,
    where={"city":"Madurai"}
)
def search_restaurants(query, city = None):
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"city": city} if city else None
    )
    return results
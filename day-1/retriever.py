from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

        self.chunks = []
        self.embeddings = None

    def fit(self, chunks):

        self.chunks = chunks

        self.embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True
        )

    def search(
        self,
        query,
        top_k=3
    ):

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        scores = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for idx in top_indices:

            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })

        return results
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SentenceEmbeddingsBERT:
    """
    Calculate the cosine similarity between two documents using sentence
    embeddings using the "all-MiniLM-L6-v2" model
    """

    def __init__(self, documents: list):
        self.documents = documents

    def cluster(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")

        sentence_embeddings = model.encode(self.documents)

        co_sim = cosine_similarity(
            [sentence_embeddings[0]], [sentence_embeddings[1]]
        )

        return co_sim[0][0]

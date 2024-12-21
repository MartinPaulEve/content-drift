import pathlib

import openai
from sklearn.metrics.pairwise import cosine_similarity


class TransformerOpenAI:
    """
    Implements a transformer model using the OpenAI API.
    Requires an OpenAI API key to be stored in ~/.openai
    """

    def __init__(self, documents: list):
        self.documents = documents

    def cluster(self):
        openai.api_key = (
            pathlib.Path("~/.openai").expanduser().read_text().strip()
        )

        embeddings = []
        for doc in self.documents:
            response = openai.Embedding.create(
                input=doc, engine="text-embedding-3-small"
            )
            embeddings.append(response["data"][0]["embedding"])

        co_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
        return co_sim[0][0]

class Cosine:
    """
    A class for implementing the cosine similarity algorithm using tf-idf as the
    vector
    """

    def __init__(self, documents):
        self.documents = documents

    def cluster(self):
        from sklearn.feature_extraction.text import (
            TfidfVectorizer,
            CountVectorizer,
        )
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import linear_kernel

        # import pandas as pd

        # Vectorize the documents

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.documents)

        # identical results to the sklearn implementation
        # co_sim = (tfidf_matrix * tfidf_matrix.T).A[0, 1]

        # identical results to the linear kernel function
        # co_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
        # return co_sim[0][1]

        co_sim = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()

        return co_sim[1]

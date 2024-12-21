from cmath import exp


class Euclidean:
    """
    This class implements a Euclidean distance algorithm for document clustering
    """

    def __init__(self, documents):
        """
        Initialize the Euclidian algorithm.

        :param documents: a list of documents to cluster
        """
        self.documents = documents

    def cluster(self):
        from sklearn.feature_extraction.text import CountVectorizer
        from scipy.spatial.distance import euclidean

        # Vectorize the documents
        vectorizer = CountVectorizer()
        fit_transform_matrix = vectorizer.fit_transform(
            [self.documents[0], self.documents[1]]
        )
        matrix_array = fit_transform_matrix.toarray()

        # Calculate Euclidean distance
        euclidean_dist = euclidean(matrix_array[0], matrix_array[1])

        return [euclidean_dist]

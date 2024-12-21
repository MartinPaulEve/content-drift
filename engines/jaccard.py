class Jaccard:
    """
    This class implements the Jaccard similarity approach for document
    clustering.

    See Broder, Andrei Z., Steven C. Glassman, Mark S. Manasse, and Geoffrey
    Zweig, ‘Syntactic Clustering of the Web’, Computer Networks and ISDN
    Systems, Papers from the Sixth International World Wide Web Conference,
    29.8 (1997), pp. 1157–66, doi:10.1016/S0169-7552(97)00031-7
    """

    def __init__(self, documents):
        """
        Initialize the Broder algorithm.

        :param documents: a list of documents to cluster
        :param k: the number of shingles to use in each document
        :param l: the number of hash functions to use
        """
        self.documents = documents
        self.k = 10
        self.shingles = self._shingle_documents()

    def _shingle_documents(self):
        """
        Create shingles for each document.

        :return: a list of sets of shingles
        """
        shingles = []
        for document in self.documents:
            shingles.append(set())
            for i in range(len(document) - self.k + 1):
                shingle = document[i : i + self.k]
                shingles[-1].add(shingle)
        return shingles

    def _intersection(self):
        """
        Find the intersection of the shingles.

        :return: a list of intersections
        """
        intersections = []
        for i in range(len(self.shingles)):
            for j in range(i + 1, len(self.shingles)):
                intersections.append(len(self.shingles[i] & self.shingles[j]))
        return intersections

    def _union(self):
        """
        Find the union of the shingles.

        :return: a list of unions
        """
        unions = []
        for i in range(len(self.shingles)):
            for j in range(i + 1, len(self.shingles)):
                unions.append(len(self.shingles[i] | self.shingles[j]))
        return unions

    def cluster(self):
        """
        Return the Jaccard similarity of the documents' shingles.
        """
        return [
            intersection / union
            for intersection, union in zip(self._intersection(), self._union())
        ]

    def containment(self):
        """
        Return the containment of the documents' shingles.
        """
        return [
            intersection / len(self.shingles[i])
            for i, intersection in enumerate(self._intersection())
        ]

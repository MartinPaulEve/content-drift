import numpy as np


class WordEmbedding:
    """
    Calculate the cosine similarity between two documents using Word2Vec
    """

    def __init__(self, documents: list):
        self.documents = documents

    @staticmethod
    def _vectorize(w2v_model, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        from nltk.corpus import stopwords

        stop_words = stopwords.words("english")

        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in stop_words]

        word_vecs = []
        for word in words:
            try:
                vec = w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return vector

    @staticmethod
    def _cosine_sim(vec_a, vec_b):
        """Find the cosine similarity distance between two vectors."""
        co_sim = np.dot(vec_a, vec_b) / (
            np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        )
        if np.isnan(np.sum(co_sim)):
            return 0
        return co_sim

    def cluster(self):
        from gensim.models.keyedvectors import KeyedVectors

        # model_path = "./engines/GoogleNews-vectors-negative300.bin"
        # w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

        model_path = "./engines/glove.6B.300d.w2vformat.txt"
        w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

        doc_a_vec = self._vectorize(w2v_model, self.documents[0])
        doc_b_vec = self._vectorize(w2v_model, self.documents[1])

        sim_score = self._cosine_sim(doc_a_vec, doc_b_vec)

        return sim_score

import importlib
import pathlib

import typer
from typing_extensions import Annotated


from timeit import default_timer as timer

app = typer.Typer()


def _pack_documents(document_a, document_b):
    document_a_text = pathlib.Path(document_a).read_text()
    document_b_text = pathlib.Path(document_b).read_text()

    return [document_a_text, document_b_text]


@app.command()
def jaccard_similarity(
    document_a: Annotated[str, typer.Argument(help="Document A")],
    document_b: Annotated[str, typer.Argument(help="Document B")],
):
    """
    Calculate the Jaccard similarity between two documents
    """
    from engines import jaccard

    jaccard_object = jaccard.Jaccard(
        documents=_pack_documents(document_a, document_b)
    )
    print(jaccard_object.cluster())


@app.command()
def jaccard_containment(
    document_a: Annotated[str, typer.Argument(help="Document A")],
    document_b: Annotated[str, typer.Argument(help="Document B")],
):
    """
    Calculate the Jaccard containment for document b in document a
    """
    from engines import jaccard

    jaccard_object = jaccard.Jaccard(
        documents=_pack_documents(document_a, document_b)
    )
    print(jaccard_object.containment())


@app.command()
def euclidean_distance(
    document_a: Annotated[str, typer.Argument(help="Document A")],
    document_b: Annotated[str, typer.Argument(help="Document B")],
):
    """
    Calculate the Euclidean distance between two documents
    """
    from engines import euclidean

    euclidean_object = euclidean.Euclidean(
        documents=_pack_documents(document_a, document_b)
    )
    print(euclidean_object.cluster())


@app.command()
def cosine_similarity(
    document_a: Annotated[str, typer.Argument(help="Document A")],
    document_b: Annotated[str, typer.Argument(help="Document B")],
):
    """
    Calculate the cosine similarity between two documents
    """
    from engines import cosine

    cosine_object = cosine.Cosine(
        documents=_pack_documents(document_a, document_b)
    )
    print(cosine_object.cluster())


@app.command()
def cosine_similarity_text(
    string_a: Annotated[str, typer.Argument(help="String A")],
    string_b: Annotated[str, typer.Argument(help="String B")],
):
    """
    Calculate the cosine similarity between two text inputs
    """
    from engines import cosine

    cosine_object = cosine.Cosine(documents=[string_a, string_b])
    print(cosine_object.cluster())


@app.command()
def word_embedding(
    document_a: Annotated[str, typer.Argument(help="Document A")],
    document_b: Annotated[str, typer.Argument(help="Document B")],
):
    """
    Calculate the cosine similarity between two documents using Word2Vec
    """
    from engines import word_embeddings

    word_embedding_object = word_embeddings.WordEmbedding(
        documents=_pack_documents(document_a, document_b)
    )
    print(word_embedding_object.cluster())


@app.command()
def sentence_embedding(
    document_a: Annotated[str, typer.Argument(help="Document A")],
    document_b: Annotated[str, typer.Argument(help="Document B")],
):
    """
    Calculate the cosine similarity between two documents using sentence
    embedding
    """

    from engines import sentence_embeddings_bert

    sentence_embedding_object = sentence_embeddings_bert.SentenceEmbeddingsBERT(
        documents=_pack_documents(document_a, document_b)
    )
    print(sentence_embedding_object.cluster())


@app.command()
def transformer_openai(
    document_a: Annotated[str, typer.Argument(help="Document A")],
    document_b: Annotated[str, typer.Argument(help="Document B")],
):
    """
    Calculate the cosine similarity between two documents using OpenAI's
    transformer
    """
    from engines import transformer_openai as transformer_open_ai

    transformer_openai_object = transformer_open_ai.TransformerOpenAI(
        documents=_pack_documents(document_a, document_b)
    )
    print(transformer_openai_object.cluster())


@app.command()
def preprocess(
    document: Annotated[str, typer.Argument(help="Document")],
):
    """
    Preprocess an HTML document into plaintext
    """
    from engines import preprocessor

    preprocess_object = preprocessor.PreProcessor(document=document)
    print(preprocess_object.process())


@app.command()
def all_tests():
    """
    Run all tests
    """
    test_list = {
        "Jaccard Similarity": ["jaccard", "Jaccard"],
        "Euclidean Distance": ["euclidean", "Euclidean"],
        "Cosine Similarity": ["cosine", "Cosine"],
        "Word Embedding": ["word_embeddings", "WordEmbedding"],
        "Sentence Embedding": [
            "sentence_embeddings_bert",
            "SentenceEmbeddingsBERT",
        ],
        "Transformer OpenAI": ["transformer_openai", "TransformerOpenAI"],
    }

    test_data = {
        "Identical Test": [
            "./test_data/identical_test/document_a.txt",
            "./test_data/identical_test/document_b.txt",
        ],
        "Long Test": [
            "./test_data/long_test/document_a.txt",
            "./test_data/long_test/document_b.txt",
        ],
        "Paragraph Test": [
            "./test_data/paragraph_test/document_a.txt",
            "./test_data/paragraph_test/document_b.txt",
        ],
        "Synonym Test": [
            "./test_data/synonym_test/document_a.txt",
            "./test_data/synonym_test/document_b.txt",
        ],
        "Typo Test": [
            "./test_data/typo_test/document_a.txt",
            "./test_data/typo_test/document_b.txt",
        ],
    }

    for test_name, test_class in test_list.items():
        for test_data_name, test_data_files in test_data.items():
            run_tests(test_name, test_data_name, test_class, test_data_files)


@app.command()
def containment_tests():
    """
    Run containment tests
    """
    test_list = {
        "Jaccard Containment": ["jaccard", "Jaccard"],
    }

    test_data = {
        "Identical Test": [
            "./test_data/identical_test/document_a.txt",
            "./test_data/identical_test/document_b.txt",
        ],
        "Long Test": [
            "./test_data/long_test/document_a.txt",
            "./test_data/long_test/document_b.txt",
        ],
        "Paragraph Test": [
            "./test_data/paragraph_test/document_a.txt",
            "./test_data/paragraph_test/document_b.txt",
        ],
        "Synonym Test": [
            "./test_data/synonym_test/document_a.txt",
            "./test_data/synonym_test/document_b.txt",
        ],
        "Typo Test": [
            "./test_data/typo_test/document_a.txt",
            "./test_data/typo_test/document_b.txt",
        ],
        "Containment Test": [
            "./test_data/containment_test/document_a.txt",
            "./test_data/containment_test/document_b.txt",
        ],
    }

    for test_name, test_class in test_list.items():
        for test_data_name, test_data_files in test_data.items():
            run_tests(
                test_name,
                test_data_name,
                test_class,
                test_data_files,
                containment=True,
            )


def run_tests(
    test_name, test_data_name, test_class, test_data_files, containment=False
):
    print("----------------------------------------------------------")
    print(f"Running test {test_name}: {test_data_name}")
    try:
        start = timer()
        module = importlib.import_module(f"engines.{test_class[0]}")
        class_ = getattr(module, f"{test_class[1]}")
        test_object = class_(_pack_documents(*test_data_files))
        if not containment:
            result = test_object.cluster()
        else:
            result = test_object.containment()
        end = timer()
        print(f"Result: {result}")
        print(f"Runtime: {end - start} seconds")

    except Exception as e:
        print(f"Error: {e}")

    print("----------------------------------------------------------")


if __name__ == "__main__":
    app()

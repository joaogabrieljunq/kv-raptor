import time
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import Document, VectorStoreIndex, StorageContext

class Retriever:
    def __init__(self, embed_model, vector_store):
        """
        Initialize the retriever using an embedding model and a vector store.
        """
        self.embed_model = embed_model
        self.retriever = vector_store

    def retrieve(self, query_text):
        """
        Perform retrieval for a given query string.

        Returns:
            context (str): Concatenated text from top-k retrieved nodes.
            retriever_time (float): Time taken for retrieval step.
            retrieved_nodes (List[Node]): List of retrieved document nodes.
        """
        query_vector = self.embed_model.get_query_embedding(query_text)
        query_obj = VectorStoreQuery(query_embedding=query_vector, similarity_top_k=20)

        start_time = time.time()
        results = self.retriever.query(query_obj)
        end_time = time.time()

        context = "\n".join(node.get_content() for node in results.nodes)
        retriever_time = end_time - start_time
        return context, retriever_time, results.nodes
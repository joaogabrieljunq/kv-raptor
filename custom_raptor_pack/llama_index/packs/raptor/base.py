from typing import Any, Dict, List, Optional

import asyncio
from enum import Enum
import torch
import uuid

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_retriever import BaseRetriever, QueryType
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import (
    BaseNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
    TransformComponent,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    BasePydanticVectorStore,
)
from llama_index.packs.raptor.clustering import get_clusters


device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the provided text, including as many key details as needed."
)

class QueryModes(str, Enum):
    tree_traversal = "tree_traversal"
    collapsed = "collapsed"

class SummaryModule(BaseModel):
    response_synthesizer: BaseSynthesizer = Field(description="LLM")
    summary_prompt: str = Field(
        default=DEFAULT_SUMMARY_PROMPT,
        description="Summary prompt.",
    )
    num_workers: int = Field(
        default=4, description="Number of workers to generate summaries."
    )
    show_progress: bool = Field(default=True, description="Show progress.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: Optional[LLM] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        num_workers: int = 4,
    ) -> None:
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", use_async=True, llm=llm
        )
        super().__init__(
            response_synthesizer=response_synthesizer,
            summary_prompt=summary_prompt,
            num_workers=num_workers,
        )

    async def generate_summaries(self, documents_per_cluster: List[List[BaseNode]]) -> List[str]:
        """Generate summaries of documents per cluster using concurrent async processing."""
        print("Run optimized summarization")
        jobs = []
        for documents in documents_per_cluster:
            with_scores = [NodeWithScore(node=doc, score=1.0) for doc in documents]
            jobs.append(self.response_synthesizer.asynthesize(self.summary_prompt, with_scores))
    
        # Limit concurrency using Semaphore
        semaphore = asyncio.Semaphore(self.num_workers)
    
        async def worker(job):
            async with semaphore:
                return await job
    
        # Execute all jobs concurrently with num_workers limit
        responses = await asyncio.gather(*[worker(job) for job in jobs])
    
        return [str(response) for response in responses]


class RaptorRetriever(BaseRetriever):
    """
    A modified retriever that accepts a list of chunk_documents and ids,
    computes embeddings in batch, and either uses the Chroma client's direct add method
    (naive rag) or builds a Raptor tree with clustering if use_direct_add is False.
    """
    def __init__(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        embed_model: Optional[BaseEmbedding] = None,
        llm: Optional[Any] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        similarity_top_k: int = 2,
        mode: QueryModes = "collapsed",
        verbose: bool = True,
        use_direct_add: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.similarity_top_k = similarity_top_k
        self.mode = mode
        self._verbose = verbose  # used internally in insert()

        if documents and use_direct_add:
            # Directly add documents to the Chroma collection (naive rag mode)
            if ids is None:
                ids = [f"chunk_{i}" for i in range(len(documents))]
            print("Adding documents directly to Chroma collection using direct add method...")
            embeddings = embed_model.get_text_embedding_batch(
                documents, device=device, show_progress=True
            )
            # Here we assume vector_store.client is the Chroma client and it has a get_or_create_collection method.
            collection_raptor = self.vector_store.client
            print(collection_raptor)
            collection_raptor.add(
                documents=documents,
                ids=ids,
                embeddings=embeddings
            )
        else:
            # Build the Raptor tree with clustering.
            # First, initialize the index and required parameters.
            transformations = kwargs.get("transformations", [])
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=StorageContext.from_defaults(vector_store=vector_store),
                embed_model=embed_model,
                transformations=transformations,
            )
            self.tree_depth = kwargs.get("tree_depth", 3)
            self.summary_module = kwargs.get("summary_module", SummaryModule(llm=llm))
            # Convert raw documents to TextNodes.
            if ids is None:
                ids = [f"chunk_{i}" for i in range(len(documents))]
            text_nodes = [TextNode(text=doc, id_=id_) for doc, id_ in zip(documents, ids)]
            asyncio.run(self.insert(text_nodes))

    def _get_embeddings_per_level(self, level: int = 0) -> List[float]:
        """Retrieve embeddings per level in the abstraction tree."""
        filters = MetadataFilters(filters=[MetadataFilter("level", level)])
        # Using the retriever from self.index to retrieve a large number of nodes.
        source_nodes = self.index.as_retriever(
            similarity_top_k=10000, filters=filters
        ).retrieve("retrieve")
        return [x.node for x in source_nodes]

    async def insert(self, documents: List[BaseNode]) -> None:
        """Insert documents into the hierarchical index, creating clusters and summaries at each level."""
        embed_model = self.index._embed_model
        transformations = self.index._transformations

        cur_nodes = run_transformations(documents, transformations, in_place=False)
        for level in range(self.tree_depth):
            if self._verbose:
                print(f"Generating batching embeddings for level {level}.")
            cur_nodes_text = [
                node.get_content() if hasattr(node, "get_content") else str(node)
                for node in cur_nodes
            ]
            print("Batch custom embedding")
            embeddings = embed_model.get_text_embedding_batch(
                cur_nodes_text,
                device=device,
                show_progress=True
            )
            assert len(embeddings) == len(cur_nodes)
            id_to_embedding = {
                node.id_: embedding for node, embedding in zip(cur_nodes, embeddings)
            }

            if self._verbose:
                print(f"Performing clustering for level {level}.")
            # Cluster the documents.
            nodes_per_cluster = get_clusters(cur_nodes, id_to_embedding)

            if self._verbose:
                print(f"Generating summaries for level {level} with {len(nodes_per_cluster)} clusters.")
            summaries_per_cluster = await self.summary_module.generate_summaries(nodes_per_cluster)

            if self._verbose:
                print(f"Level {level} created summaries/clusters: {len(nodes_per_cluster)}")
            # Create summary nodes for each cluster.
            new_nodes = [
                TextNode(
                    text=summary,
                    metadata={"level": level},
                    excluded_embed_metadata_keys=["level"],
                    excluded_llm_metadata_keys=["level"],
                    id_=f"summary_{level}_{uuid.uuid4().hex[:8]}"
                )
                for summary in summaries_per_cluster
            ]

            nodes_with_embeddings = []
            for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
                for node in cluster:
                    node.metadata["parent_id"] = summary_doc.id_
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    
                    # Retrieve the embedding using the current (original) node id.
                    if node.id_ in id_to_embedding:
                        embedding = id_to_embedding[node.id_]
                    else:
                        text = node.get_content() if hasattr(node, "get_content") else str(node)
                        embedding = embed_model.get_text_embedding(text)
                    
                    # Generate a new unique id for the node.
                    node.id_ = f"node_{uuid.uuid4().hex[:8]}"
                    node.embedding = embedding
                    
                    nodes_with_embeddings.append(node)

            # Deduplicate nodes based on their new IDs before insertion.
            unique_nodes = []
            seen_ids = set()
            for node in nodes_with_embeddings:
                if node.id_ not in seen_ids:
                    unique_nodes.append(node)
                    seen_ids.add(node.id_)
            
            self.index.insert_nodes(unique_nodes)
            # Set current nodes to the new summary nodes for the next level.
            cur_nodes = new_nodes

        self.index.insert_nodes(cur_nodes)

    async def collapsed_retrieval(self, query_str: str) -> Response:
        """Query the index as a collapsed tree (a single pool of nodes)."""
        return await self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        ).aretrieve(query_str)

    async def tree_traversal_retrieval(self, query_str: str) -> Response:
        """Query the index as a tree, traversing from the top down."""
        parent_ids = None
        selected_node_ids = set()
        selected_nodes = []
        level = self.tree_depth - 1
        while level >= 0:
            if parent_ids is None:
                nodes = await self.index.as_retriever(
                    similarity_top_k=self.similarity_top_k,
                    filters=MetadataFilters(filters=[MetadataFilter(key="level", value=level)]),
                ).aretrieve(query_str)
                for node in nodes:
                    if node.id_ not in selected_node_ids:
                        selected_nodes.append(node)
                        selected_node_ids.add(node.id_)
                parent_ids = [node.id_ for node in nodes]
                if self._verbose:
                    print(f"Retrieved parent IDs from level {level}: {parent_ids!s}")
            elif parent_ids is not None and len(parent_ids) > 0:
                nested_nodes = await asyncio.gather(
                    *[
                        self.index.as_retriever(
                            similarity_top_k=self.similarity_top_k,
                            filters=MetadataFilters(filters=[MetadataFilter(key="parent_id", value=id_)])
                        ).aretrieve(query_str)
                        for id_ in parent_ids
                    ]
                )
                nodes = [node for nested in nested_nodes for node in nested]
                for node in nodes:
                    if node.id_ not in selected_node_ids:
                        selected_nodes.append(node)
                        selected_node_ids.add(node.id_)
                if self._verbose:
                    print(f"Retrieved {len(nodes)} from parents at level {level}.")
                level -= 1
                parent_ids = None
        return selected_nodes

    async def aretrieve(self, query_str_or_bundle: QueryType, mode: Optional[QueryModes] = None) -> List[NodeWithScore]:
        """Retrieve results by querying the Chroma collection directly."""
        if isinstance(query_str_or_bundle, QueryBundle):
            query_str = query_str_or_bundle.query_str
        else:
            query_str = query_str_or_bundle
        query_embedding = self.embed_model.get_text_embedding(query_str)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.vector_store.client.query(
                query_embeddings=[query_embedding],
                n_results=self.similarity_top_k
            )
        )
        nodes_with_scores = []
        for doc, id_val, distance in zip(results["documents"][0], results["ids"][0], results["distances"][0]):
            node = TextNode(text=doc, id_=id_val)
            nodes_with_scores.append(NodeWithScore(node=node, score=distance))
        return nodes_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        # not used, needed for type checking

    def retrieve(self, query_str_or_bundle: QueryType, mode: Optional[QueryModes] = None) -> List[NodeWithScore]:
        if isinstance(query_str_or_bundle, QueryBundle):
            query_str = query_str_or_bundle.query_str
        else:
            query_str = query_str_or_bundle
        return asyncio.run(self.aretrieve(query_str, mode=mode or self.mode))

    def persist(self, persist_dir: str) -> None:
        self.index.storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def from_persist_dir(
        cls: "RaptorRetriever",
        persist_dir: str,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> "RaptorRetriever":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return cls(
            [],
            existing_index=load_index_from_storage(storage_context, embed_model=embed_model),
            **kwargs,
        )

class RaptorPack(BaseLlamaPack):
    """
    The modified RaptorPack now accepts chunk_documents and ids,
    and instantiates the retriever in full Raptor (tree) mode when use_direct_add is False.
    """
    def __init__(
        self,
        documents: List[str],
        ids: List[str],
        llm: Optional[Any] = None,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        similarity_top_k: int = 2,
        mode: QueryModes = "collapsed",
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self.retriever = RaptorRetriever(
            documents,
            ids=ids,
            embed_model=embed_model,
            llm=llm,
            vector_store=vector_store,
            similarity_top_k=similarity_top_k,
            mode=mode,
            verbose=verbose,
            use_direct_add=False,  # Run full clustering and tree-building.
            **kwargs,
        )

    def get_modules(self) -> Dict[str, Any]:
        return {"retriever": self.retriever}

    def run(self, query: str, mode: Optional[QueryModes] = None) -> Any:
        return self.retriever.retrieve(query, mode=mode)
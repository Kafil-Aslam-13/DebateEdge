"""Pinecone retrieval; Rag 
presistant vector sgtore:
cross-session , cloud hosted , survives restarts,
Uses Coherence embeddings for richer semantic understanding
in this project pinecone holds a larger argument database
that presists across all sessions. ChromaDb holds  only whats needed for current session.
"""

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from src.core.config import get_settings
from src.core.logger import get_logger
from src.retrieval.embeddings import get_cohere_embeddings
logger=get_logger(__name__)
_INDEX_NAME = "debateedge-arguments"
_NAMESPACE = "arguments"

class DebatePineconeDB:
    """Uses PineCone + cohere embeddings
    Fallsback if api of pineback not configured"""

    def __init__(self):
        self._store:PineconeVectorStore | None=None
        self._available = False
        self._try_connect()


    def _try_connect(self)->None:
        """Attempt to connect to pinecone. Degrades gracefully on failure."""
        settings=get_settings()
        if not settings.pinecone_api_key:
            logger.warning(
                "PINECONE_API_KEY not set — "
                "Pinecone retrieval disabled. "
                "Using ChromaDB only."
            )
            return

        try:
            from pinecone import Pinecone,ServerlessSpec
            pc=Pinecone(api_key=settings.pinecone_api_key)
            # create index  if it doesnt exist 
            existing = [idx.name for idx in pc.list_indexes()]

            if _INDEX_NAME not in existing:
                logger.info(f"Creating pinecone index: {_INDEX_NAME}")
                pc.create_index(name=_INDEX_NAME,
                                dimension=1024, # Cohere embed-English-v3.0 dimention
                                metric="cosine",
                                spec=ServerlessSpec(
                                    cloud="aws",
                                    region="us-east-1"
                                ))
            embeddings=get_cohere_embeddings()

            self._store = PineconeVectorStore(
                index_name=_INDEX_NAME,
                embedding=embeddings,
                namespace=_NAMESPACE
            )

            self._available = True
            logger.info(f"pinecone connected | index{_INDEX_NAME}")
        except Exception as e:
            logger.warning(
                f"Pinecone connection failed: {e} — "
                "continuing without Pinecone retrieval."
            )
            self._available = False

    def is_available(self)->None:
        "return true if pinecone connected else false"
        return self._available

    def upsert_documents(self,documents:list[Document])->None:
        """Add documents to Pinecone index.

        Used to populate the persistent argument database.
        Safe to call multiple times — Pinecone deduplicates by content hash.
        """
        if not self._available or self._store is None:
            logger.warning("Logger not available - upsert skipped")
            return

        try:
            self._store.add_documents(documents)
            logger.info(f"pinecone:{len(documents)} documentsd added")
        except Exception as e:
            logger.warning(f"pinecone upsert failed")

    def retrieve_similar(
        self,
        query: str,
        k: int = 3,
    ) -> list[Document]:
        """Retrieve k most similar documents using similarity search.

        Returns empty list if Pinecone not available.
        """
        if not self._available or not query.strip():
            return []

        try:
            results = self._store.similarity_search(query=query, k=k)
            logger.info(
                f"Pinecone similarity | "
                f"query='{query[:50]}' | found={len(results)}"
            )
            return results

        except Exception as e:
            logger.warning(f"Pinecone similarity search failed: {e}")
            return []

    def retrieve_mmr(
        self,
        query: str,
        k: int = 3,
        fetch_k: int = 10,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """Retrieve k diverse documents using MMR.

        Returns empty list if Pinecone not available.
        Same strategy as ChromaDB MMR but against persistent index.
        """
        if not self._available or not query.strip():
            return []

        try:
            results = self._store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
            logger.info(
                f"Pinecone MMR | "
                f"query='{query[:50]}' | found={len(results)}"
            )
            return results

        except Exception as e:
            logger.warning(f"Pinecone MMR search failed: {e}")
            return []


        
    def as_retriever(self, search_type: str = "mmr", k: int = 3):
        """Return LangChain retriever interface for LCEL chains."""
        if not self._available or self._store is None:
            return None

        if search_type == "mmr":
            return self._store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 3,
                    "lambda_mult": 0.5,
                },
            )

        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )


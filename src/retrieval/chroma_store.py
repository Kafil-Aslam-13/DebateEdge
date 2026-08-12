"""ChromaDB vector store for RAG
VECTOR STORE vs VECTOR DATABASE:
ChromaDB here = VECTOR STORE
  - in-memory, local
  - session-scoped
  - fast, no network
  - throwaway

Pinecone (pinecone_db.py) = VECTOR DATABASE
  - cloud, persistent
  - cross-session
  - slower (network), more scalable
  - survives restarts
  
  TWO RETRIEVAL STRATEGIES:
1. similarity_search  → finds most similar documents
   use when: you want the closest match to the query

2. MMR (Max Marginal Relevance)  → finds diverse documents
   use when: you want variety — not 3 near-identical results
   lambda_mult controls diversity: 0.0=max diversity, 1.0=max similarity
   We use 0.5 — balanced between relevant AND diverse

WHY MMR FOR DEBATE:
Debate needs diverse counterarguments not repetitive ones.
Similarity alone might return 3 versions of the same point.
MMR ensures each retrieved argument adds new information."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.core.exceptions import RetrievalError
from src.core.logger import get_logger
from src.retrieval.embeddings import get_hf_embeddings

logger = get_logger(__name__)


_ARGUMENT_KNOWLEDGE_BASE = [
    # Social media
    Document(
        page_content=(
            "Multiple peer-reviewed studies show excessive social media use "
            "correlates with increased anxiety, depression, and loneliness, "
            "particularly in teenagers aged 13-17."
        ),
        metadata={"topic": "social media", "type": "evidence", "strength": "strong"},
    ),
    Document(
        page_content=(
            "Social media platforms have enabled global social movements "
            "including Arab Spring, #MeToo, and climate activism, "
            "connecting millions of activists across borders."
        ),
        metadata={"topic": "social media", "type": "counterevidence", "strength": "strong"},
    ),
    Document(
        page_content=(
            "Facebook's internal research leaked in 2021 showed the company "
            "knew Instagram was harmful to teenage girls but continued "
            "optimising for engagement metrics over user wellbeing."
        ),
        metadata={"topic": "social media", "type": "evidence", "strength": "strong"},
    ),
    # AI
    Document(
        page_content=(
            "The World Economic Forum projects AI will displace 85 million jobs "
            "by 2025 but create 97 million new roles, resulting in a net "
            "positive of 12 million jobs globally."
        ),
        metadata={"topic": "ai jobs", "type": "evidence", "strength": "strong"},
    ),
    Document(
        page_content=(
            "Historical technological revolutions — steam engine, electricity, "
            "computers — all initially displaced workers but ultimately created "
            "far more jobs than they eliminated over 20-30 year horizons."
        ),
        metadata={"topic": "ai jobs", "type": "historical", "strength": "moderate"},
    ),
    # Climate
    Document(
        page_content=(
            "The IPCC Sixth Assessment Report (2021) confirms with 'unequivocal' "
            "certainty that human-caused greenhouse gas emissions are the dominant "
            "driver of global warming since the 1850s."
        ),
        metadata={"topic": "climate", "type": "evidence", "strength": "strong"},
    ),
    Document(
        page_content=(
            "Renewable energy costs have dropped 90% in the last decade. "
            "Solar power is now the cheapest source of electricity in history "
            "according to the International Energy Agency (2020)."
        ),
        metadata={"topic": "climate", "type": "evidence", "strength": "strong"},
    ),
    # Universal healthcare
    Document(
        page_content=(
            "Countries with universal healthcare systems, including Canada, UK, "
            "and Germany, consistently outperform the US on key health outcomes "
            "including life expectancy and infant mortality, at lower per-capita cost."
        ),
        metadata={"topic": "healthcare", "type": "evidence", "strength": "strong"},
    ),
    # Education
    Document(
        page_content=(
            "Finland's education system, which eliminated standardised testing "
            "and homework until age 15, consistently ranks among the world's "
            "best, suggesting high pressure does not improve outcomes."
        ),
        metadata={"topic": "education", "type": "evidence", "strength": "strong"},
    ),
    # General debate principles
    Document(
        page_content=(
            "Correlation does not imply causation. Observing that two variables "
            "move together does not prove one causes the other without "
            "controlling for confounding variables."
        ),
        metadata={"topic": "general", "type": "principle", "strength": "strong"},
    ),
    Document(
        page_content=(
            "Anecdotal evidence, while compelling emotionally, cannot substitute "
            "for peer-reviewed research and statistically significant data "
            "when making policy arguments."
        ),
        metadata={"topic": "general", "type": "principle", "strength": "strong"},
    ),
]

class DebateChromaStore:
    """In session vector store for RAG debate responses.
    Loaded once per session with the pre-build knowledge base.
    Uses HF Embeddings 
    supports both similarity and MMR retrieval"""

    def __init__(self)->None:
        self._store:Chroma | None = None
        logger.info("DebateChromaStore initialized")

    def _get_store(self)->Chroma:
        """Build ChromaDB store lazily on first retrieval call."""
        if self._store is None:
            logger.info("Knowledge Base Loading in ChromaDB")
            embeddings=get_hf_embeddings()
            self._store = Chroma.from_documents(
                documents=_ARGUMENT_KNOWLEDGE_BASE,
                embedding=embeddings,
                collection_name="debate_knowledge"
            )
            logger.info(
                f"ChromaDb ready | "
                f"{len(_ARGUMENT_KNOWLEDGE_BASE)} documents loaded"
            )
        return self._store

    def retrieve_similar(self,query:str,k:int=3,)->list[Document]:
        """retrieves k most similar documents using similariity search.
        Use when : we want closest match to query
        good for finding most relevant evidence or counter arguments"""
        if not query.strip():
            return []
        try:
            store = self._get_store()
            result=store.similarity_search(query=query,k=k)
            logger.info(f"Chroma similarity| query={query[:50]} | "
                        f"found={len(result)}")
            return result
        except Exception as e:
            logger.warning(f"ChromaDb Similarity Search Failed: {e}")
            raise RetrievalError(f"ChromaDB retrieval failed: {e}") from e

    def retrieve_mmr(
            self,query:str,k:int = 3 , fetch_k: int = 10,
            lambda_mult:float =0.5
    ) -> list[Document]:
        """ARetrieves k diverse documents using max Margional relevance
        used when we want diverse counterarguments , not repetative ones.
        Args:
            query:       The argument to find counterevidence for
            k:           Number of documents to return
            fetch_k:     Candidate pool size before MMR reranking
            lambda_mult: 0.0=max diversity, 1.0=max similarity (0.5=balanced)"""

        if not query.strip():
            return []
        try:
            store=self._get_store()
            results=store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            logger.info(f"ChromaDB MMR | query = {query}"
                        f"found={len(results)} diverse  results")
            return results
        except Exception as e:
            logger.warning(f"ChromaDB MMR search failed: {e}")
            raise RetrievalError(f"Chromadb mmr failed: {e}") from e

    def as_retriever(self,search_type:str = "mmr",k:int=3):
        """Returns a Langchain Retriever Interface.
        used when plugged into lecl chains direcly"""

        store = self._get_store()
        if search_type=="mmr":
            return store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k":k,
                    "fetch_k":k*3,
                    "lambda_mult":0.5,
                },
            )
        return store.as_retriever(search_type="similarity",search_kwargs={"k":k})

    def clear(self)->None:
        if self._store is not None:
            try:
                self._store.delete_collection()
            except Exception as e:
                logger.warning(f"Chromadb clear failed: {e}")
            finally:
                self._store=None
        logger.info("chromadb cleared")


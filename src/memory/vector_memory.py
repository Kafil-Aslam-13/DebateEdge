"""Vector memory for debate arguments.

Stores user arguments as embeddings and retrieves semantically
similar arguments from the current debate session.

Architecture:

    User argument
          ↓
    HuggingFace Embeddings
          ↓
        Chroma
          ↓
    semantic similarity search

:
    - In-memory Chroma collection
    - Local HuggingFace embeddings
    - Session-scoped memory
    - Semantic similarity search
    - Metadata for quality, score, turn, and fallacy

Future Sprint:
    - Replace in-memory Chroma with a persistent vector database
      such as Pinecone for cross-session memory.
"""

from __future__ import annotations

from typing import Any


from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.logger import get_logger

logger = get_logger(__name__)


# Small, fast, local embedding model.
# No API key required.
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DebateVectorMemory:
    """Semantic memory for arguments made during a debate session.

    Unlike buffer memory:

        Buffer → retrieves recent messages.

    Unlike summary memory:

        Summary → compresses older conversation.

    Vector memory:

        Vector → retrieves arguments based on semantic similarity.

    Example:

        User previously said:
            "Social media makes teenagers addicted to their phones."

        Current argument:
            "Apps encourage young people to become dependent on
             their phones."

        Vector memory can recognise these as semantically similar
        even though the wording is different.
    """

    def __init__(self, top_k: int = 3) -> None:
        """
        Args:
            top_k:
                Maximum number of similar arguments to retrieve.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.top_k = top_k

        # Lazy initialisation.
        #
        # We don't load the embedding model or create Chroma until
        # the first argument actually needs to be stored.
        self._store: Chroma | None = None

        # Track number of documents ourselves.
        self._doc_count = 0

        logger.info(
            f"VectorMemory initialised | top_k={top_k}"
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def _get_store(self) -> Chroma:
        """Create the Chroma store lazily."""

        if self._store is None:

            logger.info(
                "Building HuggingFace embedding model..."
            )

            embeddings = HuggingFaceEmbeddings(
                model_name=_EMBEDDING_MODEL,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

            logger.info(
                "Creating in-memory Chroma vector store..."
            )

            self._store = Chroma(
                collection_name="debate_arguments",
                embedding_function=embeddings,
            )

            logger.info(
                "Chroma vector store ready."
            )

        return self._store

    def store_argument(
        self,
        argument: str,
        turn_number: int,
        quality: str,
        score: float,
        fallacy_name: str = "none",
    ) -> None:
        """Store one user argument in vector memory.

        Args:
            argument:
                The user's argument.

            turn_number:
                Debate turn number.

            quality:
                Argument classification, for example:
                "strong", "weak", or "fallacy".

            score:
                Overall argument score.

            fallacy_name:
                Detected fallacy name, or "none".
        """

        if not argument or not argument.strip():
            logger.warning(
                "VectorMemory: empty argument ignored."
            )
            return

        store = self._get_store()

        document = Document(
            page_content=argument.strip(),
            metadata={
                "turn_number": turn_number,
                "quality": quality,
                "score": float(score),
                "fallacy_name": fallacy_name,
            },
        )

        store.add_documents([document])

        self._doc_count += 1

        logger.info(
            "VectorMemory: argument stored | "
            f"turn={turn_number} | "
            f"quality={quality} | "
            f"score={score} | "
            f"fallacy={fallacy_name} | "
            f"total={self._doc_count}"
        )

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def find_similar(
        self,
        argument: str,
        similarity_threshold: float = 0.20,
    ) -> list[dict[str, Any]]:
        """Find arguments semantically similar to the current argument.

        Args:
            argument:
                Current argument to compare against.

            similarity_threshold:
                Minimum similarity score to return.

        Returns:
            List of dictionaries containing:

                argument
                turn_number
                quality
                score
                fallacy_name
                similarity
        """

        if not argument or not argument.strip():
            return []

        if self._store is None or self._doc_count == 0:
            return []

        try:

            results = self._store.similarity_search_with_score(
                query=argument.strip(),
                k=min(self.top_k, self._doc_count),
            )

            similar_arguments: list[dict[str, Any]] = []

            for document, distance in results:

                # Chroma returns a distance.
                #
                # Lower distance = more similar.
                #
                # Because embeddings are normalized, we can convert
                # the distance into a simple similarity-style value.
                similarity = max(0.0, 1.0 - float(distance))

                if similarity < similarity_threshold:
                    continue

                metadata = document.metadata

                similar_arguments.append(
                    {
                        "argument": document.page_content,
                        "turn_number": metadata.get(
                            "turn_number"
                        ),
                        "quality": metadata.get(
                            "quality"
                        ),
                        "score": metadata.get(
                            "score"
                        ),
                        "fallacy_name": metadata.get(
                            "fallacy_name",
                            "none",
                        ),
                        "similarity": round(
                            similarity,
                            3,
                        ),
                    }
                )

            if similar_arguments:
                logger.info(
                    "VectorMemory: found "
                    f"{len(similar_arguments)} similar arguments."
                )

            return similar_arguments

        except Exception as e:
            logger.warning(
                f"VectorMemory similarity search failed: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Weak argument retrieval
    # ------------------------------------------------------------------

    def find_weak_arguments(self) -> list[dict[str, Any]]:
        """Return previously weak or fallacious arguments.

        This is useful for coaching:

            "You have made this type of weak argument before."

        Returns:
            List of weak/fallacious arguments.
        """

        if self._store is None or self._doc_count == 0:
            return []

        try:

            results = self._store.get(
                where={
                    "quality": {
                        "$in": [
                            "weak",
                            "fallacy",
                        ]
                    }
                }
            )

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            weak_arguments: list[dict[str, Any]] = []

            for index, content in enumerate(documents):

                metadata = (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                )

                weak_arguments.append(
                    {
                        "argument": content,
                        "turn_number": metadata.get(
                            "turn_number"
                        ),
                        "quality": metadata.get(
                            "quality"
                        ),
                        "score": metadata.get(
                            "score"
                        ),
                        "fallacy_name": metadata.get(
                            "fallacy_name",
                            "none",
                        ),
                    }
                )

            return weak_arguments

        except Exception as e:
            logger.warning(
                f"VectorMemory weak argument search failed: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Fallacy retrieval
    # ------------------------------------------------------------------

    def find_previous_fallacies(
        self,
        fallacy_name: str,
    ) -> list[dict[str, Any]]:
        """Find arguments containing the same fallacy.

        Args:
            fallacy_name:
                Example: "ad_hominem"

        Returns:
            Previously detected arguments containing that fallacy.
        """

        if self._store is None or self._doc_count == 0:
            return []

        if not fallacy_name or fallacy_name == "none":
            return []

        try:

            results = self._store.get(
                where={
                    "fallacy_name": fallacy_name,
                }
            )

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            fallacies: list[dict[str, Any]] = []

            for index, content in enumerate(documents):

                metadata = (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                )

                fallacies.append(
                    {
                        "argument": content,
                        "turn_number": metadata.get(
                            "turn_number"
                        ),
                        "quality": metadata.get(
                            "quality"
                        ),
                        "score": metadata.get(
                            "score"
                        ),
                        "fallacy_name": metadata.get(
                            "fallacy_name"
                        ),
                    }
                )

            return fallacies

        except Exception as e:
            logger.warning(
                f"VectorMemory fallacy search failed: {e}"
            )
            return []

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    def get_argument_count(self) -> int:
        """Return number of stored arguments."""

        return self._doc_count

    def is_empty(self) -> bool:
        """Return True if no arguments are stored."""

        return self._doc_count == 0

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all vector memory for the current debate session."""

        if self._store is not None:

            try:
                self._store.delete_collection()

            except Exception as e:
                logger.warning(
                    f"VectorMemory cleanup failed: {e}"
                )

            finally:
                self._store = None

        self._doc_count = 0

        logger.info(
            "VectorMemory cleared."
        )
"""Document ingestion pipeline: load -> chunk -> embed -> index"""
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles document ingestion into Qdrant vector store"""

    def __init__(self, embeddings, qdrant_client, qdrant_url: str, collection_name: str):
        """Initialize ingester with pre-built dependencies"""
        self.embeddings = embeddings
        self.qdrant_client = qdrant_client
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, docs_dir: str) -> List:
        """
        Load all markdown files from directory

        Args:
            docs_dir: Path to documents directory

        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading documents from {docs_dir}")

        loader = DirectoryLoader(
            docs_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        return documents

    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into chunks

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        logger.info("Chunking documents...")

        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        return chunks

    def create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def index_documents(self, chunks: List):
        """
        Embed and index chunks in Qdrant

        Args:
            chunks: List of chunked Document objects
        """
        logger.info(f"Indexing {len(chunks)} chunks in Qdrant...")

        self.create_collection()

        QdrantVectorStore.from_documents(
            chunks,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name,
        )

        logger.info("Indexing complete!")

    def ingest(self, docs_dir: str) -> dict:
        """
        Full ingestion pipeline

        Args:
            docs_dir: Path to documents directory

        Returns:
            Dict with documents_loaded and chunks_created counts
        """
        logger.info("Starting ingestion pipeline...")

        documents = self.load_documents(docs_dir)
        chunks = self.chunk_documents(documents)
        self.index_documents(chunks)

        logger.info("Ingestion pipeline complete!")
        return {"documents_loaded": len(documents), "chunks_created": len(chunks)}


def run_ingestion(embeddings, qdrant_url: str, collection_name: str, docs_dir: str = None) -> dict:
    """
    Run ingestion pipeline

    Args:
        embeddings: Pre-built embeddings model
        qdrant_url: Qdrant server URL
        collection_name: Qdrant collection name
        docs_dir: Path to documents directory

    Returns:
        Dict with documents_loaded and chunks_created counts
    """
    if docs_dir is None:
        project_root = Path(__file__).parent.parent.parent
        docs_dir = str(project_root / "data" / "documents" / "msmarco")

    qdrant_client = QdrantClient(url=qdrant_url)
    ingester = DocumentIngester(embeddings, qdrant_client, qdrant_url, collection_name)
    return ingester.ingest(docs_dir)

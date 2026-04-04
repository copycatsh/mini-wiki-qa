"""Document ingestion pipeline: load → chunk → embed → index"""
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles document ingestion into Qdrant vector store"""

    def __init__(self):
        """Initialize ingester with embeddings and vector store client"""
        # Initialize embeddings model
        logger.info(f"Loading embeddings model: {settings.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)

        # Text splitter
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

        if settings.QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating collection: {settings.QDRANT_COLLECTION}")

            self.qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {settings.QDRANT_COLLECTION} already exists")

    def index_documents(self, chunks: List):
        """
        Embed and index chunks in Qdrant

        Args:
            chunks: List of chunked Document objects
        """
        logger.info(f"Indexing {len(chunks)} chunks in Qdrant...")

        # Create collection
        self.create_collection()

        # Index documents
        QdrantVectorStore.from_documents(
            chunks,
            self.embeddings,
            url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION,
        )

        logger.info("✅ Indexing complete!")

    def ingest(self, docs_dir: str):
        """
        Full ingestion pipeline

        Args:
            docs_dir: Path to documents directory
        """
        logger.info("🚀 Starting ingestion pipeline...")

        # Load
        documents = self.load_documents(docs_dir)

        # Chunk
        chunks = self.chunk_documents(documents)

        # Embed + Index
        self.index_documents(chunks)

        logger.info("✅ Ingestion pipeline complete!")


def run_ingestion(docs_dir: str = None):
    """
    Run ingestion pipeline

    Args:
        docs_dir: Path to documents directory
    """
    if docs_dir is None:
        # Default: project_root/data/documents/squad
        project_root = Path(__file__).parent.parent.parent  # app/rag/ -> app/ -> root
        docs_dir = str(project_root / "data" / "documents" / "squad")

    ingester = DocumentIngester()
    ingester.ingest(docs_dir)


if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
import os
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FILE = "docs/Explorando o Universo das IAs com Hugging Face.pdf"
FULL_PDF_PATH = os.path.join(BASE_DIR, PDF_FILE)
CHROMA_DB_DIR = "db/chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

SEPARATORS = ["\n\n", "\n", ".", " ", ""]


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def load_document(pdf_path: str) -> list:
    """Carrega um documento PDF e retorna seus dados paginados."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Arquivo não localizado: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    return loader.load()


def build_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Cria e retorna um splitter de texto configurado."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )


def load_vector_store(
    embedding: OpenAIEmbeddings,
    persist_directory: str,
) -> Chroma:
    """Carrega um vector store ChromaDB existente."""
    return Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
    )


def search_documents(vector_store: Chroma, question: str, k: int = TOP_K_RESULTS) -> list:
    """Realiza busca por similaridade no vector store."""
    return vector_store.similarity_search(question, k=k)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Carrega e divide o documento
    pages = load_document(FULL_PDF_PATH)
    chunks = build_text_splitter().split_documents(pages)

    # Inicializa embeddings e vector store
    embedding_model = OpenAIEmbeddings()
    vector_store = load_vector_store(embedding_model, CHROMA_DB_DIR)

    # Realiza a busca
    question = "O que é o Hugging Face?"
    results = search_documents(vector_store, question)

    # Exibe os resultados
    for doc in results:
        print(doc.page_content)
        print("=" * 5, doc.metadata, "\n")
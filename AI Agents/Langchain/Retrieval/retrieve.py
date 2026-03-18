import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM Aided Retrieval
from langchain_openai.llms import OpenAI
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_HUGGING_FACE = 'docs/Explorando o Universo das IAs com Hugging Face.pdf'
PDF_OPENAI = 'docs/Explorando a API da OpenAI.pdf'
FULL_HUGGING_FACE_PATH = os.path.join(BASE_DIR, PDF_HUGGING_FACE)
FULL_OPENAI_PATH = os.path.join(BASE_DIR, PDF_OPENAI)
CHROMA_DB_DIR = 'db/chroma_db'

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

SEPARATORS = ["\n\n", "\n", ".", " ", ""]

agent = OpenAI()

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def load_multi_docs(hugging_face_pdf: str, openai_pdf: str) -> list:
    if not os.path.isfile(hugging_face_pdf):
        raise FileNotFoundError(f'Arquivo não localizado: {hugging_face_pdf}')
    
    if not os.path.isfile(openai_pdf):
        raise FileNotFoundError(f'Arquivo não localizado: {openai_pdf}')
    
    paths = [
        hugging_face_pdf,
        openai_pdf    
    ]

    pages = []
    for path in paths:
        loader = PyPDFLoader(path)
        pages.extend(loader.load())

    return pages


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


def alter_metadata(documents: list):
    """Altera o metadados da primeira página
        Exemplo:
            
            Antes:
                {'source': 'arquivos/Explorando o Universo das IAS com HuggingFace.pdf'}
            Depois:
                {'source': 'Explorando o Universo das IAS com HuggingFace.pdf'}
    """

    for i, doc in enumerate(documents):
        doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '')
        doc.metadata['doc_id'] = i

    return documents


def load_vector_store(
    documents: list,
    embedding: OpenAIEmbeddings,
    persist_directory: str,
) -> Chroma:
    """Carrega um vector store ChromaDB existente."""
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
    )


def search_documents(vector_store: Chroma, question: str, k: int = TOP_K_RESULTS) -> list:
    """Realiza busca por similaridade no vector store."""
    return vector_store.similarity_search(question, k=k)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Carrega e divide os documentos
    pages = load_multi_docs(FULL_HUGGING_FACE_PATH, FULL_OPENAI_PATH)
    pages = alter_metadata(pages)
    chunks = build_text_splitter().split_documents(pages)

    """ metadata_info = [
        AttributeInfo(
            name='source',
            description='Nome da apostila de onde o texto original foi retirado. Deve ter o valor de: ' \
            'Explorando o Universo das IAs com Hugging Face.pdf ou Explorando a API da OpenAI.pdf',
            type='string'
        ),
        AttributeInfo(
            name='page',
            description='A página da apostila de onde o texto se origina.',
            type='integer'
        )
    ] """

    # Inicializa embeddings e vector store
    embedding_model = OpenAIEmbeddings()
    vector_store = load_vector_store(chunks, embedding_model, CHROMA_DB_DIR)

    # Realiza a busca (Semantic Search)
    question = "O que a apostila de Hugging Face fala sobre a Open AI e ChatGPT?"
    results = search_documents(vector_store, question)

    # Exibe os resultados
    for doc in results:
        print(doc.page_content)
        print("=" * 5, doc.metadata, "\n")
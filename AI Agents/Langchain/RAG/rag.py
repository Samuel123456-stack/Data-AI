from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.prompts import PromptTemplate

import os
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

SEPARATORS = ["\n\n", "\n", ".", " ", ""]

agent = ChatOpenAI(model='gpt-5-nano')

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def load_multi_docs(hugging_face_file: str, openai_file: str) -> list:
    if not os.path.isfile(hugging_face_file):
        raise FileNotFoundError(f'Arquivo não localizado: {hugging_face_file}')
    
    if not os.path.isfile(openai_file):
        raise FileNotFoundError(f'Arquivo não localizado: {openai_file}')
    
    documents = [
        hugging_face_file,
        openai_file 
    ]

    pages = []
    for doc in documents:
        loader = PyPDFLoader(doc)
        pages.extend(loader.load())

    return pages

def build_text_splitter(
        chunk_size: int=CHUNK_SIZE,
        chunk_overlap: int=CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS
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
        embeddings: OpenAIEmbeddings,
        persist_dir: str
) -> Chroma:
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

def search_documents(vector_store: Chroma) -> RetrievalQA:
     # Personaliza o prompt para o modelo

    prompt = PromptTemplate.from_template(
        """
        Utilize o contexto fornecido para responder a pergunta ao final.
        Se você não souber a resposta, apenas diga que não sabe e não tente inventar a resposta.
        Utilize três frases no máximo e mantenha a resposta concisa.

        Contexto: {context}

        Pergunta: {question}

        Resposta:
        """
    )

    chat_chain = RetrievalQA.from_chain_type(
        llm=agent,
        retriever=vector_store.as_retriever(search_type='mmr'),
        chain_type_kwargs={'prompt': prompt},
        return_source_documents=True
    )

    return chat_chain

if __name__ == '__main__':
    pages = load_multi_docs(FULL_HUGGING_FACE_PATH, FULL_OPENAI_PATH)
    update_metadata = alter_metadata(pages)
    chunks = build_text_splitter().split_documents(update_metadata)

    # Inicializa embeddings e armazenamento vetorial
    embedder = OpenAIEmbeddings()
    vector_store = load_vector_store(chunks, embedder, CHROMA_DB_DIR)


    # Realiza a busca (Semantic Search)
    question = 'O que é Hugging Face e como faço para acessá-lo?'
    chat_chain = search_documents(vector_store=vector_store)
    print(chat_chain.invoke({'query': question})['result'])
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain_classic.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOLDER = 'docs'
PDF_FILE = 'Explorando o Universo das IAs com Hugging Face.pdf'
CSV_FILE = 'Top 1000 IMDB movies.csv'
FULL_PDF_PATH = os.path.join(BASE_DIR, FOLDER, PDF_FILE)
FULL_CSV_PATH = os.path.join(BASE_DIR, FOLDER, CSV_FILE)

# Instancia o modelo da OpenAI
agent = ChatOpenAI(model='gpt-5-nano')

# Acessa o conteúdo do arquivo PDF
def pdf_reader() -> list:
    if not os.path.isfile(FULL_PDF_PATH):
        raise FileNotFoundError(f'Arquivo não encontrado: {FULL_PDF_PATH}')
    
    loader = PyPDFLoader(FULL_PDF_PATH)
    return loader.load()

# Cria a cadeia da interação no formato de pergunta-resposta
def qa_chain():
    return load_qa_chain(
        llm=agent,
        chain_type='stuff',
        verbose=True
    )

if __name__ == '__main__':
    question = 'Quais assuntos são tratados no documento?'

    pdf = pdf_reader()
    chain = qa_chain()

    response = chain.run(input_documents=pdf[:10], question=question)
    print(response)
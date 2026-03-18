from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain_classic.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOLDER = 'docs'
CSV_FILE = 'Top 1000 IMDB movies.csv'
FULL_CSV_PATH = os.path.join(BASE_DIR, FOLDER, CSV_FILE)

# Instancia o modelo da OpenAI
agent = ChatOpenAI(model='gpt-5-nano')

# Acessa o conteúdo do arquivo csv
def csv_reader() -> list:
    if not os.path.isfile(FULL_CSV_PATH):
        raise FileNotFoundError(f'Arquivo não encontrado: {FULL_CSV_PATH}')
    
    loader = CSVLoader(FULL_CSV_PATH)
    return loader.load()

# Cria uma cadeia de pergunta-resposta
def qa_chain():
    return load_qa_chain(
        llm=agent,
        chain_type='stuff',
        verbose=False
    )

# Invoca o "main" para retornar uma resposta com base na pergunta do usuário
if __name__ == '__main__':

    question = 'Qual é o filme com maior metascore?'
    csv = csv_reader()
    chain = qa_chain()

    #response = csv[0].page_content
    response = chain.run(input_documents=csv[:10], question=question)
    print(response)
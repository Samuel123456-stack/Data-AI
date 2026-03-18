from langchain_community.document_loaders.web_base import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

# Realiza a leitura da API da agno
def web_url_reader(url: str):
    link = url
    loader = WebBaseLoader(link)

    return loader.load()

# Invoca o "main" para recuperar uma parte do conteúdo lido
if __name__ == '__main__':
    url = 'https://docs.agno.com/'

    web_reader = web_url_reader(url=url)
    print(web_reader[0].page_content[:1000])
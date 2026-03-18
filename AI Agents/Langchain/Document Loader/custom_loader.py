from langchain_community.document_loaders.base import BaseLoader
from langchain_classic.schema import Document

class MyCustomLoader(BaseLoader):
    def __init__(self, source):
        self.source = source

    def load(self):
        # Lógica para carregar os dados da fonte
        documents = []
        # Exemplo: Carregar dados de um arquivo de texto
        with open(self.source, 'r', encoding='utf-8') as file:
            content = file.read()
            # Criar um documento com o conteúdo e metadados
            documents.append(page_content=content, metadata={'source': self.source})

        return documents
    
if __name__ == '__main__':
    # Cria uma instância do loader
    path = ''
    loader = MyCustomLoader(path)
    docs = loader.load()

    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)
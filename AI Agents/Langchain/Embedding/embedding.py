from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Representação vetorial dos textos
def embedded_text(split_text: list):
    embedding_model = OpenAIEmbeddings()

    embedded = embedding_model.embed_documents(split_text)

    return embedded

# Retorna o coeficiente de similaridade (aproximação) entre as palavras
def similarity_coefficient(embed: list):
    embedding = embedded_text(embed)
    for i in range(len(embedding)):
        for j in range(len(embedding)):
            print(round(np.dot(embedding[i], embedding[j]), 3), end=' | ')


if __name__ == '__main__':
    text = [
        'Eu gosto de cachorros',
        'Eu gosto de animais',
        'O tempo está ruim lá fora'
    ]

    embed = embedded_text(text)
    print(embed[0][:10])
    print()

    similarity_coefficient(text)
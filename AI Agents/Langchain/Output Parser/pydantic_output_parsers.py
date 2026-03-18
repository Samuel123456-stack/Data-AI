from langchain_openai import ChatOpenAI
#from langchain_core.prompts import PromptTemplate
#from langchain_core.messages import SystemMessage, HumanMessage
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_community.document_loaders import PyPDFLoader
#from langchain_core.chat_history import InMemoryChatMessageHistory

from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

from typing import Optional
from pydantic import BaseModel, Field

agent = ChatOpenAI()

review_cliente = """Este soprador de folhas é bastante incrível. Ele tem 
quatro configurações: sopro de vela, brisa suave, cidade ventosa 
e tornado. Chegou em dois dias, bem a tempo para o presente de 
aniversário da minha esposa. Acho que minha esposa gostou tanto 
que ficou sem palavras. Até agora, fui o único a usá-lo, e tenho 
usado em todas as manhãs alternadas para limpar as folhas do 
nosso gramado. É um pouco mais caro do que os outros sopradores 
de folhas disponíveis no mercado, mas acho que vale a pena pelas 
características extras."""

class Joke(BaseModel):
    """Piada para contar ao usuário"""
    intro: str = Field(description='A introdução da piada')
    punchline: str = Field(description='A conclusão da piada')
    evaluation: Optional[int] = Field(description='O quão é engraçada a piada de 1 a 10')


# Exemplo mais prático

class EvaluationReview(BaseModel):
    gift: bool = Field(description='Verdadeiro se foi para presente e False se não foi')
    delivery_days: int = Field(description='Quantos dias para entrega do produto')
    value_perception: list[str] = Field(description='Extraia qualquer frase sobre o valor ou \
                             ou preço do produto. Retorne uma lista')


if __name__ == '__main__':
    structured_llm_joke = agent.with_structured_output(Joke)
    response_joke = structured_llm_joke.invoke('Conte uma piada sobre gatinhos')
    print(response_joke)
    print()

    structured_llm_reviwe = agent.with_structured_output(EvaluationReview)
    response_review = structured_llm_reviwe.invoke(review_cliente)
    print(response_review)
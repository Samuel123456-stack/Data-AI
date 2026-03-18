from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

agent = ChatOpenAI(model='gpt-4o-mini')


# --- Agentes especializados ---

def math_agent(question: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        Você é um professor de matemática de ensino fundamental
        capaz de dar respostas muito detalhadas e didáticas. Responda a seguinte pergunta de um aluno:
        Pergunta: {question}
        """
    )
    chain = prompt | agent
    return chain.invoke({'question': question}).content


def physics_agent(question: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        Você é um professor de física de ensino fundamental capaz de dar respostas muito detalhadas e
        didáticas. Responda a seguinte pergunta de um aluno.
        Pergunta: {question}
        """
    )
    chain = prompt | agent
    return chain.invoke({'question': question}).content


def history_agent(question: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        Você é um professor de história de ensino fundamental capaz de dar respostas muito detalhadas e
        didáticas. Responda a seguinte pergunta do aluno.
        Pergunta: {question}
        """
    )
    chain = prompt | agent
    return chain.invoke({'question': question}).content


# --- Categorizador ---

class Categorizer(BaseModel):
    """Categoriza as perguntas de alunos do ensino fundamental"""
    knowledge_field: str = Field(
        description='A área de conhecimento da pergunta feita pelo aluno. '
                    'Deve ser "física", "matemática" ou "história". '
                    'Caso não se encaixe em nenhuma delas, retorne "outra".'
    )


categorizer_prompt = ChatPromptTemplate.from_template(
    'Você deve categorizar a seguinte pergunta: {question}.' \
    'Não faça complementações.' \
    'Não explique o que foi perguntado.'
)

categorizer_chain = categorizer_prompt | agent.with_structured_output(Categorizer)


# --- Roteador ---

def route_question(question: str) -> str:
    category = categorizer_chain.invoke({'question': question})
    field = category.knowledge_field.lower()

    print(f'Categoria identificada: {field}\n')

    if field == 'matemática':
        return math_agent(question)
    elif field == 'física':
        return physics_agent(question)
    elif field == 'história':
        return history_agent(question)
    else:
        return 'Desculpe, só consigo responder perguntas de matemática, física ou história.'


# --- Execução ---

if __name__ == '__main__':
    question = 'Quando foi a proclamação da república?'
    response = route_question(question)
    print(response)
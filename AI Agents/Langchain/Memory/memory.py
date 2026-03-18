from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

# Armazena as sessões do histórico de conversa
store = {}
def get_session(session_id: str) -> InMemoryChatMessageHistory:

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Integra instruções claras para o prompt do agente
def prompt(question: str, session_id: str = "default") -> str:

    template = ChatPromptTemplate.from_messages([
        ('system', 'Você é um tutor de programação chamado Asimo. Responda as perguntas de forma didática.'),
        ('placeholder', '{history}'),
        ('human', '{question}')
    ])

    chain = template | ChatOpenAI(model="gpt-5-nano")

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session,
        input_messages_key='question',
        history_messages_key='history'
    )

    config = {'configurable': {'session_id': session_id}}

    response = chain_with_memory.invoke({'question': question}, config=config)
    return response.content

if __name__ == '__main__':
    session_id = 'samuel'

    print('Asimo: Olá! Sou o Asimo, seu tutor de programação. Como posso ajudar?á')
    print("(Digite 'sair' para encerrar)\n")

    while True:
        question = input('Digite: ').strip()
        
        if not question:
            continue 

        if question.lower() == 'sair':
            print('Goodbye!')
            break

        response = prompt(question=question, session_id=session_id)
        print(f'Asimo: {response}')
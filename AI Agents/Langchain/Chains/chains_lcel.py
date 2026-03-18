from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

agent = ChatOpenAI(model='gpt-5-nano')

# Cria uma função para designar um template formatado ao agente
def simple_prompt(subject: str) -> str:
    template = PromptTemplate.from_template(
        """
        Crie uma frase sobre o seguinte assunto: 
        {subject}
        """
    )

    chain = template | agent

    return chain.invoke({'subject': subject})

# Cria uma função com a mesma finalidade acima, porém formata o conteúdo gerado na resposta
def add_chain_elements(subject: str) -> str:
    template = PromptTemplate.from_template(
        """
        Crie uma frase com o seguinte assunto:
        {subject}
        """
    )

    chain = template | agent | StrOutputParser()
    return chain.invoke({'subject': subject})

# Invoca o "main"
if __name__ == '__main__':
    #response = simple_prompt('futebol')
    #print(response)
    std_reply = add_chain_elements('futebol')
    print(std_reply)
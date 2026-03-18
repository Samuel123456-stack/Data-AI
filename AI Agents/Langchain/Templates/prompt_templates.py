from langchain_openai.llms import OpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

agent = OpenAI()

def answering(question):
    prompt_template = PromptTemplate.from_template(
        """Responda da seguinte pergunta do usuário:
        {question}"""
    )
    
    prompt = prompt_template.format(question=question)
    
    response = agent.invoke(prompt)
    return response

# Unindo múltiplos prompts

def multi_prompts(n_words, lang, question) -> str:
    final_template = PromptTemplate.from_template(
        """
        Responda a pergunta em até {n_words} palavras.
        Responda a pergunta no idioma {lang}.
        Responda a pergunta seguinte seguindo as instruções: {question}
        """
    )
    
    # Formata o template com os valores
    prompt = final_template.format(n_words=n_words, lang=lang, question=question)
    response = agent.invoke(prompt)
    
    return response

def chat_prompt_template(question: str) -> str:
    chat_template = ChatPromptTemplate.from_template(f'Essa é a minha dúvida: {question}')
    chat_template.format_messages(question=question)
    
    return chat_template

def chat_models(asset_name: str, question: str) -> str:
    chat = ChatOpenAI()

    chat_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'Você é um assistente engraçado e se chama {asset_name}'),
            ('human', 'Olá, como vai?'),
            ('ai', 'Melhor agora! como posso ajudá-lo?'),
            ('human', '{question}')
        ]
    )

    fmt_msg = chat_template.format_messages(asset_name=asset_name, question=question)
    reply = chat.invoke(fmt_msg)

    return reply.content

if __name__ == '__main__':
    response = answering(question='O que é um buraco negro?')
    print(response)
    mlt_pmpt = multi_prompts(n_words=10, lang='inglês', question='O que é uma estrela?')
    print(mlt_pmpt)
    chat_model_resp = chat_models(asset_name='Asimov', question='Qual o seu nome?')
    print(chat_model_resp)
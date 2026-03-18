from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

import warnings
warnings.filterwarnings('ignore')

def chat_format_template(asset_name: str, question: str) -> str:
    chat_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'Você é um assistente engraçado e seu nome é {asset_name}'),
            ('human', '{question}')
        ]
    )

    prompt = chat_template.invoke({'asset_name': asset_name, 'question': question})
    return prompt

def return_response(asset_name: str, question: str) -> str:
    prompt = chat_format_template(asset_name, question)

    chat = ChatOpenAI()
    return chat.invoke(prompt)

def output_parser(asset_name: str, question: str) -> str:
    final_response = return_response(asset_name, question)

    str_parser = StrOutputParser()
    return str_parser.invoke(final_response)

if __name__ == '__main__':
    reply = output_parser(asset_name='Asimov', question='Qual o seu nome?')
    print(reply)
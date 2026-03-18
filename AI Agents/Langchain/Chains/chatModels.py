from langchain_openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage
""" from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.db.sqlite import SqliteDb
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.chunking.semantic import SemanticChunking """

from dotenv import load_dotenv
load_dotenv()

import os
os.getenv('OPENAI_API_KEY')

# Instancia o modelo da OpenAI
chat = OpenAI(model='gpt-3.5-turbo-instruct')

# Simula a interação do usuário e o agente de maneira simplificada
msg = [
    SystemMessage(content='Você é um assistente que conta piadas'),
    HumanMessage(content='Quanto é 1 + 1?')
]

print(chat.invoke(msg))
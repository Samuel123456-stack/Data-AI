from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.websearch import WebSearchTools
import warnings

from .models.movies import MovieRecommendation
from .tools.tmdb import enrich_recommendation_posters, search_movie
from .config import Config
from .prompts import *

Config.validate()

warnings.filterwarnings('ignore')

class FilmPro:
    """Agente de recomendação de filmes com AI"""
    
    DEFAULT_MODEL = 'gpt-4o'
    
    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model
        self._agent = self._build_agent()

    def _build_agent(self) -> Agent:
        return Agent(
            name='FilmPro',
            model=OpenAIChat(id=self.model, api_key=self.api_key),
            description=description,
            instructions=instructions,
            tools=[WebSearchTools(backend='google'), search_movie],
            markdown=True,
            add_datetime_to_context=True,
            output_schema=MovieRecommendation,
            debug_mode=True,
            debug_level=1
        )
    
    async def recommend(self, query: str, stream: bool = False) -> None:
        """Envia uma consulta ao agente"""
        self.outcome = await self._agent.arun(input=query, stream=stream)

        if self.outcome and self.outcome.content:
            data: MovieRecommendation = self.outcome.content
            data = await enrich_recommendation_posters(data)
            self.outcome.content = data
            pretty_json_output = data.model_dump_json(indent=2)
            print(pretty_json_output)

        return self.outcome
    
# Instância única reutilizada em toda a aplicação
_agent = FilmPro()
 
 
async def recommend(query: str) -> MovieRecommendation | None:
    """Função de entrada para o router — chama o agente e retorna o conteúdo tipado"""
    result = await _agent.recommend(query)
    if result and result.content:
        return result.content
    return None

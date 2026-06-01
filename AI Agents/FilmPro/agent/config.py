"""Espaço para leitura e validação das variáveis de ambiente"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Gerenciador de configurações centralizado"""

    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    OMDB_API_KEY: str = os.getenv('OMDB_API_KEY')
    TMDB_API_KEY: str = os.getenv('TMDB_API_KEY')

    _REQUIRED_KEYS = ('OPENAI_API_KEY', 'TMDB_API_KEY')

    def __new__(cls):
        raise TypeError('Config could not be instanced directly.')

    @classmethod
    def validate(cls) -> None:
        """Valida se todas as variáveis de ambiente obrigatórias estão definidas"""

        missing = [key for key in cls._REQUIRED_KEYS
                   if not getattr(cls, key)]
        
        if missing:
            keys = ', '.join(missing)
            raise ValueError(f'Environment variable(s) missing in the .env file: {keys}')

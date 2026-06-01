# Permite a requisição em diferentes domínios de forma segura
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

# Importa a inicialização da instância com a classe "APIRouter"
from .routers import router

# Inicializa a instância da aplicação FastAPI
app = FastAPI(
    title='FilmPro API',
    description='API de recomendação de filmes',
    version='1.0.0',
    docs_url='/docs', # Local dos endpoints
    redoc_url='/redoc_url', # Local dos documentos em outro formato
    openapi_url='/openapi.json'
)

# Configura CORS para acessos externos
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get(
    '/',
    tags=['Sistema'],
    summary='Informações da API',
    description='Retorna informações sobre a API FilmPro'
)
async def root():
    """Endpoint com informação raiz da API"""
    return {
        'name': 'FilmPro API',
        'description': 'Sistema de recomendação inteligente de filmes',
        'version': '1.0.0',
        'endpoints': {
            'recommendations': '/recommendations',
            'docs': '/docs',
            'redoc': '/redoc'
        }
    }

app.include_router(router)
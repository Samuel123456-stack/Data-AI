from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.models.movies import MovieRecommendation
from agent.core import recommend

router = APIRouter()

class RecommendationRequest(BaseModel):
    preferences: str = Field(
        description='Descrição das preferências do usuário para filme',
        min_length=50,
        max_length=100
    )

class RecommendationResponse(BaseModel):
    success: bool = Field(..., description='Índice se a recomendação foi bem sucedida')
    data: MovieRecommendation = Field(..., description='Dados de recomendação')
    message: str = Field(default='Recomendação gerada com sucesso', description='Mensagem informativa')

# Iniciando a rota
@router.post(
    '/recommendations',
    response_model=RecommendationResponse,
    summary='Obter recomendações de filmes',
    description='Gera recomendações personalizadas de filmes baseados na preferência do usuário',
    tags=['Recomendações']
)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    try:
        # O agente executa a requisição para pegar as preferências do usuário
        req = await recommend(request.preferences)
        if not req:
            raise HTTPException(
                status_code=500,
                detail='Error processing request. Try again!'
            )
        
        return RecommendationResponse(
            success=True,
            data=req,
            message='Recomendações geradas com sucesso.'
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Error processing recommendations: {str(e)}'
        )
    
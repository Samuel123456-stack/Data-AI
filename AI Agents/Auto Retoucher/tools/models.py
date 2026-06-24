from typing import Literal
from pydantic import BaseModel, Field

class ReportItem(BaseModel):
    description: str = Field(description='A descrição do problema')
    relevance: Literal['ESSENCIAL', 'RECOMENDADO', 'OPCIONAL']
    photoshop_technique: str = Field(description='A técnica Photoshop sugerida para resolver o problema')
    query: str = Field(description='A query para a ferramenta Fal AI para encontrar a localização do problema')
    x_point: float = Field(description='Coordenada X normalizada entre 0.0 e 1.0 (0.0 = borda esquerda, 1.0 = borda direita). Use SEMPRE o campo "x" (não "x_px") retornado pela ferramenta locate_in_image.')
    y_point: float = Field(description='Coordenada Y normalizada entre 0.0 e 1.0 (0.0 = topo da imagem, 1.0 = base da imagem). Use SEMPRE o campo "y" (não "y_px") retornado pela ferramenta locate_in_image.')

class SkinAnalysisSchema(BaseModel):
    report: list[ReportItem] = Field(description='O relatório de análise da pele')
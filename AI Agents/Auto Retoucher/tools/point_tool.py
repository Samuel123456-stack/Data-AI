import json
from pathlib import Path
from agno.tools import Toolkit
from agno.tools.function import ToolResult
from .point_detection import annotated_image_path, detect_points


class PointDetection(Toolkit):
    """Localiza coordenadas precisas de elementos na imagem para retoque fotográfico."""

    def __init__(self, **kwargs):
        super().__init__(
            name="point_detection",
            tools=[self.locate_in_image],
            instructions=(
                "Use `locate_in_image` sempre que precisar marcar com precisão um defeito ou "
                "região da pele na foto. Passe o IMG_PATH do prompt e descreva o elemento "
                "de forma objetiva (ex.: 'espinha na bochecha esquerda', 'olheira olho direito')."
            ),
            add_instructions=True,
            **kwargs,
        )

    def locate_in_image(self, image_path: str, query: str) -> ToolResult:
        """Localiza coordenadas (x, y) de um elemento na imagem.

        Use esta ferramenta sempre que o laudo de retoque exigir marcações precisas —
        manchas, espinhas, olheiras, rugas, brilhos, capilares ou qualquer detalhe
        pontual que precise de coordenadas exatas para Healing Brush, máscaras ou
        seleções localizadas no Photoshop.

        Args:
            image_path: Caminho absoluto da foto analisada (valor de IMG_PATH).
            query: Descrição curta e específica do elemento a localizar.

        Returns:
            Pontos em coordenadas normalizadas (x, y: 0–1) e em pixels (x_px, y_px).
        """
        path = Path(image_path)
        if not path.is_file():
            return ToolResult(content=f"Erro: imagem não encontrada em `{image_path}`.")

        try:
            output_path = annotated_image_path(path.resolve())
            points = detect_points(
                str(path.resolve()),
                query,
                draw=True,
                output_path=str(output_path),
            )
        except Exception as exc:
            return ToolResult(content=f"Erro ao detectar pontos: {exc}")

        payload = {
            "image_path": str(path.resolve()),
            "annotated_image_path": str(output_path) if points else None,
            "query": query,
            "count": len(points),
            "points": points,
        }
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=2)
        )

if __name__ == '__main__':
    fal = PointDetection()
    image_url = Path(__file__).parent.parent / 'images' / 'img2.jpeg'
    locator = fal.locate_in_image(image_url, 'left eye')

    print(locator)

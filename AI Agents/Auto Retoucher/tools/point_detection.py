from pathlib import Path

import fal_client
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

MODEL = "fal-ai/moondream3-preview/point"


def annotated_image_path(image_path: str | Path) -> Path:
    """Retorna o caminho padrao para a imagem com os pontos marcados."""
    path = Path(image_path)
    return path.with_name(f"{path.stem}_points{path.suffix}")


def save_annotated_image(
    image_path: str | Path,
    points: list[dict],
    output_path: str | Path | None = None,
) -> Path | None:
    """Salva uma copia da imagem exibindo os pontos detectados."""
    if not points:
        return None

    path = Path(image_path)
    destination = Path(output_path) if output_path else annotated_image_path(path)

    with Image.open(path) as img:
        annotated = img.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for point in points:
            x, y = point["x_px"], point["y_px"]
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red', width=2)
        #annotated.save(destination)

    return destination


def detect_points(
    image_path: str,
    query: str,
    *,
    draw: bool = False,
    output_path: str | None = None,
) -> list[dict]:
    """Detecta pontos de um elemento na imagem via Moondream3 (Fal AI)."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    url = fal_client.upload_file(str(path))
    result = fal_client.subscribe(
        MODEL,
        arguments={"image_url": url, "prompt": query},
    )

    with Image.open(path) as img:
        width, height = img.size
        points = [
            {
                "x": point["x"],
                "y": point["y"],
                "x_px": round(point["x"] * width),
                "y_px": round(point["y"] * height),
            }
            for point in result["points"]
        ]

    if draw:
        save_annotated_image(path, points, output_path)

    return points

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent_api import AutoRetoucher
from tools.models import ReportItem, SkinAnalysisSchema

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Auto Retoucher")
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
FRONTEND_DIR = ROOT / "frontend"
retoucher: AutoRetoucher | None = None


class AnalyzeResponse(BaseModel):
    image_url: str
    report: list[ReportItem]


def get_retoucher() -> AutoRetoucher:
    global retoucher
    if retoucher is None:
        retoucher = AutoRetoucher()
    return retoucher


def parse_response(content) -> SkinAnalysisSchema:
    if isinstance(content, SkinAnalysisSchema):
        return content

    if isinstance(content, dict):
        # Detect upstream error dicts (e.g. Gemini / Fal AI 4xx responses)
        # before attempting schema validation, to surface a clear message.
        if "error" in content and "report" not in content:
            err = content["error"]
            if isinstance(err, dict):
                detail = (
                    err.get("message")
                    or err.get("status")
                    or f"código {err.get('code', 'desconhecido')}"
                )
            else:
                detail = str(err)
            raise ValueError(f"A API de IA retornou um erro: {detail}")
        return SkinAnalysisSchema.model_validate(content)

    # Try to parse as JSON string / repr
    try:
        return SkinAnalysisSchema.model_validate_json(str(content))
    except Exception:
        raise ValueError(
            f"Resposta inesperada do modelo: {str(content)[:200]}"
        )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem.")

    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image").name
    image_path = UPLOAD_DIR / filename
    image_path.write_bytes(await file.read())

    try:
        response = get_retoucher().analyze_image(str(image_path))
        result = parse_response(response.content)
        return AnalyzeResponse(image_url=f"/uploads/{filename}", report=result.report)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno durante a análise: {exc}",
        ) from exc


# Static file mounts — defined after routes so named routes take priority
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini 
from dotenv import load_dotenv

from tools.point_tool import PointDetection
from tools.models import SkinAnalysisSchema

load_dotenv()
warnings.filterwarnings("ignore")


class AutoRetoucher:
    """Agente especialista em análise de pele e sugestões de retoque fotográfico."""

    GEMINI_MODEL = "gemini-3.1-pro-preview"
    PROMPT_PATH = ROOT / "prompts" / "skin.md"

    def __init__(self, api_key: str | None = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY does not exist")

        self.model = model
        self._agent = self._build_agent()

    def _build_agent(self) -> Agent:
        return Agent(
            name="AutoRetoucher",
            model=Gemini(id=self.model, api_key=self.api_key),
            tools=[PointDetection()],
            markdown=True
        )

    def _load_prompt_template(self) -> str:
        if not self.PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.PROMPT_PATH}")

        prompt = self.PROMPT_PATH.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError("Prompt file is empty")

        return prompt

    def analyze_image(self, image_path: str):
        """Analisa a imagem e retorna o laudo de retoque."""
        path = str(Path(image_path).resolve())
        return self._agent.run(
            input=self._load_prompt_template().format(img_path=path),
            images=[Image(filepath=path)],
            output_schema=SkinAnalysisSchema
        )

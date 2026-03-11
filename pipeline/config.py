"""Configuration for the co-optimization pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DATA_DIR = PROJECT_ROOT / "data" / "pipeline"
PROMPTS_DIR = PROJECT_ROOT / "data" / "pipeline" / "prompts"
_INIT_PROMPTS_DIR = Path(__file__).parent / "prompts"
CONF_DIR = Path(__file__).parent / "conf"
CHARTER_PATH = PROJECT_ROOT / "resources" / "SwissAICharter.md"
ANNOTATION_DATA_DIR = PROJECT_ROOT / "data" / "annotation"


@dataclass
class ScoringConfig:
    scale_min: int = 1
    scale_max: int = 5
    accept_threshold: int = 4
    dimensions: list[str] = field(
        default_factory=lambda: ["relevance", "specificity", "charter_grounding", "voice_tone"]
    )


@dataclass
class IterationConfig:
    n_items: int = 50
    n_gold: int = 12
    max_concurrent: int = 10


@dataclass
class PromptsConfig:
    generator: str = "generator_v1.md"
    judge: str = "judge_v1.md"
    improver: str = "improver.md"


@dataclass
class LoopConfig:
    n_iterations: int = 5
    improver_timeout_s: int = 600


@dataclass
class PipelineConfig:
    model: str = "jminder/data-annotator-glm45"
    endpoint: str = "https://api.swissai.cscs.ch/v1"
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)


def load_config(overrides: list[str] | None = None) -> PipelineConfig:
    """Load pipeline config from YAML with optional CLI overrides.

    Uses OmegaConf for structured config merging. Hydra is available for
    CLI entry points but this function works standalone too.
    """
    yaml_path = CONF_DIR / "config.yaml"
    base = OmegaConf.load(yaml_path)
    schema = OmegaConf.structured(PipelineConfig)
    merged = OmegaConf.merge(schema, base)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli)
    cfg: PipelineConfig = OmegaConf.to_object(merged)  # type: ignore
    return cfg


def model_slug(model: str) -> str:
    """Convert a model name to a filesystem-safe directory slug.

    E.g. 'jminder/data-annotator-glm45' -> 'jminder_data-annotator-glm45'
    """
    return model.replace("/", "_")


def _init_model_prompts(model: str) -> None:
    """Initialize a model's prompt directory from the init templates.

    Copies pipeline/prompts/init_generator.md → data/.../generator_v1.md and
    pipeline/prompts/init_judge.md → data/.../judge_v1.md.
    Only runs once per model (skips if dir already exists).
    """
    import shutil

    model_dir = PROMPTS_DIR / model_slug(model)
    if model_dir.exists():
        return
    model_dir.mkdir(parents=True)
    for init_name, v1_name in [("init_generator.md", "generator_v1.md"), ("init_judge.md", "judge_v1.md")]:
        src = _INIT_PROMPTS_DIR / init_name
        assert src.exists(), f"Init template not found: {src}"
        shutil.copy2(src, model_dir / v1_name)


def resolve_prompt_path(filename: str, model: str) -> Path:
    """Resolve a prompt filename within the model-specific directory.

    Prompts live at data/pipeline/prompts/{model_slug}/{filename}. If the
    model directory doesn't exist yet, initializes it from init templates.
    """
    model_dir = PROMPTS_DIR / model_slug(model)
    if not model_dir.exists():
        _init_model_prompts(model)
    path = model_dir / filename
    assert path.exists(), f"Prompt file not found: {path}"
    return path

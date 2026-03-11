"""Configuration for the co-optimization pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DATA_DIR = PROJECT_ROOT / "data" / "pipeline"
PROMPTS_DIR = Path(__file__).parent / "prompts"
CONF_DIR = Path(__file__).parent / "conf"
CHARTER_PATH = PROJECT_ROOT / "resources" / "SwissAICharter.md"
ANNOTATION_DATA_DIR = PROJECT_ROOT / "data" / "annotation"


@dataclass
class ScoringConfig:
    scale_min: int = 1
    scale_max: int = 5
    accept_threshold: int = 4
    dimensions: list[str] = field(
        default_factory=lambda: ["relevance", "depth", "charter_grounding", "clarity"]
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
class PipelineConfig:
    model: str = "jminder/data-annotator-glm45"
    endpoint: str = "https://api.swissai.cscs.ch/v1"
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)


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


def resolve_prompt_path(filename: str) -> Path:
    """Resolve a prompt filename to its absolute path."""
    path = PROMPTS_DIR / filename
    assert path.exists(), f"Prompt file not found: {path}"
    return path

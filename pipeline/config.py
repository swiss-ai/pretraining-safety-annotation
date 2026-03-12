"""Unified configuration for the annotation + pipeline system."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "configs" / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_DATA_DIR = DATA_DIR / "pipeline"
ANNOTATION_DATA_DIR = DATA_DIR / "annotation"
PROMPTS_DIR = PIPELINE_DATA_DIR / "prompts"
_INIT_PROMPTS_DIR = Path(__file__).parent / "prompts"

CHARTER_PATH = PROJECT_ROOT / "resources" / "SwissAICharter.md"


def load_charter_element_ids() -> list[str]:
    """Extract all [X.Y] element IDs from the charter, in order."""
    charter = CHARTER_PATH.read_text(encoding="utf-8")
    return list(dict.fromkeys(re.findall(r"\[(\d+\.\d+)\]", charter)))


CHARTER_ELEMENT_IDS: list[str] = load_charter_element_ids()


# --- Dataclasses ---

@dataclass
class ModelConfig:
    alias: str = ""
    api_name: str = ""
    hf_slug: str = ""


@dataclass
class RoleConfig:
    model: str = ""
    prompt: str = ""


@dataclass
class ImproverConfig:
    judge_prompt: str = "improver_judge.md"
    generator_prompt: str = "improver_generator.md"
    max_batches_per_phase: int = 5
    timeout_s: int = 900


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
class LoopConfig:
    pass


@dataclass
class Phase1Config:
    dataset: str = "locuslab/fineweb_annotated"
    subsets: list[str] = field(default_factory=lambda: [f"score_{i}" for i in range(6)])
    sample_size: int = 200


@dataclass
class Phase2Config:
    endpoint: str = ""
    models: list[ModelConfig] = field(default_factory=list)
    generator: RoleConfig = field(default_factory=RoleConfig)
    judge: RoleConfig = field(default_factory=RoleConfig)
    improver: ImproverConfig = field(default_factory=ImproverConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)


@dataclass
class DashboardConfig:
    port: int = 8600


@dataclass
class AppConfig:
    charter_path: str = "resources/SwissAICharter.md"
    data_dir: str = "data"
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# --- Helper functions ---

def resolve_model(cfg: AppConfig, alias: str) -> ModelConfig:
    """Find a model config by alias (linear scan)."""
    for m in cfg.phase2.models:
        if m.alias == alias:
            return m
    raise ValueError(f"No model with alias '{alias}' in config. Available: {[m.alias for m in cfg.phase2.models]}")


def generator_api_name(cfg: AppConfig) -> str:
    """Shortcut: resolve the generator model's API name."""
    return resolve_model(cfg, cfg.phase2.generator.model).api_name


def judge_api_name(cfg: AppConfig) -> str:
    """Shortcut: resolve the judge model's API name."""
    return resolve_model(cfg, cfg.phase2.judge.model).api_name


def load_config(overrides: list[str] | None = None) -> AppConfig:
    """Load unified config from YAML with optional CLI overrides.

    Uses OmegaConf for structured config merging.
    """
    base = OmegaConf.load(CONFIG_YAML_PATH)
    schema = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(schema, base)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli)
    cfg: AppConfig = OmegaConf.to_object(merged)  # type: ignore
    return cfg


def _init_model_prompts(alias: str) -> None:
    """Initialize a model's prompt directory from the init templates.

    Copies pipeline/prompts/init_generator.md -> data/.../generator_v1.md and
    pipeline/prompts/init_judge.md -> data/.../judge_v1.md.
    Only runs once per model (skips if dir already exists).
    """
    import shutil

    model_dir = PROMPTS_DIR / alias
    if model_dir.exists():
        return
    model_dir.mkdir(parents=True)
    for init_name, v1_name in [("init_generator.md", "generator_v1.md"), ("init_judge.md", "judge_v1.md")]:
        src = _INIT_PROMPTS_DIR / init_name
        assert src.exists(), f"Init template not found: {src}"
        shutil.copy2(src, model_dir / v1_name)


def resolve_prompt_path(filename: str, alias: str) -> Path:
    """Resolve a prompt filename within the model-specific directory.

    Prompts live at data/pipeline/prompts/{alias}/{filename}. If the
    model directory doesn't exist yet, initializes it from init templates.
    """
    model_dir = PROMPTS_DIR / alias
    if not model_dir.exists():
        _init_model_prompts(alias)
    path = model_dir / filename
    assert path.exists(), f"Prompt file not found: {path}"
    return path

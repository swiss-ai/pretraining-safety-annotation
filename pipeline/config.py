"""Unified configuration for the annotation + pipeline system."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "configs" / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_DATA_DIR = DATA_DIR / "pipeline"
ANNOTATION_DATA_DIR = DATA_DIR / "annotation"
PROMPTS_DIR = PIPELINE_DATA_DIR / "prompts"
_INIT_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _resolve_charter_path() -> Path:
    """Read charter_path from config YAML (falls back to dataclass default)."""
    raw = OmegaConf.load(CONFIG_YAML_PATH)
    return PROJECT_ROOT / raw["charter_path"]


CHARTER_PATH = _resolve_charter_path()


def load_charter_element_ids() -> list[str]:
    """Extract all element IDs (X.Y) from the charter, in order.

    Supports both [X.Y] inline references (SwissAI Charter style)
    and ### X.Y headings (ModelRaising Constitution style).
    """
    charter = CHARTER_PATH.read_text(encoding="utf-8")
    inline = re.findall(r"\[(\d+\.\d+)\]", charter)
    headings = re.findall(r"^###\s+(\d+\.\d+)\b", charter, re.MULTILINE)
    return list(dict.fromkeys(inline or headings))


CHARTER_ELEMENT_IDS: list[str] = load_charter_element_ids()
_CHARTER_ID_SET = set(CHARTER_ELEMENT_IDS)


def extract_charter_elements(text: str) -> list[str]:
    """Extract charter element IDs ([X.Y] patterns) from text, preserving order.

    Only returns IDs that exist in the charter.
    """
    matches = re.findall(r"\[(\d+\.\d+)\]", text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m in _CHARTER_ID_SET and m not in seen:
            seen.add(m)
            result.append(m)
    return result


# --- Dataclasses ---


@dataclass
class ModelConfig:
    alias: str = ""
    api_name: str = ""
    hf_slug: str = ""


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
        default_factory=lambda: [
            "relevance",
            "specificity",
            "charter_grounding",
            "voice_tone",
        ]
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
    judge_models: list[ModelConfig] = field(default_factory=list)
    generator_models: list[ModelConfig] = field(default_factory=list)
    improver: ImproverConfig = field(default_factory=ImproverConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)


@dataclass
class DashboardConfig:
    port: int = 8600


@dataclass
class AppConfig:
    charter_path: str = MISSING
    data_dir: str = "data"
    max_tokens: int = 3840
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# --- Helper functions ---


def resolve_model(cfg: AppConfig, alias: str) -> ModelConfig:
    """Find a model config by alias, searching both judge and generator model lists."""
    for m in cfg.phase2.judge_models + cfg.phase2.generator_models:
        if m.alias == alias:
            return m
    all_aliases = [
        m.alias for m in cfg.phase2.judge_models + cfg.phase2.generator_models
    ]
    raise ValueError(
        f"No model with alias '{alias}' in config. Available: {all_aliases}"
    )


def resolve_judge_model(cfg: AppConfig, alias: str) -> ModelConfig:
    """Find a judge model config by alias."""
    for m in cfg.phase2.judge_models:
        if m.alias == alias:
            return m
    raise ValueError(
        f"No judge model with alias '{alias}'. Available: {[m.alias for m in cfg.phase2.judge_models]}"
    )


def resolve_generator_model(cfg: AppConfig, alias: str) -> ModelConfig:
    """Find a generator model config by alias."""
    for m in cfg.phase2.generator_models:
        if m.alias == alias:
            return m
    raise ValueError(
        f"No generator model with alias '{alias}'. Available: {[m.alias for m in cfg.phase2.generator_models]}"
    )


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
    for init_name, v1_name in [
        ("init_generator.md", "generator_v1.md"),
        ("init_judge.md", "judge_v1.md"),
    ]:
        src = _INIT_PROMPTS_DIR / init_name
        assert src.exists(), f"Init template not found: {src}"
        shutil.copy2(src, model_dir / v1_name)


def _resolve_latest_version(model_dir: Path, filename: str) -> Path:
    """Resolve a 'latest' prompt filename to the highest versioned file.

    E.g. 'judge_latest.md' finds the highest 'judge_vN.md' in model_dir.
    """
    import re

    stem = filename.replace("_latest.md", "")
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+)\.md$")
    candidates = []
    for p in model_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    assert candidates, f"No versioned files matching '{stem}_vN.md' in {model_dir}"
    candidates.sort()
    return candidates[-1][1]


def resolve_prompt_path(filename: str, alias: str) -> Path:
    """Resolve a prompt filename within the model-specific directory.

    Prompts live at data/pipeline/prompts/{alias}/{filename}. If the
    model directory doesn't exist yet, initializes it from init templates.

    Supports '_latest.md' suffix (e.g. 'judge_latest.md') which resolves
    to the highest '_vN.md' version found on disk.
    """
    model_dir = PROMPTS_DIR / alias
    if not model_dir.exists():
        _init_model_prompts(alias)
    if "_latest.md" in filename:
        return _resolve_latest_version(model_dir, filename)
    path = model_dir / filename
    assert path.exists(), f"Prompt file not found: {path}"
    return path

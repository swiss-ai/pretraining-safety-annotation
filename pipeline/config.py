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


def _resolve_writing_guidelines_path() -> Path:
    """Read writing_guidelines_path from config YAML."""
    raw = OmegaConf.load(CONFIG_YAML_PATH)
    return PROJECT_ROOT / raw["writing_guidelines_path"]


CHARTER_PATH = _resolve_charter_path()
WRITING_GUIDELINES_PATH = _resolve_writing_guidelines_path()


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
    """Extract charter element IDs from bracketed citations, preserving order.

    Supported citation formats:
    - ``[1.2]``                 single citation
    - ``[1.2][1.4]``            consecutive bracketed citations
    - ``[1.2,1.4]`` / ``[1.2, 1.4]``  comma-separated within one bracket pair

    Only returns IDs that exist in the charter, deduplicated in first-seen order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for group in re.findall(r"\[([0-9., ]+)\]", text):
        for raw in group.split(","):
            candidate = raw.strip()
            if not re.fullmatch(r"\d+\.\d+", candidate):
                continue
            if candidate in _CHARTER_ID_SET and candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


def union_charter_elements(*texts: str | None) -> list[str]:
    """Order-preserving union of charter elements extracted from multiple texts."""
    seen: set[str] = set()
    result: list[str] = []
    for text in texts:
        if not text:
            continue
        for el in extract_charter_elements(text):
            if el not in seen:
                seen.add(el)
                result.append(el)
    return result


# --- Dataclasses ---


@dataclass
class ModelConfig:
    alias: str = ""
    api_name: str = ""
    hf_slug: str = ""
    thinking: bool = False
    endpoint: str = ""
    json_mode: bool = False
    completion_max_tokens: int | None = None
    context_window_tokens: int | None = None


@dataclass
class ImproverConfig:
    judge_prompt: str = "improver_judge.md"
    generator_prompt: str = "improver_generator.md"
    max_batches_per_phase: int = 5
    timeout_s: int = 900
    trusted_reviewers: list[str] = field(default_factory=lambda: ["Julian"])


@dataclass
class ScoringConfig:
    scale_min: int = 1
    scale_max: int = 5
    accept_threshold: int = 4
    floor_threshold: int = 2
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
class CandidateModel:
    """One candidate model in a phase 3 eval (generator or judge).

    Unlike `ModelConfig`, the prompt filename is explicit (e.g.
    `judge_v3.md`) — we never auto-pick the latest. This makes evals
    reproducible across re-runs even if new prompt versions land.
    """

    alias: str = ""
    api_name: str = ""
    hf_slug: str = ""
    endpoint: str = ""  # per-model override; falls back to phase3.endpoint
    prompt_reflection: str = ""  # e.g. "generator_reflection_v7.md"
    prompt_preflection: str = ""  # e.g. "generator_preflection_v2.md"
    thinking: bool = False
    json_mode: bool = False
    completion_max_tokens: int | None = None
    context_window_tokens: int | None = None


@dataclass
class GeneratorEvalConfig:
    candidates: list[CandidateModel] = field(default_factory=list)
    gold_prompt_reflection: str = ""  # override gold_judge prompt for this eval
    gold_prompt_preflection: str = ""
    mode: str = ""  # "reflection", "preflection", or "" for both
    n_items: int = 5000
    seed: int = 42
    max_concurrent: int = 50
    chunk_size: int = 200
    store_reasoning: bool = False
    failure_attempt_cap: int = 3


@dataclass
class JudgeEvalConfig:
    candidates: list[CandidateModel] = field(default_factory=list)
    generator: CandidateModel = field(default_factory=CandidateModel)
    n_items: int = 5000
    seed: int = 42
    max_concurrent: int = 50
    chunk_size: int = 200
    include_reviewed: bool = True
    reviewer_policy: str = "average"  # "average" | "first" | "all"
    store_reasoning: bool = False
    failure_attempt_cap: int = 3


@dataclass
class Phase3Config:
    endpoint: str = ""
    eval_dir: str = ""  # root for run dirs; resolves env vars
    gold_judge: CandidateModel = field(default_factory=CandidateModel)
    generator_eval: GeneratorEvalConfig = field(default_factory=GeneratorEvalConfig)
    judge_eval: JudgeEvalConfig = field(default_factory=JudgeEvalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass
class Phase4SglangConfig:
    hf_slug: str = ""
    model_path: str = ""  # local HF cache path on /capstor/, or empty to download
    tp_size: int = 4
    dp_size: int = 1
    port: int = 30000
    reasoning_parser: str = ""  # sglang --reasoning-parser (e.g. glm45, kimi_k2)
    env_toml: str = ""  # path to sglang TOML (selects container image)
    extra_args: str = ""  # model-specific sglang flags
    pre_launch_cmds: str = ""  # pip installs inside container


@dataclass
class Phase4SlurmConfig:
    partition: str = "normal"
    account: str = "a141"
    time: str = "24:00:00"
    cpus_per_task: int = 4
    mem_per_cpu_gb: int = 8
    workers: int = -1


@dataclass
class Phase4Config:
    sidecar_path: str = ""
    output_dir: str = ""
    reflection_prompt: str = "generator_reflection_v1.md"
    preflection_prompt: str = "generator_preflection_v1.md"
    generator_alias: str = "glm-4.5-air"
    thinking: bool = False
    json_mode: bool = False
    max_rows: int = 0  # 0 = all rows
    rows_per_task: int = 100000
    max_concurrent_requests: int = 2048
    save_batch_size: int = 200
    progress_interval: int = 1000
    canary_seed: int = 42
    reflection_seed: int = 42  # independent from canary_seed
    max_retries_per_doc: int = 5
    sglang: Phase4SglangConfig = field(default_factory=Phase4SglangConfig)
    slurm: Phase4SlurmConfig = field(default_factory=Phase4SlurmConfig)


@dataclass
class AppConfig:
    charter_path: str = MISSING
    writing_guidelines_path: str = MISSING
    data_dir: str = "data"
    max_tokens: int = 3840
    api_keys: dict[str, str] = field(default_factory=dict)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
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

    Copies init templates to versioned v1 files for each role+mode combination:
    - init_generator_reflection.md -> generator_reflection_v1.md
    - init_generator_preflection.md -> generator_preflection_v1.md
    - init_judge_reflection.md -> judge_reflection_v1.md
    - init_judge_preflection.md -> judge_preflection_v1.md
    Only runs once per model (skips if dir already exists).
    """
    import shutil

    model_dir = PROMPTS_DIR / alias
    if model_dir.exists():
        return
    model_dir.mkdir(parents=True)
    for init_name, v1_name in [
        ("init_generator_reflection.md", "generator_reflection_v1.md"),
        ("init_generator_preflection.md", "generator_preflection_v1.md"),
        ("init_judge_reflection.md", "judge_reflection_v1.md"),
        ("init_judge_preflection.md", "judge_preflection_v1.md"),
    ]:
        src = _INIT_PROMPTS_DIR / init_name
        assert src.exists(), f"Init template not found: {src}"
        shutil.copy2(src, model_dir / v1_name)


def _resolve_latest_version(model_dir: Path, filename: str) -> Path:
    """Resolve a 'latest' prompt filename to the highest versioned file.

    E.g. 'judge_reflection_latest.md' finds the highest 'judge_reflection_vN.md' in model_dir.
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


_EXPLICIT_VERSION_RE = re.compile(
    r"^(generator|judge)_(reflection|preflection)_v\d+\.md$"
)


def resolve_prompt_path(filename: str, alias: str) -> Path:
    """Resolve a prompt filename within the model-specific directory.

    Prompts live at data/pipeline/prompts/{alias}/{filename}. If the
    model directory doesn't exist yet AND we're being asked for a
    `_latest.md` flow, initializes it from init templates.

    Supports '_latest.md' suffix (e.g. 'judge_latest.md') which resolves
    to the highest '_vN.md' version found on disk.

    Explicit version filenames (e.g. 'judge_v3.md') do NOT trigger init —
    they assert that the file already exists, so a typo'd alias surfaces
    as a missing-file error rather than silently materialising a stub
    directory.
    """
    model_dir = PROMPTS_DIR / alias
    is_explicit_version = bool(_EXPLICIT_VERSION_RE.match(filename))
    if not model_dir.exists() and not is_explicit_version:
        _init_model_prompts(alias)
    if "_latest.md" in filename:
        return _resolve_latest_version(model_dir, filename)
    path = model_dir / filename
    assert path.exists(), f"Prompt file not found: {path}"
    return path

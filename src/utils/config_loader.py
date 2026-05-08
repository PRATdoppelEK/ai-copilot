"""
Configuration loader — reads config.yaml and exposes a typed config object.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMConfig:
    model: str = "llama3"
    temperature: float = 0.2
    max_tokens: int = 2048


@dataclass
class CandidateConfig:
    name: str = "Prateek Gaur"
    email: str = "prateekgaur@gmx.de"
    location: str = "Munich, Germany"
    linkedin: str = ""
    cv_files: List[str] = field(default_factory=list)
    profile_summary: str = ""


@dataclass
class JobSearchConfig:
    queries: List[str] = field(default_factory=list)
    max_results_per_query: int = 5


@dataclass
class NotificationConfig:
    ntfy_topic: str = ""
    enabled: bool = False


@dataclass
class CopilotConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    job_search: JobSearchConfig = field(default_factory=JobSearchConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    paths: dict = field(default_factory=dict)


def load_config(config_path: str = "configs/config.yaml") -> CopilotConfig:
    """Load YAML config file and return typed CopilotConfig object."""
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found at '{config_path}'. Using defaults.")
        return CopilotConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = CopilotConfig()

    if "llm" in raw:
        cfg.llm = LLMConfig(**raw["llm"])

    if "candidate" in raw:
        c = raw["candidate"]
        cfg.candidate = CandidateConfig(
            name            = c.get("name", ""),
            email           = c.get("email", ""),
            location        = c.get("location", ""),
            linkedin        = c.get("linkedin", ""),
            cv_files        = c.get("cv_files", []),
            profile_summary = c.get("profile_summary", "").strip(),
        )

    if "job_search" in raw:
        j = raw["job_search"]
        cfg.job_search = JobSearchConfig(
            queries               = j.get("queries", []),
            max_results_per_query = j.get("max_results_per_query", 5),
        )

    if "notifications" in raw:
        n = raw["notifications"]
        cfg.notifications = NotificationConfig(
            ntfy_topic = n.get("ntfy_topic", ""),
            enabled    = n.get("enabled", False),
        )

    cfg.paths = raw.get("paths", {})
    return cfg

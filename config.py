from __future__ import annotations

"""Typed configuration and user-profile loading for the automation agent."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class UserProfile:
    full_name: str = ""
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = ""
    linkedin: str = ""
    website: str = ""
    company: str = ""
    role: str = ""
    summary: str = ""
    college_name: str = ""
    university_name: str = ""
    degree: str = ""
    specialization: str = ""
    degree_specialization: str = ""
    graduation_score: str = ""
    post_graduation_score: str = ""
    active_backlogs: str = ""
    custom_fields: Dict[str, str] = field(default_factory=dict)

    def as_prompt_dict(self) -> Dict[str, str]:
        data = {
            "full_name": self.full_name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "postal_code": self.postal_code,
            "country": self.country,
            "linkedin": self.linkedin,
            "website": self.website,
            "company": self.company,
            "role": self.role,
            "summary": self.summary,
            "college_name": self.college_name,
            "university_name": self.university_name,
            "degree": self.degree,
            "specialization": self.specialization,
            "degree_specialization": self.degree_specialization,
            "graduation_score": self.graduation_score,
            "post_graduation_score": self.post_graduation_score,
            "active_backlogs": self.active_backlogs,
        }
        data.update(self.custom_fields)
        return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}


@dataclass
class AIConfig:
    local_model: str = "llama3.1:8b"
    local_endpoint: str = "http://localhost:11434/api/chat"
    local_timeout_seconds: int = 120
    local_temperature: float = 0.1
    min_confidence: float = 0.6
    enable_fallback: bool = False
    fallback_provider: str = "gemini"  # openai | gemini
    fallback_model: str = "gemini-2.5-flash"
    fallback_timeout_seconds: int = 90
    openai_endpoint: str = "https://api.openai.com/v1/chat/completions"
    openai_api_key_env: str = "OPENAI_API_KEY"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_endpoint_template: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={api_key}"
    )


@dataclass
class BrowserConfig:
    browser_channel: str = "chromium"  # chromium | chrome | msedge
    connect_cdp_url: str = ""
    chrome_user_data_dir: str = ""
    chrome_profile_directory: str = "Default"
    headless: bool = False
    slow_mo_ms: int = 0
    profile_dir: str = ""
    navigation_timeout_ms: int = 45_000
    action_timeout_ms: int = 8_000
    retry_attempts: int = 2
    min_delay_seconds: float = 0.4
    max_delay_seconds: float = 1.1
    auth_wait_seconds: int = 180
    attempt_auth_click: bool = True
    screenshot_dir: str = "debug_screenshots"
    take_debug_screenshots: bool = True
    multi_page_enabled: bool = True
    max_pages: int = 5
    auto_submit: bool = False


@dataclass
class AgentConfig:
    target_url: str
    user_data_path: str = "user_data.json"
    log_level: str = "INFO"
    log_file: str = "agent.log"
    ai: AIConfig = field(default_factory=AIConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    user_profile: UserProfile = field(default_factory=UserProfile)


def _coerce_str(data: Dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def load_user_profile(path: str) -> UserProfile:
    profile_path = Path(path)
    if not profile_path.exists():
        return UserProfile()

    with profile_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"User profile file must be a JSON object: {path}")

    reserved = {
        "full_name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "city",
        "state",
        "postal_code",
        "country",
        "linkedin",
        "website",
        "company",
        "role",
        "summary",
        "college_name",
        "university_name",
        "degree",
        "specialization",
        "degree_specialization",
        "graduation_score",
        "post_graduation_score",
        "active_backlogs",
    }
    custom = {k: str(v) for k, v in raw.items() if k not in reserved and v is not None}

    return UserProfile(
        full_name=_coerce_str(raw, "full_name"),
        first_name=_coerce_str(raw, "first_name"),
        last_name=_coerce_str(raw, "last_name"),
        email=_coerce_str(raw, "email"),
        phone=_coerce_str(raw, "phone"),
        address=_coerce_str(raw, "address"),
        city=_coerce_str(raw, "city"),
        state=_coerce_str(raw, "state"),
        postal_code=_coerce_str(raw, "postal_code"),
        country=_coerce_str(raw, "country"),
        linkedin=_coerce_str(raw, "linkedin"),
        website=_coerce_str(raw, "website"),
        company=_coerce_str(raw, "company"),
        role=_coerce_str(raw, "role"),
        summary=_coerce_str(raw, "summary"),
        college_name=_coerce_str(raw, "college_name"),
        university_name=_coerce_str(raw, "university_name"),
        degree=_coerce_str(raw, "degree"),
        specialization=_coerce_str(raw, "specialization"),
        degree_specialization=_coerce_str(raw, "degree_specialization"),
        graduation_score=_coerce_str(raw, "graduation_score"),
        post_graduation_score=_coerce_str(raw, "post_graduation_score"),
        active_backlogs=_coerce_str(raw, "active_backlogs"),
        custom_fields=custom,
    )


def bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

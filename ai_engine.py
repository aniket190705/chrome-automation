from __future__ import annotations

"""Hybrid decision engine: heuristics -> API (Gemini/OpenAI) -> local Ollama fallback."""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from browser import FieldMetadata
from config import AIConfig, UserProfile


@dataclass
class Decision:
    action: str
    value: str
    confidence: float
    source: str
    raw: str = ""
    valid: bool = True
    reason: str = ""


class RuleBasedEngine:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile
        self.rules = [
            ("email", ["email", "e-mail", "mail id"]),
            ("phone", ["phone", "mobile", "telephone", "contact number", "cell"]),
            ("college_name", ["college/university", "college or university", "college", "university", "school name"]),
            ("degree_specialization", ["degree and area of specialization", "degree & area of specialization", "degree - specialization"]),
            ("specialization", ["specialization", "specialisation", "major", "stream"]),
            ("degree", ["degree", "qualification", "course"]),
            ("graduation_score", ["graduation percentage", "graduation cgpa", "cgpa (out of 10)", "percentage or cgpa", "graduation score"]),
            ("post_graduation_score", ["post graduation percentage", "post graduation cgpa", "post-graduation percentage", "post graduation score"]),
            ("active_backlogs", ["active backlogs", "current backlogs", "any backlogs", "standing arrears"]),
            ("linkedin", ["linkedin", "linked in"]),
            ("website", ["website", "portfolio", "site", "url"]),
            ("company", ["company", "organization", "employer"]),
            ("role", ["current role", "job title", "designation", "position"]),
            ("first_name", ["first name", "given name"]),
            ("last_name", ["last name", "surname", "family name"]),
            ("full_name", ["full name", "your name", "applicant name", "first and last name"]),
            ("address", ["address", "street"]),
            ("city", ["city", "town"]),
            ("state", ["state", "province", "region"]),
            ("postal_code", ["zip", "postal", "pin code", "postcode"]),
            ("country", ["country", "nation"]),
        ]

    def resolve(self, field: FieldMetadata) -> Optional[Decision]:
        text_blob = " ".join(
            [
                field.name.lower(),
                field.element_id.lower(),
                field.label.lower(),
                field.placeholder.lower(),
                field.near_text.lower(),
            ]
        ).strip()

        if not text_blob:
            return None

        if self._is_plain_name_field(text_blob):
            value = self._value_for_key("full_name")
            if value:
                return Decision(
                    action="fill" if field.kind not in {"radio", "checkbox", "select"} else "select",
                    value=value,
                    confidence=0.99,
                    source="rule",
                    reason="Matched plain person-name field",
                )

        for key, keywords in self.rules:
            if any(keyword in text_blob for keyword in keywords):
                value = self._value_for_key(key)
                if value:
                    return Decision(
                        action="fill" if field.kind not in {"radio", "checkbox", "select"} else "select",
                        value=value,
                        confidence=0.99,
                        source="rule",
                        reason=f"Matched rule key '{key}'",
                    )
        custom = self._custom_match(text_blob)
        if custom:
            return Decision(
                action="fill",
                value=custom,
                confidence=0.9,
                source="rule",
                reason="Matched custom field key",
            )
        return None

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _is_plain_name_field(self, text_blob: str) -> bool:
        normalized = self._normalize_text(text_blob)
        if "name" not in normalized.split():
            return False
        blocked_terms = {
            "college",
            "university",
            "school",
            "company",
            "organization",
            "organisation",
            "degree",
            "specialization",
            "specialisation",
            "father",
            "mother",
            "guardian",
            "reference",
            "linkedin",
            "project",
        }
        if any(term in normalized.split() for term in blocked_terms):
            return False
        allowed_phrases = {
            "name",
            "your name",
            "full name",
            "applicant name",
            "candidate name",
            "first and last name",
        }
        return any(phrase in normalized for phrase in allowed_phrases)

    def _custom_match(self, text_blob: str) -> str:
        normalized_blob = self._normalize_text(text_blob)
        for key, value in self.profile.custom_fields.items():
            normalized_key = self._normalize_text(key.replace("_", " "))
            key_tokens = [token for token in normalized_key.split() if token]
            if not value.strip():
                continue
            if normalized_key and normalized_key in normalized_blob:
                return value
            if key_tokens and all(token in normalized_blob.split() for token in key_tokens):
                return value
        return ""

    def _value_for_key(self, key: str) -> str:
        direct = getattr(self.profile, key, "")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        if key == "full_name":
            name = " ".join([self.profile.first_name.strip(), self.profile.last_name.strip()]).strip()
            return name
        if key == "college_name":
            return self.profile.college_name.strip() or self.profile.university_name.strip()
        if key == "degree_specialization":
            direct_combo = self.profile.degree_specialization.strip()
            if direct_combo:
                return direct_combo
            degree = self.profile.degree.strip()
            specialization = self.profile.specialization.strip()
            if degree and specialization:
                return f"{degree} - {specialization}"
            return degree or specialization
        return ""


class BaseAIClient:
    @staticmethod
    def build_prompt(field: FieldMetadata, profile: UserProfile) -> str:
        options = [{"text": o.text, "value": o.value} for o in field.options if o.text or o.value]
        payload = {
            "question": field.question_text,
            "field_type": field.kind,
            "name": field.name,
            "required": field.required,
            "options": options,
            "user_profile": profile.as_prompt_dict(),
            "instruction": (
                "Return JSON only with keys action, value, confidence. "
                "action must be fill or select. confidence between 0 and 1."
            ),
            "json_schema": {"action": "fill|select", "value": "string", "confidence": 0.0},
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def parse_decision(raw_text: str, source: str) -> Decision:
        if not raw_text:
            return Decision(action="", value="", confidence=0.0, source=source, valid=False, reason="Empty response")

        cleaned = raw_text.strip()
        candidate = BaseAIClient._extract_json_object(cleaned)
        if not candidate:
            return Decision(
                action="",
                value="",
                confidence=0.0,
                source=source,
                raw=cleaned,
                valid=False,
                reason="No JSON object found",
            )
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return Decision(
                action="",
                value="",
                confidence=0.0,
                source=source,
                raw=cleaned,
                valid=False,
                reason="Invalid JSON",
            )

        action = str(data.get("action", "")).strip().lower()
        value = str(data.get("value", "")).strip()
        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if action not in {"fill", "select"} or not value:
            return Decision(
                action=action,
                value=value,
                confidence=confidence,
                source=source,
                raw=cleaned,
                valid=False,
                reason="Missing/invalid action or value",
            )
        return Decision(
            action=action,
            value=value,
            confidence=confidence if confidence > 0 else 0.55,
            source=source,
            raw=cleaned,
            valid=True,
        )

    @staticmethod
    def _extract_json_object(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)
        simple = re.search(r"\{.*\}", text, re.DOTALL)
        return simple.group(0) if simple else ""


class LocalOllamaEngine(BaseAIClient):
    def __init__(self, config: AIConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.session = requests.Session()

    def decide(self, field: FieldMetadata, profile: UserProfile) -> Decision:
        prompt = self.build_prompt(field, profile)
        system = (
            "You are a browser form assistant. "
            "Respond with strict JSON only."
        )
        payload = {
            "model": self.config.local_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.config.local_temperature},
        }

        try:
            response = self.session.post(
                self.config.local_endpoint,
                json=payload,
                timeout=self.config.local_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = (
                body.get("message", {}).get("content")
                or body.get("response")
                or ""
            )
            decision = self.parse_decision(str(content), source="local_ai")
            return decision
        except requests.RequestException as exc:
            self.logger.warning("Local Ollama request failed: %s", exc)
            return Decision(action="", value="", confidence=0.0, source="local_ai", valid=False, reason=str(exc))


class FallbackAPIEngine(BaseAIClient):
    def __init__(self, config: AIConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.session = requests.Session()
        self.gemini_disabled_reason: str = ""

    def decide(self, field: FieldMetadata, profile: UserProfile) -> Decision:
        provider = self.config.fallback_provider.strip().lower()
        if provider == "openai":
            return self._call_openai(field, profile)
        if provider == "gemini":
            return self._call_gemini(field, profile)
        return Decision(
            action="",
            value="",
            confidence=0.0,
            source="fallback_ai",
            valid=False,
            reason=f"Unsupported provider '{provider}'",
        )

    def _call_openai(self, field: FieldMetadata, profile: UserProfile) -> Decision:
        api_key = os.getenv(self.config.openai_api_key_env, "").strip()
        if not api_key:
            return Decision(
                action="",
                value="",
                confidence=0.0,
                source="fallback_ai",
                valid=False,
                reason=f"Missing env var {self.config.openai_api_key_env}",
            )

        payload = {
            "model": self.config.fallback_model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": self.build_prompt(field, profile)},
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = self.session.post(
                self.config.openai_endpoint,
                json=payload,
                headers=headers,
                timeout=self.config.fallback_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self.parse_decision(str(content), source="fallback_ai")
        except requests.RequestException as exc:
            self.logger.warning("OpenAI fallback request failed: %s", exc)
            return Decision(action="", value="", confidence=0.0, source="fallback_ai", valid=False, reason=str(exc))

    @staticmethod
    def _normalize_gemini_model_name(model_name: str) -> str:
        model = model_name.strip()
        if model.startswith("models/"):
            model = model.split("/", 1)[1]
        return model

    def _gemini_model_candidates(self) -> List[str]:
        configured = self._normalize_gemini_model_name(self.config.fallback_model)
        candidates = [
            configured,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-flash-latest",
        ]
        unique: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique

    @staticmethod
    def _extract_error_message(response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or f"HTTP {response.status_code}"

        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            status = str(error.get("status", "")).strip()
            message = str(error.get("message", "")).strip()
            if status and message:
                return f"{status}: {message}"
            if message:
                return message
        return response.text.strip() or f"HTTP {response.status_code}"

    def _call_gemini(self, field: FieldMetadata, profile: UserProfile) -> Decision:
        api_key = os.getenv(self.config.gemini_api_key_env, "").strip()
        if not api_key:
            return Decision(
                action="",
                value="",
                confidence=0.0,
                source="fallback_ai",
                valid=False,
                reason=f"Missing env var {self.config.gemini_api_key_env}",
            )
        if self.gemini_disabled_reason:
            return Decision(
                action="",
                value="",
                confidence=0.0,
                source="fallback_ai",
                valid=False,
                reason=self.gemini_disabled_reason,
            )

        payload = {
            "contents": [{"parts": [{"text": "Return strict JSON only.\n" + self.build_prompt(field, profile)}]}],
            "generationConfig": {"temperature": 0.1},
        }
        last_error = "No compatible Gemini model found"
        for model_name in self._gemini_model_candidates():
            endpoint = self.config.gemini_endpoint_template.format(model=model_name, api_key=api_key)
            try:
                response = self.session.post(
                    endpoint,
                    json=payload,
                    timeout=self.config.fallback_timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                candidates = body.get("candidates", [])
                parts: List[Dict[str, str]] = (
                    candidates[0].get("content", {}).get("parts", []) if candidates else []
                )
                text = " ".join(str(part.get("text", "")) for part in parts)
                decision = self.parse_decision(text, source="fallback_ai")
                if decision.valid:
                    self.logger.info("Gemini fallback model used: %s", model_name)
                return decision
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                error_message = self._extract_error_message(exc.response) if exc.response is not None else str(exc)
                last_error = error_message
                if status_code == 404:
                    self.logger.info("Gemini model unavailable (%s), trying next candidate", model_name)
                    continue
                if status_code == 403:
                    self.gemini_disabled_reason = error_message
                self.logger.warning("Gemini fallback request failed (%s): %s", model_name, error_message)
                return Decision(
                    action="",
                    value="",
                    confidence=0.0,
                    source="fallback_ai",
                    valid=False,
                    reason=error_message,
                )
            except requests.RequestException as exc:
                self.logger.warning("Gemini fallback request failed (%s): %s", model_name, exc)
                return Decision(
                    action="",
                    value="",
                    confidence=0.0,
                    source="fallback_ai",
                    valid=False,
                    reason=str(exc),
                )
        return Decision(
            action="",
            value="",
            confidence=0.0,
            source="fallback_ai",
            valid=False,
            reason=last_error,
        )


class HybridDecisionEngine:
    def __init__(self, profile: UserProfile, config: AIConfig, logger) -> None:
        self.profile = profile
        self.config = config
        self.logger = logger
        self.rule_engine = RuleBasedEngine(profile)
        self.local_engine = LocalOllamaEngine(config, logger)
        self.fallback_engine = FallbackAPIEngine(config, logger) if config.enable_fallback else None

    def decide(self, field: FieldMetadata) -> Optional[Decision]:
        rule_decision = self.rule_engine.resolve(field)
        if rule_decision:
            return rule_decision

        fallback: Optional[Decision] = None
        if self.fallback_engine:
            fallback = self.fallback_engine.decide(field, self.profile)
            if fallback.valid and fallback.confidence >= self.config.min_confidence:
                return fallback

            self.logger.info(
                "Primary API unresolved for field '%s' (valid=%s confidence=%.2f reason=%s)",
                field.question_text,
                fallback.valid,
                fallback.confidence,
                fallback.reason,
            )

        local = self.local_engine.decide(field, self.profile)
        if local.valid and local.confidence >= self.config.min_confidence:
            return local

        self.logger.info(
            "Local Ollama fallback unresolved for field '%s' (valid=%s confidence=%.2f reason=%s)",
            field.question_text,
            local.valid,
            local.confidence,
            local.reason,
        )

        # If neither clears the confidence gate, prefer a valid primary API answer.
        if fallback and fallback.valid:
            return fallback
        return local if local.valid else None

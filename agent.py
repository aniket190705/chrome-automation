from __future__ import annotations

"""Core orchestration loop for multi-page hybrid form automation."""

import logging
from typing import List

from ai_engine import Decision, HybridDecisionEngine
from browser import BrowserController, FieldMetadata
from config import AgentConfig


class FormAutomationAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.logger = self._build_logger()
        self.browser = BrowserController(config.browser, self.logger)
        self.decision_engine = HybridDecisionEngine(config.user_profile, config.ai, self.logger)

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("hybrid_form_agent")
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        if logger.handlers:
            return logger
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(self.config.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def run(self) -> None:
        self.logger.info("Starting agent for URL: %s", self.config.target_url)
        self.browser.start()
        try:
            self.browser.goto(self.config.target_url)
            self.browser.capture_screenshot("page_loaded")
            if not self.browser.ensure_form_access():
                self.logger.error("Form is locked behind sign-in. Aborting fill run.")
                self.browser.capture_screenshot("auth_required")
                return
            self._run_page_loop()
            if self.config.browser.auto_submit:
                submitted = self.browser.click_submit()
                self.logger.info("Auto submit attempted: %s", submitted)
        finally:
            self.browser.capture_screenshot("run_complete")
            self.browser.stop()
            self.logger.info("Agent stopped")

    def _run_page_loop(self) -> None:
        for page_index in range(1, self.config.browser.max_pages + 1):
            fields = self.browser.extract_fields()
            if not fields:
                self.logger.info("No fields detected on page %s", page_index)
                break

            self.logger.info("Page %s: detected %s fields", page_index, len(fields))
            self._process_fields(fields)

            if not self.config.browser.multi_page_enabled:
                break
            moved = self.browser.click_next_page()
            if not moved:
                break
            self.browser.capture_screenshot(f"next_page_{page_index}")

    def _process_fields(self, fields: List[FieldMetadata]) -> None:
        for index, field in enumerate(fields, start=1):
            if field.disabled:
                self.logger.info("[%s] Skip disabled field: %s", index, field.question_text)
                continue
            if self._should_skip(field):
                self.logger.info("[%s] Skip pre-filled field: %s", index, field.question_text)
                continue

            decision = self.decision_engine.decide(field)
            if not decision:
                self.logger.warning("[%s] No decision for field: %s", index, field.question_text)
                continue

            applied = self.browser.apply_action(field, decision.action, decision.value)
            self._log_decision(index, field, decision, applied)
            if not applied:
                self.browser.capture_screenshot(f"failed_field_{field.agent_id}")

    @staticmethod
    def _should_skip(field: FieldMetadata) -> bool:
        if field.kind in {"checkbox", "radio"}:
            return bool(field.checked)
        if field.value and field.value.strip():
            return True
        return False

    def _log_decision(self, idx: int, field: FieldMetadata, decision: Decision, applied: bool) -> None:
        self.logger.info(
            "[%s] field='%s' kind='%s' source='%s' confidence=%.2f action='%s' value='%s' applied=%s",
            idx,
            field.question_text[:120],
            field.kind,
            decision.source,
            decision.confidence,
            decision.action,
            decision.value[:120],
            applied,
        )

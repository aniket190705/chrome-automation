from __future__ import annotations

"""Playwright browser control, dynamic form introspection, and resilient actions."""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from playwright.sync_api import Browser, BrowserContext, Error, Page, TimeoutError, sync_playwright

from config import BrowserConfig


@dataclass
class OptionMetadata:
    text: str
    value: str
    agent_id: str = ""


@dataclass
class FieldMetadata:
    agent_id: str
    tag: str
    field_type: str
    role: str
    name: str
    element_id: str
    label: str
    placeholder: str
    near_text: str
    required: bool
    disabled: bool = False
    value: str = ""
    checked: bool = False
    group_key: str = ""
    options: List[OptionMetadata] = field(default_factory=list)

    @property
    def question_text(self) -> str:
        parts = [self.label, self.placeholder, self.near_text]
        fallback = " ".join(p for p in parts if p).strip()
        return fallback or self.name or self.element_id

    @property
    def kind(self) -> str:
        if self.tag == "textarea":
            return "text"
        if self.tag == "select":
            return "select"
        return self.field_type or "text"


class BrowserController:
    def __init__(self, config: BrowserConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self._playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._owns_browser = True
        self._target_host: str = ""
        self._target_url: str = ""
        self.screenshot_dir = Path(config.screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self._playwright = sync_playwright().start()
        channel = self._effective_channel()
        cdp_url = self.config.connect_cdp_url.strip()
        chrome_user_data_dir = self.config.chrome_user_data_dir.strip()

        if cdp_url:
            self._owns_browser = False
            self.browser = self._playwright.chromium.connect_over_cdp(cdp_url)
            self.context = self.browser.contexts[0] if self.browser.contexts else self.browser.new_context()
            self.page = self.context.new_page()
            self.logger.info("Connected to existing browser via CDP: %s", cdp_url)
        elif chrome_user_data_dir:
            chrome_data = Path(chrome_user_data_dir).expanduser().resolve()
            chrome_data.mkdir(parents=True, exist_ok=True)
            self.context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(chrome_data),
                channel="chrome",
                headless=self.config.headless,
                slow_mo=self.config.slow_mo_ms,
                args=[f'--profile-directory={self.config.chrome_profile_directory}'],
            )
            self.browser = self.context.browser
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            self.logger.info(
                "Using Chrome user data dir: %s (profile=%s)",
                chrome_data,
                self.config.chrome_profile_directory,
            )
        elif self.config.profile_dir.strip():
            profile_dir = Path(self.config.profile_dir).expanduser().resolve()
            profile_dir.mkdir(parents=True, exist_ok=True)
            self.context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                channel=channel,
                headless=self.config.headless,
                slow_mo=self.config.slow_mo_ms,
            )
            self.browser = self.context.browser
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            self.logger.info("Using persistent browser profile: %s", profile_dir)
        else:
            self.browser = self._playwright.chromium.launch(
                channel=channel,
                headless=self.config.headless,
                slow_mo=self.config.slow_mo_ms,
            )
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
        self.page.set_default_timeout(self.config.action_timeout_ms)
        self.page.set_default_navigation_timeout(self.config.navigation_timeout_ms)

    def stop(self) -> None:
        try:
            if self.context and self._owns_browser:
                self.context.close()
            if self.browser and self._owns_browser:
                self.browser.close()
        finally:
            if self._playwright:
                self._playwright.stop()
            self.context = None
            self.browser = None
            self.page = None
            self._playwright = None
            self._owns_browser = True

    def _effective_channel(self) -> Optional[str]:
        channel = (self.config.browser_channel or "").strip().lower()
        if channel in {"chrome", "msedge"}:
            return channel
        return None

    def goto(self, url: str) -> None:
        if not self.page:
            raise RuntimeError("Browser is not started")
        self._target_url = url
        self._target_host = (urlparse(url).netloc or "").lower()
        self.page.goto(url, wait_until="domcontentloaded")
        self.page.wait_for_timeout(800)
        self.logger.info("Loaded URL: %s", url)

    def ensure_form_access(self) -> bool:
        if not self.page:
            raise RuntimeError("Browser is not started")
        gate_was_seen = self._is_sign_in_gate_present()
        if not gate_was_seen:
            return True

        self.logger.warning("Sign-in gate detected on form page")
        if self.config.headless:
            self.logger.error(
                "Cannot complete manual login in headless mode. "
                "Use headed mode with --profile-dir to save session."
            )
            return False

        if self.config.attempt_auth_click:
            self._click_sign_in_button()

        deadline = time.time() + max(10, self.config.auth_wait_seconds)
        clear_streak = 0
        while time.time() < deadline:
            self.page.wait_for_timeout(1000)
            if self._is_sign_in_gate_present():
                clear_streak = 0
                continue
            if gate_was_seen and not self._is_on_target_domain():
                clear_streak = 0
                continue
            if gate_was_seen and not self._has_fill_candidates():
                clear_streak = 0
                continue
            clear_streak += 1
            if clear_streak >= 2:
                self.logger.info("Sign-in gate cleared, continuing automation")
                return True

        self.logger.error("Timed out waiting for sign-in completion")
        return False

    def _is_sign_in_gate_present(self) -> bool:
        if not self.page:
            return False
        cues = [
            "Sign in to continue",
            "To fill out this form, you must be signed in.",
        ]
        for cue in cues:
            locator = self.page.get_by_text(cue, exact=False)
            if locator.count() > 0 and locator.first.is_visible():
                return True
        return False

    def _click_sign_in_button(self) -> bool:
        if not self.page:
            return False
        selectors = [
            'button:has-text("SIGN IN")',
            'button:has-text("Sign in")',
            '[role="button"]:has-text("SIGN IN")',
            'a:has-text("Sign in")',
            "text=SIGN IN",
        ]
        for selector in selectors:
            locator = self.page.locator(selector)
            if locator.count() == 0:
                continue
            try:
                locator.first.click()
                self.logger.info("Clicked sign-in trigger: %s", selector)
                return True
            except Error:
                continue
        self.logger.warning("Could not click sign-in trigger automatically")
        return False

    def _is_on_target_domain(self) -> bool:
        if not self.page:
            return False
        current_host = (urlparse(self.page.url).netloc or "").lower()
        if not self._target_host:
            return True
        return current_host.endswith(self._target_host)

    def _has_fill_candidates(self) -> bool:
        if not self.page:
            return False
        try:
            return bool(
                self.page.evaluate(
                    """
                    () => {
                      const isVisible = (el) => {
                        const style = window.getComputedStyle(el);
                        if (!style || style.visibility === "hidden" || style.display === "none") return false;
                        const rect = el.getBoundingClientRect();
                        return rect.width > 0 && rect.height > 0;
                      };
                      const candidates = document.querySelectorAll(
                        'input, textarea, select, [role="textbox"], [role="radio"], [role="checkbox"], [role="combobox"], [contenteditable="true"]'
                      );
                      for (const el of candidates) {
                        if (!(el instanceof HTMLElement)) continue;
                        if (!isVisible(el)) continue;
                        if ((el.getAttribute("aria-hidden") || "").toLowerCase() === "true") continue;
                        if (el.tagName.toLowerCase() === "input" && (el.getAttribute("type") || "").toLowerCase() === "hidden") continue;
                        return true;
                      }
                      return false;
                    }
                    """
                )
            )
        except Error:
            return False

    def _random_delay(self) -> None:
        delay = random.uniform(self.config.min_delay_seconds, self.config.max_delay_seconds)
        time.sleep(delay)

    def capture_screenshot(self, label: str) -> Optional[str]:
        if not self.page or not self.config.take_debug_screenshots:
            return None
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:80]
        timestamp = int(time.time() * 1000)
        path = self.screenshot_dir / f"{timestamp}_{safe_label}.png"
        try:
            self.page.screenshot(path=str(path), full_page=True)
            self.logger.info("Captured screenshot: %s", path)
            return str(path)
        except Error as exc:
            self.logger.warning("Skipping screenshot '%s': %s", label, exc)
            return None

    def extract_fields(self) -> List[FieldMetadata]:
        if not self.page:
            raise RuntimeError("Browser is not started")

        raw_fields = self.page.evaluate(
            """
            () => {
              // Build metadata from native and ARIA-driven controls without site-specific selectors.
              const normalize = (txt) => (txt || "").replace(/\\s+/g, " ").trim();
              const roleValue = (el) => normalize(el.getAttribute("role") || "").toLowerCase();
              const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                if (!style || style.visibility === "hidden" || style.display === "none") return false;
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              };
              const textFromIds = (ids) => {
                if (!ids) return "";
                const chunks = [];
                for (const id of ids.split(/\\s+/)) {
                  const node = document.getElementById(id);
                  const text = normalize(node ? node.textContent : "");
                  if (text) chunks.push(text);
                }
                return normalize(chunks.join(" "));
              };
              const ensureAgentId = (el) => {
                if (!window.__aiAgentCounter) window.__aiAgentCounter = 1;
                if (!el.dataset.aiAgentId) {
                  el.dataset.aiAgentId = `ai-field-${window.__aiAgentCounter++}`;
                }
                return el.dataset.aiAgentId;
              };
              const labelText = (el) => {
                if (!el) return "";
                const ariaLabel = normalize(el.getAttribute("aria-label") || "");
                if (ariaLabel) return ariaLabel;
                const ariaLabelledBy = textFromIds(el.getAttribute("aria-labelledby"));
                if (ariaLabelledBy) return ariaLabelledBy;
                if (el.labels && el.labels.length) {
                  const v = normalize(el.labels[0].textContent || "");
                  if (v) return v;
                }
                if (el.id) {
                  const found = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
                  const v = normalize(found ? found.textContent : "");
                  if (v) return v;
                }
                const closest = el.closest("label");
                const c = normalize(closest ? closest.textContent : "");
                if (c) return c;
                const fieldset = el.closest("fieldset");
                const legend = normalize(fieldset ? (fieldset.querySelector("legend") || {}).textContent : "");
                if (legend) return legend;
                const item = el.closest('[role="listitem"], [role="group"], [role="radiogroup"]');
                if (item) {
                  const heading = item.querySelector('[role="heading"], h1, h2, h3, h4, h5, h6');
                  const headingText = normalize(heading ? heading.textContent : "");
                  if (headingText) return headingText;
                }
                return "";
              };
              const nearbyText = (el) => {
                if (!el) return "";
                const attrs = ["title", "data-testid"];
                for (const attr of attrs) {
                  const v = normalize(el.getAttribute(attr) || "");
                  if (v) return v;
                }
                const describedBy = textFromIds(el.getAttribute("aria-describedby"));
                if (describedBy) return describedBy;
                const around = [
                  el.previousElementSibling,
                  el.nextElementSibling,
                  el.parentElement,
                  el.closest("div"),
                  el.closest("fieldset"),
                  el.closest('[role="listitem"]')
                ];
                for (const node of around) {
                  const text = normalize(node ? node.textContent : "");
                  if (text && text.length < 220) return text;
                }
                return "";
              };
              const inferType = (el, tag, role) => {
                if (tag === "input") {
                  return normalize(el.getAttribute("type") || "text").toLowerCase();
                }
                if (tag === "textarea") return "text";
                if (tag === "select") return "select";
                if (role === "textbox" || role === "searchbox") return "text";
                if (role === "combobox" || role === "listbox") return "select";
                if (role === "radio") return "radio";
                if (role === "checkbox" || role === "switch") return "checkbox";
                if (el.isContentEditable) return "text";
                return "";
              };
              const collectSelectOptions = (el) => {
                const role = roleValue(el);
                const options = [];
                const seen = new Set();
                const addOption = (optEl) => {
                  if (!optEl || !isVisible(optEl)) return;
                  const text = normalize(optEl.textContent || optEl.getAttribute("aria-label") || "");
                  const value = normalize(
                    optEl.getAttribute("data-value") ||
                    optEl.getAttribute("value") ||
                    optEl.getAttribute("aria-label") ||
                    text
                  );
                  if (!text && !value) return;
                  const key = `${text}|${value}`;
                  if (seen.has(key)) return;
                  seen.add(key);
                  options.push({
                    text,
                    value,
                    agent_id: ensureAgentId(optEl)
                  });
                };

                if ((el.tagName || "").toLowerCase() === "select") {
                  for (const opt of Array.from(el.options || [])) {
                    const text = normalize(opt.textContent || "");
                    const value = normalize(opt.value || "");
                    if (!text && !value) continue;
                    options.push({ text, value, agent_id: "" });
                  }
                  return options;
                }

                let optionNodes = Array.from(el.querySelectorAll('[role="option"]'));
                const controls = el.getAttribute("aria-controls");
                if (controls) {
                  for (const id of controls.split(/\\s+/)) {
                    const controlled = document.getElementById(id);
                    if (controlled) {
                      optionNodes = optionNodes.concat(Array.from(controlled.querySelectorAll('[role="option"]')));
                    }
                  }
                }
                if (!optionNodes.length && role === "combobox") {
                  const expanded = (el.getAttribute("aria-expanded") || "").toLowerCase() === "true";
                  if (expanded) {
                    optionNodes = Array.from(document.querySelectorAll('[role="option"]'));
                  }
                }
                for (const opt of optionNodes) addOption(opt);
                return options;
              };

              const elements = Array.from(document.querySelectorAll(
                [
                  "input",
                  "textarea",
                  "select",
                  '[role="textbox"]',
                  '[role="searchbox"]',
                  '[role="combobox"]',
                  '[role="listbox"]',
                  '[role="radio"]',
                  '[role="checkbox"]',
                  '[role="switch"]',
                  '[contenteditable="true"]'
                ].join(",")
              ));
              const fields = [];
              const seen = new Set();

              for (const el of elements) {
                if (!(el instanceof HTMLElement)) continue;
                if (seen.has(el)) continue;
                seen.add(el);

                const tag = (el.tagName || "").toLowerCase();
                const role = roleValue(el);
                const inputType = inferType(el, tag, role);
                if (!inputType) continue;
                if (!isVisible(el)) continue;
                if ((el.getAttribute("aria-hidden") || "").toLowerCase() === "true") continue;
                if (tag === "input" && ["hidden", "submit", "button", "image", "reset", "file"].includes(inputType)) continue;

                // Skip wrapper elements when a native input exists inside and the wrapper is not an explicit choice control.
                if (!["input", "textarea", "select"].includes(tag) && role && !["radio", "checkbox", "switch"].includes(role)) {
                  if (el.querySelector("input, textarea, select")) continue;
                }

                const agentId = ensureAgentId(el);
                const ariaChecked = (el.getAttribute("aria-checked") || "").toLowerCase();
                const ariaRequired = (el.getAttribute("aria-required") || "").toLowerCase();
                const isDisabled = !!el.disabled || (el.getAttribute("aria-disabled") || "").toLowerCase() === "true";
                let value = "";
                if (["input", "textarea", "select"].includes(tag)) {
                  value = normalize(el.value || "");
                } else if (["text", "email", "tel", "url", "search", "number", "password"].includes(inputType)) {
                  value = normalize(el.textContent || "");
                } else if (inputType === "select") {
                  value = normalize(el.getAttribute("aria-valuetext") || el.getAttribute("data-value") || "");
                } else {
                  value = normalize(el.getAttribute("value") || el.getAttribute("data-value") || "");
                }
                const required = !!el.required || ariaRequired === "true";
                const checked =
                  (tag === "input" && !!el.checked) ||
                  (["radio", "checkbox"].includes(inputType) && ariaChecked === "true");
                let label = labelText(el);
                const nearText = nearbyText(el);
                if (["radio", "checkbox"].includes(inputType)) {
                  const questionRoot =
                    el.closest('[role="listitem"]') ||
                    el.closest('[role="group"]') ||
                    el.closest('[role="radiogroup"]');
                  if (questionRoot) {
                    const heading = questionRoot.querySelector('[role="heading"], h1, h2, h3, h4, h5, h6');
                    const headingText = normalize(heading ? heading.textContent : "");
                    if (headingText) {
                      label = headingText;
                    }
                  }
                }
                const name = normalize(el.getAttribute("name") || el.getAttribute("data-name") || label || "");
                const elementId = normalize(el.getAttribute("id") || "");
                const placeholder = normalize(el.getAttribute("placeholder") || "");

                const base = {
                  agent_id: agentId,
                  tag,
                  role,
                  field_type: inputType,
                  name,
                  element_id: elementId,
                  label,
                  placeholder,
                  near_text: nearText,
                  required,
                  disabled: isDisabled,
                  value,
                  checked,
                  group_key: "",
                  options: []
                };

                if (inputType === "select") {
                  base.options = collectSelectOptions(el);
                }

                if (inputType === "radio") {
                  const optionLabel = normalize(el.getAttribute("aria-label") || "");
                  const listItem = el.closest('[role="listitem"]');
                  const listItemHeading = normalize(
                    listItem
                      ? ((listItem.querySelector('[role="heading"], h1, h2, h3, h4, h5, h6') || {}).textContent || "")
                      : ""
                  );
                  const groupedBy = normalize(
                    el.getAttribute("name") ||
                    el.getAttribute("data-name") ||
                    listItemHeading ||
                    textFromIds(el.getAttribute("aria-labelledby")) ||
                    textFromIds((el.closest('[role="radiogroup"], [role="group"]') || {}).getAttribute?.("aria-labelledby") || "") ||
                    label ||
                    nearText
                  );
                  base.group_key = groupedBy ? `radio:${groupedBy}` : `radio:${agentId}`;
                  base.options = [{
                    text: optionLabel || nearText || base.value || label || "option",
                    value: normalize(el.getAttribute("value") || el.getAttribute("data-value") || ""),
                    agent_id: agentId
                  }];
                }

                if (inputType === "checkbox") {
                  const optionLabel = normalize(el.getAttribute("aria-label") || "");
                  base.options = [{
                    text: optionLabel || nearText || base.value || label || "checkbox",
                    value: normalize(el.getAttribute("value") || el.getAttribute("data-value") || "true"),
                    agent_id: agentId
                  }];
                }
                fields.push(base);
              }
              return fields;
            }
            """
        )

        return self._normalize_fields(raw_fields)

    def _normalize_fields(self, raw_fields: List[Dict[str, Any]]) -> List[FieldMetadata]:
        radio_groups: Dict[str, FieldMetadata] = {}
        normalized: List[FieldMetadata] = []

        for item in raw_fields:
            options = [
                OptionMetadata(
                    text=str(opt.get("text", "")).strip(),
                    value=str(opt.get("value", "")).strip(),
                    agent_id=str(opt.get("agent_id", "")).strip(),
                )
                for opt in item.get("options", [])
            ]
            field = FieldMetadata(
                agent_id=str(item.get("agent_id", "")),
                tag=str(item.get("tag", "")),
                field_type=str(item.get("field_type", "")),
                role=str(item.get("role", "")),
                name=str(item.get("name", "")),
                element_id=str(item.get("element_id", "")),
                label=str(item.get("label", "")),
                placeholder=str(item.get("placeholder", "")),
                near_text=str(item.get("near_text", "")),
                required=bool(item.get("required", False)),
                disabled=bool(item.get("disabled", False)),
                value=str(item.get("value", "")),
                checked=bool(item.get("checked", False)),
                group_key=str(item.get("group_key", "")),
                options=options,
            )

            if field.kind == "radio":
                key = field.group_key or field.name or field.agent_id
                existing = radio_groups.get(key)
                if existing is None:
                    field.options = list(field.options)
                    radio_groups[key] = field
                    normalized.append(field)
                    continue
                existing.options.extend(field.options)
                if not existing.label and field.label:
                    existing.label = field.label
                if not existing.near_text and field.near_text:
                    existing.near_text = field.near_text
                if field.checked:
                    existing.checked = True
                continue

            normalized.append(field)

        deduped: List[FieldMetadata] = []
        seen_ids = set()
        for field in normalized:
            if field.agent_id in seen_ids and field.kind != "radio":
                continue
            seen_ids.add(field.agent_id)
            deduped.append(field)
        return deduped

    def _locator_for(self, field: FieldMetadata):
        if not self.page:
            raise RuntimeError("Browser is not started")
        return self.page.locator(f'[data-ai-agent-id="{field.agent_id}"]')

    @staticmethod
    def _to_bool(value: str) -> bool:
        true_values = {"true", "1", "yes", "y", "check", "checked", "on"}
        return str(value).strip().lower() in true_values

    def apply_action(self, field: FieldMetadata, action: str, value: str) -> bool:
        if not self.page:
            raise RuntimeError("Browser is not started")

        for attempt in range(1, self.config.retry_attempts + 2):
            try:
                ok = self._apply_action_once(field, action, value)
                if ok:
                    self._random_delay()
                    return True
            except (TimeoutError, Error) as exc:
                self.logger.warning(
                    "Action failed on attempt %s for field %s: %s",
                    attempt,
                    field.question_text,
                    exc,
                )
            self.capture_screenshot(f"action_retry_{field.agent_id}_{attempt}")
            self._random_delay()
        return False

    def _apply_action_once(self, field: FieldMetadata, action: str, value: str) -> bool:
        action = (action or "").strip().lower()
        if action not in {"fill", "select"}:
            action = "fill"

        kind = field.kind
        locator = self._locator_for(field)
        if locator.count() == 0 and kind != "radio":
            return False

        if kind in {"text", "email", "tel", "url", "search", "number", "password"}:
            if action == "fill":
                return self._fill_text_field(locator, value)
            return False

        if kind == "textarea":
            return self._fill_text_field(locator, value)

        if kind == "select":
            return self._select_dropdown(locator, field, value)

        if kind == "radio":
            return self._select_radio(field, value)

        if kind == "checkbox":
            return self._toggle_checkbox(locator, action, value)

        if action == "fill":
            return self._fill_text_field(locator, value)
        return False

    def _fill_text_field(self, locator, value: str) -> bool:
        target = locator.first
        text = value or ""
        try:
            target.fill(text)
            return True
        except Error:
            # Many modern apps use contenteditable/ARIA textboxes where .fill may fail.
            target.click()
            target.press("Control+A")
            target.type(text, delay=18)
            return True

    def _select_dropdown(self, locator, field: FieldMetadata, value: str) -> bool:
        value_clean = (value or "").strip()
        if not value_clean and field.options:
            value_clean = field.options[0].value or field.options[0].text

        if field.tag != "select":
            return self._select_custom_dropdown(locator, field, value_clean)

        lowered = value_clean.lower()
        for opt in field.options:
            if lowered == opt.value.lower() and opt.value:
                locator.first.select_option(value=opt.value)
                return True
            if lowered == opt.text.lower() and opt.value:
                locator.first.select_option(value=opt.value)
                return True

        locator.first.select_option(label=value_clean)
        return True

    def _select_custom_dropdown(self, locator, field: FieldMetadata, value: str) -> bool:
        if not self.page:
            return False
        locator.first.click()
        self.page.wait_for_timeout(280)

        target = (value or "").strip().lower()
        chosen = None
        for opt in field.options:
            if target and (target == opt.text.lower() or target == opt.value.lower()):
                chosen = opt
                break

        if chosen and chosen.agent_id:
            choice_locator = self.page.locator(f'[data-ai-agent-id="{chosen.agent_id}"]')
            if choice_locator.count() > 0:
                choice_locator.first.click()
                return True

        option_locator = self.page.locator('[role="option"]')
        count = option_locator.count()
        for idx in range(count):
            candidate = option_locator.nth(idx)
            text = (candidate.inner_text() or "").strip().lower()
            data_value = (candidate.get_attribute("data-value") or "").strip().lower()
            if target and (target == text or target == data_value):
                candidate.click()
                return True

        if count > 0:
            option_locator.first.click()
            return True
        return False

    def _select_radio(self, field: FieldMetadata, value: str) -> bool:
        if not self.page:
            return False
        if not field.options:
            return False

        target = (value or "").strip().lower()
        chosen = None
        for opt in field.options:
            if target and (target == opt.text.lower() or target == opt.value.lower()):
                chosen = opt
                break
        if chosen is None:
            chosen = field.options[0]

        if not chosen.agent_id:
            return False
        locator = self.page.locator(f'[data-ai-agent-id="{chosen.agent_id}"]')
        if locator.count() == 0:
            return False
        return self._set_choice_state(locator.first, desired=True, kind="radio")

    def _toggle_checkbox(self, locator, action: str, value: str) -> bool:
        desired = True
        if action == "fill":
            desired = self._to_bool(value)
        elif action == "select":
            desired = True
        return self._set_choice_state(locator.first, desired=desired, kind="checkbox")

    def _set_choice_state(self, element_locator, desired: bool, kind: str) -> bool:
        tag = (
            element_locator.evaluate("el => (el.tagName || '').toLowerCase()")
            or ""
        ).strip().lower()
        input_type = (
            element_locator.evaluate(
                "el => ((el.tagName || '').toLowerCase() === 'input' ? (el.getAttribute('type') || '') : '')"
            )
            or ""
        ).strip().lower()

        if tag == "input" and input_type in {"checkbox", "radio"}:
            if kind == "radio":
                element_locator.check()
                return True
            if desired:
                element_locator.check()
            else:
                element_locator.uncheck()
            return True

        current = (
            element_locator.get_attribute("aria-checked")
            or ""
        ).strip().lower()
        if kind == "radio":
            if current != "true":
                self._click_with_fallback(element_locator)
            return True

        if desired and current != "true":
            self._click_with_fallback(element_locator)
        if not desired and current == "true":
            self._click_with_fallback(element_locator)
        if current not in {"true", "false"}:
            # Fallback for toggle controls without aria-checked.
            self._click_with_fallback(element_locator)
            if not desired:
                self._click_with_fallback(element_locator)
        return True

    def _click_with_fallback(self, element_locator) -> None:
        try:
            element_locator.click()
            return
        except Error:
            pass
        try:
            element_locator.click(force=True)
            return
        except Error:
            pass
        element_locator.evaluate("el => el.click()")

    def click_submit(self) -> bool:
        if not self.page:
            return False
        selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button:has-text("Submit")',
            'button:has-text("Apply")',
            'button:has-text("Send")',
        ]
        for selector in selectors:
            locator = self.page.locator(selector)
            if locator.count() > 0:
                try:
                    locator.first.click()
                    self.logger.info("Clicked submit button with selector: %s", selector)
                    self._random_delay()
                    return True
                except Error:
                    continue
        return False

    def click_next_page(self) -> bool:
        if not self.page:
            return False
        selectors = [
            'button:has-text("Next")',
            'button:has-text("Continue")',
            'a:has-text("Next")',
            'input[value="Next"]',
            'button[aria-label*="Next" i]',
        ]
        before_url = self.page.url
        for selector in selectors:
            locator = self.page.locator(selector)
            if locator.count() == 0:
                continue
            try:
                locator.first.click()
                self.page.wait_for_timeout(1200)
                if self.page.url != before_url:
                    self.logger.info("Moved to next page with selector: %s", selector)
                else:
                    self.logger.info("Clicked next/continue button: %s", selector)
                self._random_delay()
                return True
            except Error:
                continue
        return False

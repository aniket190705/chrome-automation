from __future__ import annotations

"""CLI entrypoint for the hybrid AI browser automation agent."""

import argparse
import os
import sys
from pathlib import Path

from agent import FormAutomationAgent
from config import AIConfig, AgentConfig, BrowserConfig, bool_from_env, load_user_profile


def load_dotenv_file(path: str = ".env", override: bool = True) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid AI Browser Form Automation Agent")
    parser.add_argument("--url", required=True, help="Target webpage URL")
    parser.add_argument(
        "--user-data",
        default="user_data.json",
        help="Path to JSON file with user profile data",
    )
    parser.add_argument(
        "--browser-channel",
        default=os.getenv("BROWSER_CHANNEL", "chromium"),
        choices=["chromium", "chrome", "msedge"],
        help="Browser channel to use",
    )
    parser.add_argument(
        "--connect-cdp-url",
        default=os.getenv("CONNECT_CDP_URL", ""),
        help="Attach to an existing browser via CDP (e.g. http://127.0.0.1:9222)",
    )
    parser.add_argument(
        "--chrome-user-data-dir",
        default=os.getenv("CHROME_USER_DATA_DIR", ""),
        help="Use an existing Chrome user data directory (for signed-in profile reuse)",
    )
    parser.add_argument(
        "--chrome-profile-directory",
        default=os.getenv("CHROME_PROFILE_DIRECTORY", "Default"),
        help="Chrome profile directory name inside user-data-dir (Default, Profile 1, ...)",
    )
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument(
        "--profile-dir",
        default=os.getenv("BROWSER_PROFILE_DIR", ""),
        help="Persistent browser profile directory (recommended for login-required forms)",
    )
    parser.add_argument(
        "--enable-fallback",
        action="store_true",
        help="Enable external API fallback (OpenAI/Gemini)",
    )
    parser.add_argument(
        "--fallback-provider",
        default=os.getenv("FALLBACK_PROVIDER", "gemini"),
        choices=["openai", "gemini"],
        help="Fallback provider when local AI is uncertain",
    )
    parser.add_argument(
        "--fallback-model",
        default=os.getenv("FALLBACK_MODEL", "gemini-2.5-flash"),
        help="Fallback model name",
    )
    parser.add_argument(
        "--local-model",
        default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        help="Local Ollama model name",
    )
    parser.add_argument(
        "--local-endpoint",
        default=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat"),
        help="Ollama chat API endpoint",
    )
    parser.add_argument(
        "--local-timeout-seconds",
        type=int,
        default=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        help="Ollama response timeout in seconds",
    )
    parser.add_argument(
        "--fallback-timeout-seconds",
        type=int,
        default=int(os.getenv("FALLBACK_TIMEOUT_SECONDS", "90")),
        help="Fallback API timeout in seconds",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.getenv("MIN_CONFIDENCE", "0.6")),
        help="Minimum acceptable confidence from local AI before fallback",
    )
    parser.add_argument(
        "--auto-submit",
        action="store_true",
        help="Click submit after filling detected fields",
    )
    parser.add_argument(
        "--auth-wait-seconds",
        type=int,
        default=int(os.getenv("AUTH_WAIT_SECONDS", "180")),
        help="How long to wait for manual sign-in when a login gate is detected",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def build_config(args: argparse.Namespace) -> AgentConfig:
    profile = load_user_profile(args.user_data)
    gemini_key_available = bool(os.getenv("GEMINI_API_KEY", "").strip())
    ai = AIConfig(
        local_model=args.local_model,
        local_endpoint=args.local_endpoint,
        local_timeout_seconds=args.local_timeout_seconds,
        min_confidence=args.min_confidence,
        enable_fallback=(
            args.enable_fallback
            or bool_from_env("ENABLE_FALLBACK", False)
            or gemini_key_available
        ),
        fallback_provider=args.fallback_provider,
        fallback_model=args.fallback_model,
        fallback_timeout_seconds=args.fallback_timeout_seconds,
    )
    browser = BrowserConfig(
        browser_channel=args.browser_channel,
        connect_cdp_url=args.connect_cdp_url,
        chrome_user_data_dir=args.chrome_user_data_dir,
        chrome_profile_directory=args.chrome_profile_directory,
        headless=args.headless,
        profile_dir=args.profile_dir,
        auth_wait_seconds=args.auth_wait_seconds,
        auto_submit=args.auto_submit,
        multi_page_enabled=bool_from_env("MULTI_PAGE_ENABLED", True),
        max_pages=int(os.getenv("MAX_PAGES", "5")),
        take_debug_screenshots=bool_from_env("TAKE_DEBUG_SCREENSHOTS", True),
    )
    return AgentConfig(
        target_url=args.url,
        user_data_path=args.user_data,
        log_level=args.log_level,
        ai=ai,
        browser=browser,
        user_profile=profile,
    )


def main() -> int:
    load_dotenv_file()
    parser = build_arg_parser()
    args = parser.parse_args()

    config = build_config(args)
    agent = FormAutomationAgent(config)
    try:
        agent.run()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Agent failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

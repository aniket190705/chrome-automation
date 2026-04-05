# Hybrid AI Browser Automation Agent

Production-ready form automation agent with a 3-layer decision architecture:

1. Rule-based autofill (name/email/phone/etc.)
2. API reasoning via Gemini/OpenAI (primary AI path)
3. Local LLM reasoning via Ollama (fallback)

## File Structure

- `main.py` - CLI entrypoint
- `agent.py` - core orchestration loop
- `ai_engine.py` - rule/local/fallback AI decision logic
- `browser.py` - Playwright extraction + actions
- `config.py` - dataclass configs and user profile loader
- `requirements.txt` - Python dependencies
- `user_data.example.json` - sample user profile

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m playwright install chromium
```

## Ollama

```bash
ollama pull llama3.1:8b
ollama serve
```

## Run

```bash
python main.py --url "https://example.com/form" --user-data "user_data.json"
```

## .env Support

The app now loads environment variables from a local `.env` file before startup.
Values in `.env` override the existing OS environment for this project run.

Example `.env`:

```bash
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
OLLAMA_MODEL=llama3.1:8b
OLLAMA_ENDPOINT=http://localhost:11434/api/chat
FALLBACK_PROVIDER=gemini
FALLBACK_MODEL=gemini-2.5-flash
```

Useful options:

```bash
python main.py --url "https://example.com/form" \
  --user-data "user_data.json" \
  --local-model "llama3.1:8b" \
  --local-timeout-seconds 120 \
  --min-confidence 0.65 \
  --enable-fallback \
  --fallback-provider gemini \
  --fallback-model gemini-2.5-flash \
  --fallback-timeout-seconds 90 \
  --auto-submit
```

## Login-Required Forms (Google, etc.)

Use a persistent browser profile so login can be reused:

```bash
python main.py --url "https://example.com/form" \
  --user-data "user_data.json" \
  --profile-dir ".playwright_profile" \
  --auth-wait-seconds 240
```

Tips:

- First run should be headed (do not pass `--headless`) so you can complete sign-in.
- After login is saved in `.playwright_profile`, future runs can be headless.

Use your installed Chrome profile directly (if already signed in):

```bash
python main.py --url "https://example.com/form" \
  --user-data "user_data.json" \
  --browser-channel chrome \
  --chrome-user-data-dir "C:\\Users\\<you>\\AppData\\Local\\Google\\Chrome\\User Data" \
  --chrome-profile-directory "Default"
```

Important:

- Close all regular Chrome windows first when using `--chrome-user-data-dir` to avoid profile lock issues.

## Environment Variables (Optional)

These can be placed in `.env` or exported in the OS environment:

- `OLLAMA_MODEL`
- `OLLAMA_ENDPOINT`
- `MIN_CONFIDENCE`
- `OLLAMA_TIMEOUT_SECONDS`
- `ENABLE_FALLBACK`
- `FALLBACK_PROVIDER`
- `FALLBACK_MODEL`
- `FALLBACK_TIMEOUT_SECONDS`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `MULTI_PAGE_ENABLED`
- `MAX_PAGES`
- `TAKE_DEBUG_SCREENSHOTS`

## Notes

- Debug screenshots are stored in `debug_screenshots/`.
- Run logs are written to `agent.log`.
- The agent avoids hardcoded selectors and uses dynamic element metadata.
# chrome-automation

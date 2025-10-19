# LightReAct

LightReAct is a small ReAct-style Python agent that demonstrates using an LLM together with a small toolset (web search + a persistent Python executor) to answer user questions, run code, and produce structured outputs (including plots).

This repository is intended as a compact research / demo project and a starting point for building tool-enabled agents.

## Features
- ReAct-style agent loop with explicit Thought/Action format
- Tools included:
  - `duckduckgo_search(query: str)` — lightweight HTML DuckDuckGo search scraper for quick lookups
  - `python_executor(code: str)` — persistent Python execution environment (exec/eval) that captures stdout
- Simple CLI and mock LLM responses when no API key is configured

## Quickstart
See `QUICKSTART.md` for a full step-by-step guide. Minimal steps:

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the example env file and set your API keys (if you want to use a real LLM):

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY or other provider keys
```

3. Run the agent (example):

```bash
python react_agent.py
```

Or run the interactive CLI:

```bash
python cli.py
```

## Usage

- The agent expects LLM responses to follow an exact format (a Thought and an Action). The README and prompt templates in `react_agent.py` include examples.
- Tools must be invoked using function-style calls, e.g.:

```
Action: python_executor("""
print('Hello from the executor')
""")

Action: duckduckgo_search("Tesla stock price today")

Action: finish("""Your final answer in Markdown""")
```

## Development notes
- The `python_executor` runs code with `exec` using a persistent `PYTHON_EXEC_ENV` dictionary. This is convenient for demos but insecure for untrusted code — do not expose this to untrusted users without sandboxing.
- The DuckDuckGo scraper is a convenience helper and may break if DuckDuckGo changes its HTML structure.

## Contributing
Contributions welcome. A suggested workflow:

1. Fork the repository
2. Create a topic branch: `git checkout -b feat/your-feature`
3. Add tests where appropriate and run them locally
4. Open a pull request describing the change

Please avoid committing secrets (API keys). Use `.env` for secrets and keep `.env` listed in `.gitignore`.

## License
This project is provided under the MIT License — see `LICENSE` for details (or add a `LICENSE` file if you prefer a different license).

---
If you'd like, I can add a polished `CONTRIBUTING.md`, `LICENSE` (MIT), and a GitHub Actions CI workflow next and push them as a follow-up commit.

## Contact

For questions or collaboration, contact Haoyi Xiong:

- Email: <haoyi.xiong.fr@ieee.org>
- Personal webpage: https://sites.google.com/site/haoyixiongshomepage/Home


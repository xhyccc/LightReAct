# LightReAct

LightReAct is a small ReAct-style Python agent that demonstrates using an LLM together with a small toolset (web search + a persistent Python executor) to answer user questions, run code, and produce structured outputs (including plots).

This repository is intended as a compact research / demo project and a starting point for building tool-enabled agents.

## Features
- **ReAct mode** — fast, interactive Thought/Action/Observation loop for single-session Q&A
- **Plan-and-Execute mode** — structured deep-research pipeline:
  1. *Plan*: LLM decomposes the question into concrete numbered sub-tasks
  2. *Execute*: each sub-task runs through a focused ReAct loop with accumulated context
  3. *Synthesize*: all results are merged into a comprehensive Markdown answer
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

### ReAct mode (default)

The agent iterates through Thought → Action → Observation steps until it calls `finish()`.

```
Action: python_executor("""
print('Hello from the executor')
""")

Action: duckduckgo_search("Tesla stock price today")

Action: finish("""Your final answer in Markdown""")
```

### Plan-and-Execute mode (deep research)

Activate plan-and-execute mode in the CLI by appending `\ plan` to your question:

```
> Research the impact of AI on healthcare \ plan
> Compare OpenAI and Google Gemini capabilities \ plan \ 10
```

Syntax summary:

| CLI input | Mode | Steps |
|---|---|---|
| `question` | ReAct | 30 (default) |
| `question \ 50` | ReAct | 50 |
| `question \ plan` | Plan-and-Execute | 15 steps/task (default) |
| `question \ plan \ 10` | Plan-and-Execute | 10 steps/task |

You can also use `PlanAndExecuteAgent` programmatically:

```python
from react_agent import PlanAndExecuteAgent, duckduckgo_search, python_executor

agent = PlanAndExecuteAgent(tools=[duckduckgo_search, python_executor])
answer = agent.run(
    question="What are the latest breakthroughs in fusion energy?",
    max_plan_steps=6,       # up to 6 sub-tasks in the plan
    max_steps_per_task=15,  # up to 15 ReAct steps per sub-task
)
print(answer)
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


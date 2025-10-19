# Quick Start Guide

## 1. First Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy the example .env file
cp .env.example .env
```

## 2. Configure Your LLM Provider

Edit the `.env` file and set your preferred provider and API key:

### For OpenAI:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4
```

### For Google Gemini:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
GEMINI_MODEL=gemini-pro
```

### For SiliconFlow:
```env
LLM_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your-siliconflow-key-here
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## 3. Get Your API Keys

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and paste into `.env`

### Google Gemini
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy and paste into `.env`

### SiliconFlow
1. Go to https://siliconflow.cn
2. Sign up and get your API key
3. Copy and paste into `.env`

## 4. Run the Agent

```bash
python react_agent.py
```

## 5. Customize Your Queries

Edit the bottom of `react_agent.py` to change the questions:

```python
if __name__ == "__main__":
    tools = [duckduckgo_search, python_executor]
    agent = ReActAgent(tools=tools)

    # Add your own questions here
    question = "Your question here"
    agent.run(question=question)
```

## Tips

- Use `LLM_TEMPERATURE=0.0` for more deterministic outputs (better for tool usage)
- Increase `LLM_MAX_TOKENS` if responses are getting cut off
- The agent will use mock responses if no API key is configured (for testing)

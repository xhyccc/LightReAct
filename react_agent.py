import ast
import os
import re
import requests
import sys
from io import StringIO
from typing import Dict, List, Tuple
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = os.environ.get("OPENAI_API_ENDPOINT", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")

# Google Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_ENDPOINT = os.environ.get("GEMINI_API_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro")

# SiliconFlow Configuration
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
SILICONFLOW_API_ENDPOINT = os.environ.get("SILICONFLOW_API_ENDPOINT", "https://api.siliconflow.cn/v1/chat/completions")
SILICONFLOW_MODEL = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# General LLM Settings
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "15000"))

# Global execution environment for python_executor
# This allows variables and imports to persist across executions
PYTHON_EXEC_ENV = {
    '__builtins__': __builtins__,
}

# --- Tool Definitions ---

def duckduckgo_search(query: str) -> str:
    """
    Searches DuckDuckGo (HTML version) and returns the top 5 results.
    This mimics a browser to avoid the need for an API key.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    params = {'q': query}
    
    try:
        response = requests.get("https://html.duckduckgo.com/html/", headers=headers, params=params)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='result')
        
        if not results:
            return "No good search results found."
            
        snippets = []
        for i, result in enumerate(results[:5]): # Get top 5 results
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')
            
            title = title_tag.get_text(strip=True) if title_tag else "No Title"
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No Snippet"
            
            snippets.append(f"{i+1}. Title: {title} | Snippet: {snippet}")
            
        return "\n".join(snippets)
        
    except requests.exceptions.RequestException as e:
        return f"Error performing search: {e}"

def python_executor(code: str) -> str:
    """
    Executes Python code and returns the output from stdout.
    Uses a persistent global environment so imports and variables persist across executions.
    
    SECURITY WARNING: This function uses `exec` which is highly insecure if exposed
    to untrusted input. In a production environment, this MUST be replaced with a
    properly sandboxed execution environment (e.g., a Docker container).
    """
    # Redirect stdout to capture the output of `print` statements
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # Execute in the global environment (imports and variables persist)
        exec(code, PYTHON_EXEC_ENV)
        output = redirected_output.getvalue()
        # If there is no printed output, it might be an expression. Try eval.
        if not output:
             # This is a fallback for simple one-line expressions, e.g. "1 + 1"
            try:
                output = str(eval(code, PYTHON_EXEC_ENV))
            except Exception:
                output = "" # No output if eval also fails
    except Exception as e:
        import traceback
        # Get full traceback for better debugging
        error_details = traceback.format_exc()
        output = f"Error executing code:\n{error_details}"
    finally:
        sys.stdout = old_stdout

    return output or "Code executed with no output."


# --- LLM Interaction ---

def query_llm(prompt: str) -> str:
    """
    Sends a prompt to the configured LLM provider (OpenAI, Gemini, or SiliconFlow)
    and gets a response.
    """
    if LLM_PROVIDER == "openai":
        return _query_openai(prompt)
    elif LLM_PROVIDER == "gemini":
        return _query_gemini(prompt)
    elif LLM_PROVIDER == "siliconflow":
        return _query_siliconflow(prompt)
    else:
        return f"Error: Unknown LLM provider '{LLM_PROVIDER}'. Supported providers: openai, gemini, siliconflow"


def _query_openai(prompt: str) -> str:
    """
    Query OpenAI API (or compatible endpoints like Azure OpenAI).
    """
    if not OPENAI_API_KEY:
        return _mock_llm_response(prompt)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE
    }

    try:
        response = requests.post(OPENAI_API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with OpenAI: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing OpenAI response: {e} - Response: {response.text}"


def _query_gemini(prompt: str) -> str:
    """
    Query Google Gemini API.
    """
    if not GEMINI_API_KEY:
        return _mock_llm_response(prompt)
    
    # Gemini uses a different URL structure
    url = f"{GEMINI_API_ENDPOINT}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": LLM_TEMPERATURE,
            "maxOutputTokens": LLM_MAX_TOKENS,
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Gemini: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing Gemini response: {e} - Response: {response.text}"


def _query_siliconflow(prompt: str) -> str:
    """
    Query SiliconFlow API (OpenAI-compatible).
    """
    if not SILICONFLOW_API_KEY:
        return _mock_llm_response(prompt)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }
    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE
    }

    try:
        response = requests.post(SILICONFLOW_API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with SiliconFlow: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing SiliconFlow response: {e} - Response: {response.text}"


def _mock_llm_response(prompt: str) -> str:
    """Provides mock responses for demonstration when no API key is configured."""
    print("\n--- MOCK LLM RESPONSE (No API key configured) ---")

    lower_prompt = prompt.lower()
    if "what is the capital of france" in lower_prompt:
        if "thought:" in prompt:
            return (
                "Thought: I have the answer, so I will finish.\n"
                    "Action: finish(\"\"\"The capital of France is Paris.\"\"\")"
            )
        return (
            "Thought: The user is asking for the capital of France. This is a simple fact I should know.\n"
                "Action: finish(\"\"\"The capital of France is Paris.\"\"\")"
        )

    if "market cap" in lower_prompt:
        if "2450" in lower_prompt or "2.45" in lower_prompt:
            return (
                "Thought: I have the stock price ($980) and the number of shares (2.5 billion). "
                "I need to calculate the market cap, which is price * shares. I will use the python_executor.\n"
                "Action: python_executor(\"\"\"\n"
                "price = 980\n"
                "shares = 2.5e9\n"
                "market_cap = price * shares\n"
                "print(f\"Market Cap: ${market_cap:,.0f}\")\n"
                "\"\"\")"
            )
        return (
              "Thought: The user wants to know NVIDIA's market cap. I first need to find the current stock price."
              "\nAction: duckduckgo_search(\"NVIDIA current stock price\")"
        )

    return (
        "Thought: Unable to provide a mock response for this query."
        "\nAction: finish(\"\"\"Error: No API key configured and no mock response available for this query.\"\"\")"
    )


# --- ReAct Agent Logic ---

class ReActAgent:
    def __init__(self, tools: List[callable]):
        self.tools = {tool.__name__: tool for tool in tools}
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> str:
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        current_day = datetime.now().strftime("%A")
        
        prompt = f"""
You are a helpful assistant that can answer questions by using a set of tools.

Current date and time: {current_time} ({current_day})

Here are the tools available to you:
- duckduckgo_search(\"query\"): Use this to search the web for current information. The search query must be a Python string (in double quotes).
- python_executor(\"\"\"code\"\"\"): Use this to execute Python code for calculations, data analysis, and visualizations.
  * All common data science libraries are available: numpy, pandas, scipy, sklearn, matplotlib, seaborn, yfinance, etc.
  * Imports and variables PERSIST across multiple executions in the same session
  * The code should print the result to stdout
  * If there's an error, you'll see the full error message with traceback to help debug
  * Don't give up if code fails - read the error, fix it, and try again!
  * IMPORTANT: Save all visualizations and generated files to the current working directory (e.g., plt.savefig('plot.png'), df.to_csv('results.csv'))
  
  COMMON PITFALLS TO AVOID:
  * Don't format pandas Series/DataFrames directly in f-strings - extract scalar values first
    ❌ Wrong: print(f"Value: {{{{series:,.2f}}}}")
    ✅ Right: print(f"Value: {{{{series.iloc[0]:,.2f}}}}") or print(f"Value: {{{{float(series):,.2f}}}}")
  * Convert numpy/pandas types to Python types for formatting: float(), int(), str()
  * For plotting, save to current directory: plt.savefig('plot.png') then print("Plot saved to plot.png")
  * For data files, save to current directory: df.to_csv('output.csv') then print("Data saved to output.csv")
- finish(\"\"\"...\"\"\"): Use this when you have the final answer, formatted in Markdown.
  
    IMPORTANT: When using python_executor, you MUST wrap your Python code inside triple quotes like python_executor(\"\"\"
    import numpy as np
    result = np.sum([1, 2, 3])
    print(result)
    \"\"\")
- finish(\"\"\"...\"\"\"): Use this when you have the final answer, formatted in Markdown.

CRITICAL RESPONSE FORMAT:
You MUST respond with BOTH a Thought and an Action in EVERY turn. Your response must follow this EXACT format:

Thought: [text about your thought - what to do next - keep it brief, one or two sentences.]
Action: [Invoke the appropriate tool, e.g., duckduckgo_search(\"query\"), python_executor(\"\"\"code\"\"\"), or finish(\"\"\"markdown answer\"\"\")]

EXAMPLES:
Example 1 - Using python executor:
Thought: I need to calculate the market cap by multiplying price by shares...
Action: python_executor(\"\"\"
market_cap = 980 * 2.5e9
print(f"Market cap: ${{{{market_cap:,.0f}}}}")
\"\"\")

Example 1b - Saving a plot:
Thought: I'll create a visualization and save it to the current directory...
Action: python_executor(\"\"\"
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig('analysis_plot.png')
print("Plot saved to analysis_plot.png")
\"\"\")

Example 2 - Using search:
Thought: I need to find the current stock price for Tesla...
Action: duckduckgo_search("Tesla stock price today")

Example 3 - Finishing:
Thought: I now have all the information needed to answer the question...
Action: finish(\"\"\"I now provide comprehensive results to the user based on all execution results.\"\"\")

IMPORTANT FORMATTING RULES:
1. For python_executor, always wrap code in triple quotes as python_executor(\"\"\"
    # your code here
\"\"\")
2. For duckduckgo_search, pass a plain-text Python string in double quotes
3. For finish, wrap your final answer in triple quotes as finish(\"\"\"your markdown answer\"\"\"), including all relevant details, figures, quantitative analysis results, discussions, and insights
4. NEVER write just a Thought without an Action
5. NEVER write just an Action without a Thought
6. NEVER add extra explanation after the Action line
7. If code fails with an error, READ the error message, DEBUG, and try again with fixed code
8. ALWAYS actually execute code - don't just say you "can't" or explain what "would" happen
9. Use multiple steps if needed - you have up to 30 steps to complete complex tasks

DON'T ASK USER FOR FURTHER CLARIFICATION. ALWAYS TRY TO EXECUTE CODE RATHER THAN EXPLAINING LIMITATIONS.

Here is the question: {{question}}
"""
        return prompt

    def run(self, question: str, max_steps: int = 100) -> str:
        scratchpad = ""
        prompt = self.prompt_template.format(question=question)

        for i in range(max_steps):
            print(f"--- Step {i+1} ---")
            
            # 1. THOUGHT and ACTION
            full_prompt = prompt + scratchpad + "\nThought:"
            print(f"Prompt sent to LLM:\n{full_prompt}\n")
            llm_response = query_llm(full_prompt).strip()
            
            print(f"LLM Response:\nThought: {llm_response}\n")

            thought, action = self._parse_llm_response(llm_response)

            scratchpad += f"{thought}"
            scratchpad += f"\nAction: {action}"
            
            # 2. FINISH or EXECUTE ACTION
            if action.lower().startswith("finish"):
                answer = self._get_tool_input(action, tool_name="finish")
                print(f"Final Answer: {answer}")
                return answer
            
            tool_name, tool_input = self._get_tool_name_and_input(action)
            if tool_name not in self.tools:
                observation = f"Error: Unknown tool '{tool_name}'."
            else:
                tool_function = self.tools[tool_name]
                
                # Show preview of the tool input (truncate if too long)
                if len(tool_input) > 3000:
                    print(f"Executing {tool_name} with input (first 100 chars): {tool_input[:100]}...")
                else:
                    print(f"Executing {tool_name} with input: {tool_input}")
                
                observation = tool_function(tool_input)

            print(f"Observation: {observation}\n")
            
            # 3. OBSERVATION
            scratchpad += f"\nObservation: {observation}"
        
        # Reached max steps without finishing
        error_msg = f"Agent stopped after reaching max steps ({max_steps}). Task incomplete."
        print(error_msg)
        return error_msg

    def _parse_llm_response(self, response: str) -> Tuple[str, str]:
        # Check if response starts with an action directly (e.g., "finish(...)", "python_executor(\"\"\"...\"\"\")")
        stripped = response.strip()
        if stripped.startswith((
            "finish(",
            "python_executor(",
            "duckduckgo_search("
        )):
            print("ℹ️  Note: LLM provided action without 'Action:' prefix. Auto-correcting...")
            # Extract the action and assume no explicit thought
            return "Proceeding with action.", stripped

        thought_action_pattern = re.compile(r"Thought:\s*(.*?)\s*Action:\s*(.+)", re.DOTALL)
        match = thought_action_pattern.search(response)
        if match:
            thought = match.group(1).strip()
            action = match.group(2).strip()
            return thought, action

        if "Action:" not in response:
            # If the model gets stuck and doesn't output "Action:", we treat its whole output as the thought.
            print("⚠️  WARNING: LLM did not provide an Action. Response was:")
            print(f"'{response[:3000]}...' (truncated)" if len(response) > 3000 else f"'{response}'")
            print("This usually means the LLM needs a clearer prompt or different model parameters.")
            print("⚠️  Forcing termination due to malformed response.")
            error_msg = f"ERROR: LLM did not provide a valid Action. Last response: {response[:100]}..."
            return response, f'finish("""{error_msg}""")'

        # Fallback: split from the last occurrence of "Action:" to avoid truncating the action payload
        thought_part, action_part = response.rsplit("Action:", 1)
        return thought_part.replace("Thought:", "", 1).strip(), action_part.strip()

    def _get_tool_input(self, action_string: str, tool_name: str | None = None) -> str:
        """Extract the argument payload from a tool invocation string.

        Supports the current function-style format, for example:
            tool_name("...")
            tool_name('...')
            tool_name(\"\"\"...\"\"\")

        Falls back to legacy bracket syntax (tool_name[...]) for backward compatibility.
        """
        action_string = action_string.strip()
        if not action_string:
            return ""

        selected_tool = (tool_name or "").strip()

        def _extract_from_call(call_node: ast.Call) -> Tuple[str | None, str | None]:
            """Return (tool_name, first_argument_as_string_or_none)."""
            parsed_tool = None
            if isinstance(call_node.func, ast.Name):
                parsed_tool = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                parsed_tool = call_node.func.attr

            if not call_node.args:
                return parsed_tool, None

            arg_node = call_node.args[0]
            try:
                literal_value = ast.literal_eval(arg_node)
                if isinstance(literal_value, str):
                    return parsed_tool, literal_value
            except (ValueError, SyntaxError):
                pass

            segment = ast.get_source_segment(action_string, arg_node)
            if segment is None:
                return parsed_tool, None

            segment = segment.strip()
            if segment.startswith(('"""', "'''")) and segment.endswith(('"""', "'''")) and len(segment) >= 6:
                return parsed_tool, segment[3:-3]
            if segment.startswith(('"', "'")) and segment.endswith(('"', "'")) and len(segment) >= 2:
                return parsed_tool, segment[1:-1]
            return parsed_tool, segment

        # Preferred path: parse the action as a Python expression call
        try:
            parsed_expr = ast.parse(action_string, mode="eval")
        except SyntaxError:
            parsed_expr = None

        if parsed_expr and isinstance(parsed_expr.body, ast.Call):
            parsed_tool_name, arg_value = _extract_from_call(parsed_expr.body)
            effective_tool = selected_tool or (parsed_tool_name or "")
            if arg_value is not None:
                if effective_tool == "duckduckgo_search":
                    return arg_value.strip()
                return arg_value

        # Legacy fallback: handle bracket-style invocations
        enclosure_match = re.match(r'^[a-zA-Z_]\w*\s*(\(|\[)', action_string)
        if not enclosure_match:
            return ""

        open_char = enclosure_match.group(1)
        close_char = ")" if open_char == "(" else "]"
        start_idx = action_string.find(open_char)

        bracket_count = 0
        end_idx = -1
        for i in range(start_idx, len(action_string)):
            char = action_string[i]
            if char == open_char:
                bracket_count += 1
            elif char == close_char:
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i
                    break

        raw_input = action_string[start_idx + 1:end_idx].strip() if end_idx != -1 else action_string[start_idx + 1:].strip()

        if selected_tool == "duckduckgo_search":
            return raw_input.strip('"\'')

        if selected_tool == "python_executor":
            if raw_input.startswith('"""') and raw_input.endswith('"""'):
                return raw_input[3:-3]
            if raw_input.startswith("'''") and raw_input.endswith("'''"):
                return raw_input[3:-3]

        if selected_tool == "finish":
            if raw_input.startswith('"""') and raw_input.endswith('"""'):
                return raw_input[3:-3]
            if raw_input.startswith("'''") and raw_input.endswith("'''"):
                return raw_input[3:-3]

        code_block_match = re.search(r'```python\s*(.*?)\s*```', raw_input, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        code_block_match = re.search(r'```\s*(.*?)\s*```', raw_input, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        return raw_input

    def _get_tool_name_and_input(self, action_string: str) -> Tuple[str, str]:
        action_string = action_string.strip()
        if not action_string:
            return "", ""

        tool_match = re.match(r'^([a-zA-Z_]\w*)', action_string)
        tool_name = tool_match.group(1) if tool_match else action_string.split('[', 1)[0].strip()
        tool_input = self._get_tool_input(action_string, tool_name=tool_name)
        return tool_name, tool_input

# --- Main Execution ---

if __name__ == "__main__":
    tools = [duckduckgo_search, python_executor]
    agent = ReActAgent(tools=tools)

    # Example 2: Complex question requiring search and calculation
    question2 = "I am trying to do Chinese EFT trading using the bank apps. Please design a trading strategy that balances risk and return, and explain how to implement it step by step."
    agent.run(question=question2, max_steps=300)


#!/usr/bin/env python3
"""
CLI Interface for LightReAct Agent

Usage:
    python cli.py

Interactive mode:
    - Enter your question directly (uses default max_steps=30, ReAct mode)
    - Enter "question \\ max_steps" to specify custom max steps (ReAct mode)
    - Enter "question \\ plan" to use Plan-and-Execute mode (deep research)
    - Enter "question \\ plan \\ N" to use Plan-and-Execute with N steps per task
    - Type 'quit', 'exit', or 'q' to exit
    - Type 'help' for usage information
    - Type 'clear' to clear the screen

Examples:
    > What is the capital of France?
    > Calculate the market cap of NVIDIA \\ 50
    > Research the impact of AI on healthcare \\ plan
    > Deep dive into quantum computing trends \\ plan \\ 10
"""

import os
import sys
from react_agent import ReActAgent, PlanAndExecuteAgent, duckduckgo_search, python_executor


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                   LightReAct Agent CLI                    ║
║                                                           ║
║  ReAct mode  : fast, interactive question answering       ║
║  Plan mode   : structured deep research (use \\ plan)     ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("Type 'help' for usage information, 'quit' to exit.\n")


def print_help():
    """Print help information."""
    help_text = """
Usage Instructions:
------------------
1. ReAct mode (default) – fast, interactive question answering:
   > What is the weather in New York?
   > Calculate NVIDIA market cap \\ 50        (custom max_steps)

2. Plan-and-Execute mode – structured deep research:
   > Research the impact of AI on healthcare \\ plan
   > Deep dive into quantum computing trends \\ plan \\ 10

   The agent will:
     a) Generate a numbered research plan
     b) Execute each sub-task with up to N steps (default 15)
     c) Synthesize all results into a comprehensive Markdown answer

3. Commands:
   - 'help'  : Show this help message
   - 'clear' : Clear the screen
   - 'quit', 'exit', 'q' : Exit the CLI

Syntax summary:
---------------
  question                         → ReAct, default max_steps (30)
  question \\ max_steps             → ReAct, specified max_steps
  question \\ plan                  → Plan-and-Execute, default steps/task (15)
  question \\ plan \\ N             → Plan-and-Execute, N steps/task

Examples:
---------
> What is the capital of France?
> Calculate the fibonacci sequence up to 100
> Analyze Tesla stock performance this year \\ plan
> Compare OpenAI and Google Gemini capabilities \\ plan \\ 10
"""
    print(help_text)


def parse_input(user_input: str) -> tuple:
    """
    Parse user input to extract question, max_steps, and agent mode.

    Syntax options:
      "question"                  → ReAct mode, default max_steps (30)
      "question \\ max_steps"     → ReAct mode, specified max_steps
      "question \\ plan"          → Plan-and-Execute mode, default steps/task (15)
      "question \\ plan \\ N"     → Plan-and-Execute mode, N steps/task

    Args:
        user_input: Raw user input string

    Returns:
        Tuple of (question, max_steps, mode) where mode is "react" or "plan"
    """
    default_max_steps = 30
    default_steps_per_task = 15
    default_mode = "react"

    if '\\' not in user_input:
        return user_input.strip(), default_max_steps, default_mode

    parts = [p.strip() for p in user_input.split('\\')]
    question = parts[0]

    if not question:
        return user_input.strip(), default_max_steps, default_mode

    # Plan-and-Execute mode: "question \ plan" or "question \ plan \ N"
    if len(parts) >= 2 and parts[1].lower() == 'plan':
        steps_per_task = default_steps_per_task
        if len(parts) >= 3:
            try:
                steps_per_task = int(parts[2])
                if steps_per_task <= 0:
                    print(f"⚠️  Warning: steps per task must be positive. Using default: {default_steps_per_task}")
                    steps_per_task = default_steps_per_task
            except ValueError:
                print(f"⚠️  Warning: Invalid steps value. Using default: {default_steps_per_task}")
        return question, steps_per_task, "plan"

    # ReAct mode with optional max_steps: "question \ max_steps"
    if len(parts) >= 2:
        try:
            max_steps = int(parts[1])
            if max_steps <= 0:
                print(f"⚠️  Warning: max_steps must be positive. Using default: {default_max_steps}")
                max_steps = default_max_steps
            return question, max_steps, default_mode
        except ValueError:
            print(f"⚠️  Warning: Invalid max_steps format. Using default: {default_max_steps}")
            return question, default_max_steps, default_mode

    return question, default_max_steps, default_mode


def run_cli():
    """Main CLI loop."""
    # Initialize both agents
    tools = [duckduckgo_search, python_executor]
    react_agent = ReActAgent(tools=tools)
    plan_agent = PlanAndExecuteAgent(tools=tools)

    print_banner()

    while True:
        try:
            # Get user input
            user_input = input("\n🤔 Your question: ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thank you for using LightReAct Agent.\n")
                sys.exit(0)

            elif user_input.lower() == 'help':
                print_help()
                continue

            elif user_input.lower() == 'clear':
                clear_screen()
                print_banner()
                continue

            # Parse question, steps, and mode
            question, max_steps, mode = parse_input(user_input)

            if not question:
                print("⚠️  Error: Question cannot be empty.")
                continue

            # Display execution info
            print(f"\n{'='*60}")
            print(f"📝 Question: {question}")
            if mode == "plan":
                print(f"🗺️  Mode: Plan-and-Execute (max {max_steps} steps/task)")
            else:
                print(f"⚡ Mode: ReAct (max {max_steps} steps)")
            print(f"{'='*60}\n")

            # Run the selected agent
            try:
                if mode == "plan":
                    answer = plan_agent.run(question=question, max_steps_per_task=max_steps)
                else:
                    answer = react_agent.run(question=question, max_steps=max_steps)

                print(f"\n{'='*60}")
                print("✅ Execution Complete")
                print(f"{'='*60}")

            except KeyboardInterrupt:
                print("\n\n⚠️  Execution interrupted by user.")
                continue
            except Exception as e:
                print(f"\n❌ Error during execution: {e}")
                import traceback
                traceback.print_exc()
                continue

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using LightReAct Agent.\n")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 Goodbye! Thank you for using LightReAct Agent.\n")
            sys.exit(0)


if __name__ == "__main__":
    run_cli()

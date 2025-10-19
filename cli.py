#!/usr/bin/env python3
"""
CLI Interface for LightReAct Agent

Usage:
    python cli.py

Interactive mode:
    - Enter your question directly (uses default max_steps=30)
    - Enter "question \ max_steps" to specify custom max steps
    - Type 'quit', 'exit', or 'q' to exit
    - Type 'help' for usage information
    - Type 'clear' to clear the screen

Examples:
    > What is the capital of France?
    > Calculate the market cap of NVIDIA \ 50
    > Search for latest AI news
"""

import os
import sys
from react_agent import ReActAgent, duckduckgo_search, python_executor


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                   LightReAct Agent CLI                    ║
║                                                           ║
║  A ReAct-based AI agent with search and code execution   ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("Type 'help' for usage information, 'quit' to exit.\n")


def print_help():
    """Print help information."""
    help_text = """
Usage Instructions:
------------------
1. Enter your question directly:
   > What is the weather in New York?
   
2. Specify custom max steps (default is 30):
   > Calculate NVIDIA market cap \\ 50
   
3. Commands:
   - 'help'  : Show this help message
   - 'clear' : Clear the screen
   - 'quit', 'exit', 'q' : Exit the CLI

Examples:
---------
> What is the capital of France?
> Based on historical data, predict stock trends \\ 100
> Search for the latest AI research papers
> Calculate the fibonacci sequence up to 100

Notes:
------
- Default max_steps is 30
- Use backslash (\\) to separate question and max_steps
- The agent has access to web search and Python code execution
- Variables and imports persist across executions in the same session
"""
    print(help_text)


def parse_input(user_input: str) -> tuple:
    """
    Parse user input to extract question and max_steps.
    
    Format: "question \\ max_steps"
    
    Args:
        user_input: Raw user input string
        
    Returns:
        Tuple of (question, max_steps)
    """
    default_max_steps = 30
    
    # Check if user specified max_steps with backslash separator
    if '\\' in user_input:
        parts = user_input.split('\\')
        if len(parts) == 2:
            question = parts[0].strip()
            try:
                max_steps = int(parts[1].strip())
                if max_steps <= 0:
                    print(f"⚠️  Warning: max_steps must be positive. Using default: {default_max_steps}")
                    max_steps = default_max_steps
                return question, max_steps
            except ValueError:
                print(f"⚠️  Warning: Invalid max_steps format. Using default: {default_max_steps}")
                return parts[0].strip(), default_max_steps
    
    # No max_steps specified, use default
    return user_input.strip(), default_max_steps


def run_cli():
    """Main CLI loop."""
    # Initialize the agent
    tools = [duckduckgo_search, python_executor]
    agent = ReActAgent(tools=tools)
    
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
            
            # Parse question and max_steps
            question, max_steps = parse_input(user_input)
            
            if not question:
                print("⚠️  Error: Question cannot be empty.")
                continue
            
            # Display execution info
            print(f"\n{'='*60}")
            print(f"📝 Question: {question}")
            print(f"🔢 Max Steps: {max_steps}")
            print(f"{'='*60}\n")
            
            # Run the agent
            try:
                answer = agent.run(question=question, max_steps=max_steps)
                
                print(f"\n{'='*60}")
                print(f"✅ Execution Complete")
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

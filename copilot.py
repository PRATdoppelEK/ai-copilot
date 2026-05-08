"""
AI Copilot — Main Orchestrator
================================
Author: Prateek Gaur
GitHub: github.com/PRATdoppelEK/ai-copilot

Single entry point for all three copilot agents:
  - Job Hunter     : search jobs, score CV match, generate cover letters, notify
  - Code Assistant : solve, debug, explain, review, generate, convert code
  - Knowledge Engine: ask, deep-dive, compare, summarise, explain, fact-check

Usage:
    python copilot.py                          # Launch interactive CLI
    python copilot.py --mode job               # Run job hunter
    python copilot.py --mode code              # Launch code assistant
    python copilot.py --mode ask "question"    # Quick question
    python copilot.py --mode solve "problem"   # Quick code solve
    python copilot.py --mode explain "topic"   # Quick concept explanation
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level  = logging.WARNING,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader  import load_config
from src.utils.llm_client     import LLMClient
from src.agents.job_hunter    import JobHunterAgent
from src.agents.code_assistant import CodeAssistantAgent
from src.agents.knowledge_engine import KnowledgeEngineAgent


BANNER = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║    █████╗ ██╗     ██████╗ ██████╗ ██████╗ ██╗██╗     ██████╗ ║
  ║   ██╔══██╗██║    ██╔════╝██╔═══██╗██╔══██╗██║██║    ██╔═══██╗║
  ║   ███████║██║    ██║     ██║   ██║██████╔╝██║██║    ██║   ██║║
  ║   ██╔══██║██║    ██║     ██║   ██║██╔═══╝ ██║██║    ██║   ██║║
  ║   ██║  ██║██║    ╚██████╗╚██████╔╝██║     ██║███████╗╚██████╔╝║
  ║   ╚═╝  ╚═╝╚═╝     ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝ ╚═════╝ ║
  ║                                                              ║
  ║   Your personal AI — Job Hunter · Code Assistant · Knowledge ║
  ║   Runs fully offline · Powered by Llama 3 via Ollama         ║
  ╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available commands:
─────────────────────────────────────────────────────────────
  JOBS
    jobs                → Run full job search pipeline
    jobs cover <co> <role> <desc>  → Generate one cover letter

  CODE
    solve  <problem>    → Generate a code solution
    debug  <code> <err> → Debug broken code
    explain <code>      → Explain what code does
    review  <code>      → Code review with suggestions
    generate <spec>     → Generate boilerplate from spec
    convert <code> <lang> → Convert code to another language

  KNOWLEDGE
    ask    <question>   → Answer any question (web-grounded)
    dive   <topic>      → Deep research on a topic
    compare <A> vs <B>  → Compare two options
    explain-concept <x> → Explain any concept clearly
    fact   <claim>      → Fact-check a statement
    sum    <text>       → Summarise long text

  GENERAL
    help                → Show this message
    exit / quit         → Exit the copilot
─────────────────────────────────────────────────────────────
"""


class AICopilot:
    """
    Unified AI Copilot — orchestrates all three agents through a
    single interactive interface or CLI flags.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        print("  Loading configuration...")
        self.config = load_config(config_path)

        print("  Initialising LLM client (Ollama)...")
        self.llm = LLMClient(
            model       = self.config.llm.model,
            temperature = self.config.llm.temperature,
        )

        print("  Loading agents...")
        self.job_agent = JobHunterAgent(
            config     = self.config,
            llm_client = self.llm,
        )
        self.code_agent = CodeAssistantAgent(
            llm_client = self.llm,
            output_dir = self.config.paths.get("code_solutions", "outputs/code_solutions/"),
        )
        self.knowledge_agent = KnowledgeEngineAgent(
            llm_client = self.llm,
            ntfy_topic = self.config.notifications.ntfy_topic,
        )
        print("  ✅ All agents ready.\n")

    # ── Job Hunter ─────────────────────────────────────────────────────────────

    def run_jobs(self):
        """Full autonomous job search pipeline."""
        self.job_agent.run(notify=self.config.notifications.enabled)

    # ── Code Assistant ─────────────────────────────────────────────────────────

    def solve(self, problem: str, language: str = "python"):
        return self.code_agent.solve(problem, language=language)

    def debug(self, code: str, error: str, language: str = ""):
        return self.code_agent.debug(code, error, language=language)

    def explain_code(self, code: str):
        return self.code_agent.explain(code)

    def review_code(self, code: str):
        return self.code_agent.review(code)

    def generate_code(self, spec: str, language: str = "python"):
        return self.code_agent.generate(spec, language=language)

    def convert_code(self, code: str, target_lang: str):
        return self.code_agent.convert(code, target_lang)

    # ── Knowledge Engine ───────────────────────────────────────────────────────

    def ask(self, question: str):
        return self.knowledge_agent.ask(question)

    def deep_dive(self, topic: str):
        return self.knowledge_agent.deep_dive(topic)

    def compare(self, a: str, b: str, context: str = "general use"):
        return self.knowledge_agent.compare(a, b, context)

    def summarise(self, text: str):
        return self.knowledge_agent.summarise(text)

    def explain_concept(self, topic: str):
        return self.knowledge_agent.explain_concept(topic)

    def fact_check(self, claim: str):
        return self.knowledge_agent.fact_check(claim)

    # ── Interactive CLI ────────────────────────────────────────────────────────

    def interactive(self):
        """Launch the interactive terminal interface."""
        print(BANNER)
        print(HELP_TEXT)

        while True:
            try:
                raw = input("Copilot › ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Goodbye. 👋\n")
                break

            if not raw:
                continue

            parts   = raw.split(None, 1)
            command = parts[0].lower()
            arg     = parts[1].strip() if len(parts) > 1 else ""

            try:
                if command in ("exit", "quit", "q"):
                    print("\n  Goodbye. 👋\n")
                    break

                elif command == "help":
                    print(HELP_TEXT)

                elif command == "jobs":
                    self.run_jobs()

                elif command == "solve":
                    lang = "python"
                    if arg.startswith("--lang "):
                        parts2 = arg.split(None, 2)
                        lang, arg = parts2[1], parts2[2] if len(parts2) > 2 else ""
                    self.solve(arg or input("  Problem: "), language=lang)

                elif command == "debug":
                    print("  Paste your broken code (end with '---' on its own line):")
                    code_lines = []
                    while True:
                        line = input()
                        if line.strip() == "---":
                            break
                        code_lines.append(line)
                    error = input("  Error message: ")
                    self.debug("\n".join(code_lines), error)

                elif command == "explain":
                    print("  Paste code to explain (end with '---'):")
                    code_lines = []
                    while True:
                        line = input()
                        if line.strip() == "---":
                            break
                        code_lines.append(line)
                    self.explain_code("\n".join(code_lines))

                elif command == "review":
                    print("  Paste code to review (end with '---'):")
                    code_lines = []
                    while True:
                        line = input()
                        if line.strip() == "---":
                            break
                        code_lines.append(line)
                    self.review_code("\n".join(code_lines))

                elif command == "generate":
                    lang = input("  Language (default: python): ").strip() or "python"
                    self.generate_code(arg or input("  Specification: "), language=lang)

                elif command == "convert":
                    target = input("  Target language: ").strip()
                    print("  Paste code to convert (end with '---'):")
                    code_lines = []
                    while True:
                        line = input()
                        if line.strip() == "---":
                            break
                        code_lines.append(line)
                    self.convert_code("\n".join(code_lines), target)

                elif command == "ask":
                    self.ask(arg or input("  Question: "))

                elif command == "dive":
                    self.deep_dive(arg or input("  Topic: "))

                elif command == "compare":
                    if " vs " in arg:
                        a, b = arg.split(" vs ", 1)
                        ctx = input("  Context (e.g. 'battery BMS applications'): ").strip()
                        self.compare(a.strip(), b.strip(), ctx or "general use")
                    else:
                        a   = input("  Option A: ")
                        b   = input("  Option B: ")
                        ctx = input("  Context: ")
                        self.compare(a, b, ctx)

                elif command == "explain-concept":
                    self.explain_concept(arg or input("  Concept: "))

                elif command == "fact":
                    self.fact_check(arg or input("  Claim to fact-check: "))

                elif command == "sum":
                    if arg:
                        self.summarise(arg)
                    else:
                        print("  Paste text to summarise (end with '---'):")
                        lines = []
                        while True:
                            line = input()
                            if line.strip() == "---":
                                break
                            lines.append(line)
                        self.summarise("\n".join(lines))

                else:
                    # Treat unknown input as a question
                    print(f"  Treating as a question...")
                    self.ask(raw)

            except KeyboardInterrupt:
                print("\n  (interrupted — type 'exit' to quit)\n")
            except Exception as e:
                logger.error(f"Command failed: {e}")
                print(f"\n  ❌ Error: {e}\n")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI Copilot — Job Hunter · Code Assistant · Knowledge Engine"
    )
    p.add_argument("--mode",   default="interactive",
                   choices=["interactive", "job", "code", "ask", "solve", "explain",
                             "dive", "compare", "fact", "debug"],
                   help="Operating mode")
    p.add_argument("--input",  default="", help="Input text for the selected mode")
    p.add_argument("--lang",   default="python", help="Programming language for code modes")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    return p.parse_args()


def main():
    args    = parse_args()
    copilot = AICopilot(config_path=args.config)

    if args.mode == "interactive" or not args.input:
        copilot.interactive()
    elif args.mode == "job":
        copilot.run_jobs()
    elif args.mode == "ask":
        copilot.ask(args.input)
    elif args.mode == "solve":
        copilot.solve(args.input, language=args.lang)
    elif args.mode == "explain":
        copilot.explain_concept(args.input)
    elif args.mode == "dive":
        copilot.deep_dive(args.input)
    elif args.mode == "compare":
        parts = args.input.split(" vs ", 1)
        copilot.compare(parts[0].strip(), parts[1].strip() if len(parts) > 1 else "")
    elif args.mode == "fact":
        copilot.fact_check(args.input)
    elif args.mode == "debug":
        copilot.debug(args.input, input("Error: "))


if __name__ == "__main__":
    main()

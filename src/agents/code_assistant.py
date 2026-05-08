"""
Code Assistant Agent
====================
Solves coding problems, debugs errors, explains code, generates boilerplate,
and saves clean solutions to file — all using a local LLM.

Capabilities:
    - solve(problem)     → generates a working solution with explanation
    - debug(code, error) → diagnoses the bug and returns fixed code
    - explain(code)      → plain-English explanation of what code does
    - review(code)       → code review with specific improvement suggestions
    - generate(spec)     → generates boilerplate from a natural language spec
    - convert(code, target_lang) → translates code between languages
"""

import os
import re
import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    SQL        = "sql"
    BASH       = "bash"
    MATLAB     = "matlab"
    CPP        = "cpp"
    JAVA       = "java"
    AUTO       = "auto"   # Let the LLM detect


@dataclass
class CodeSolution:
    task:        str
    code:        str
    explanation: str
    language:    str
    suggestions: list = field(default_factory=list)
    saved_path:  str  = ""


# ── Prompt Templates ──────────────────────────────────────────────────────────

SOLVE_PROMPT = """
You are an expert software engineer. Solve the following programming problem.

PROBLEM:
{problem}

REQUIREMENTS:
- Language: {language}
- Write clean, production-quality code with meaningful variable names
- Add concise inline comments for non-obvious logic
- Include a brief docstring explaining what the function/module does
- After the code block, write "EXPLANATION:" followed by a plain-English
  explanation of your approach (3–5 sentences max)

Output format:
```{language}
[your code here]
```

EXPLANATION:
[your explanation here]
"""

DEBUG_PROMPT = """
You are an expert software engineer debugging code.

BROKEN CODE:
```
{code}
```

ERROR MESSAGE:
{error}

TASK:
1. Identify the root cause of the error in 1–2 sentences
2. Provide the complete fixed code
3. Explain what you changed and why

Output format:
ROOT CAUSE:
[1-2 sentence diagnosis]

FIXED CODE:
```{language}
[complete fixed code]
```

CHANGES:
[what was changed and why]
"""

EXPLAIN_PROMPT = """
You are a senior engineer explaining code to a colleague.

CODE:
```
{code}
```

Provide a clear explanation covering:
1. What this code does (overall purpose — 2 sentences)
2. How it works (step-by-step logic)
3. Key design decisions or patterns used
4. Any potential edge cases or limitations

Be specific and technical — assume the reader is a developer, not a beginner.
"""

REVIEW_PROMPT = """
You are a senior software engineer conducting a code review.

CODE:
```
{code}
```

Provide a structured review covering:
1. CORRECTNESS — any bugs, logic errors, or incorrect assumptions
2. PERFORMANCE — any inefficiencies or scalability concerns
3. READABILITY — naming, structure, documentation quality
4. BEST PRACTICES — adherence to language conventions and patterns
5. SECURITY — any potential security issues (if applicable)

For each issue found, state: what the problem is, why it matters, and the fix.
End with an overall assessment: Approve / Approve with changes / Request changes.
"""

GENERATE_PROMPT = """
You are an expert software engineer generating boilerplate code.

SPECIFICATION:
{spec}

Generate complete, production-ready {language} code for the above specification.
Include:
- Proper module structure and imports
- Type hints / type annotations where applicable
- Docstrings for all public functions and classes
- Error handling for foreseeable failure cases
- A simple usage example at the bottom (in an if __name__ == "__main__": block for Python)

Output clean, well-commented code ready to be committed to a repository.
"""

CONVERT_PROMPT = """
You are an expert programmer who converts code between languages.

SOURCE CODE ({source_lang}):
```{source_lang}
{code}
```

Convert this to idiomatic {target_lang} code.
Requirements:
- Preserve all logic and functionality exactly
- Use {target_lang} conventions and idioms (not a line-by-line translation)
- Add comments where the conversion involves non-obvious changes
- Include all necessary imports/dependencies

Output the converted code only, in a code block.
"""


# ── Code Extraction Helpers ───────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    """Extract the first code block from an LLM response."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def extract_section(text: str, section_header: str) -> str:
    """Extract a named section from structured LLM output."""
    pattern = rf"{re.escape(section_header)}[:\s]*(.*?)(?=\n[A-Z ]+:|$)"
    match   = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def detect_language(code: str) -> str:
    """Heuristically detect the programming language of a code snippet."""
    code_lower = code.lower()
    if "def " in code and ("import " in code or ":" in code):
        return "python"
    if "function " in code or "const " in code or "let " in code:
        return "javascript"
    if "select " in code_lower and "from " in code_lower:
        return "sql"
    if "#include" in code or "::" in code:
        return "cpp"
    if "public class" in code or "System.out" in code:
        return "java"
    return "python"  # Default


# ── Save to File ──────────────────────────────────────────────────────────────

LANGUAGE_EXTENSIONS = {
    "python": ".py", "javascript": ".js", "typescript": ".ts",
    "sql": ".sql", "bash": ".sh", "matlab": ".m",
    "cpp": ".cpp", "java": ".java",
}

def save_solution(solution: CodeSolution, output_dir: str = "outputs/code_solutions") -> str:
    """Save a code solution to a file."""
    os.makedirs(output_dir, exist_ok=True)
    ext       = LANGUAGE_EXTENSIONS.get(solution.language.lower(), ".txt")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = re.sub(r"[^\w]", "_", solution.task[:40]).strip("_")
    filename  = f"{safe_task}_{timestamp}{ext}"
    filepath  = os.path.join(output_dir, filename)

    header = f"""# Task: {solution.task}
# Language: {solution.language}
# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
# ─────────────────────────────────────────────────────────────────

"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + solution.code)
        if solution.explanation:
            f.write(f"\n\n# ── Explanation ──────────────────────────────────────────────\n")
            f.write("# " + "\n# ".join(solution.explanation.split("\n")))

    return filepath


# ── Code Assistant Agent ──────────────────────────────────────────────────────

class CodeAssistantAgent:
    """
    AI-powered coding assistant.
    All inference runs locally via Ollama — no API key required.
    """

    def __init__(self, llm_client, output_dir: str = "outputs/code_solutions"):
        self.llm        = llm_client
        self.output_dir = output_dir

    def solve(
        self,
        problem:  str,
        language: str = "python",
        save:     bool = True,
    ) -> CodeSolution:
        """
        Solve a coding problem from a natural language description.

        Args:
            problem  : plain English problem description
            language : target programming language
            save     : whether to save the solution to a file

        Returns:
            CodeSolution with code, explanation, and optional file path
        """
        print(f"🧠 Solving: {problem[:60]}...")
        prompt   = SOLVE_PROMPT.format(problem=problem, language=language)
        response = self.llm.invoke(prompt)

        code        = extract_code_block(response)
        explanation = extract_section(response, "EXPLANATION")
        if not explanation:
            # Fallback: take text after the code block
            parts       = response.split("```")
            explanation = parts[-1].strip() if len(parts) > 2 else ""

        solution = CodeSolution(
            task        = problem,
            code        = code,
            explanation = explanation,
            language    = language,
        )

        if save:
            solution.saved_path = save_solution(solution, self.output_dir)
            print(f"  ✅ Solution saved: {solution.saved_path}")

        return solution

    def debug(
        self,
        code:     str,
        error:    str,
        language: str = "",
        save:     bool = True,
    ) -> CodeSolution:
        """
        Debug broken code given the error message.

        Args:
            code     : the broken code snippet
            error    : the full error message or traceback
            language : programming language (auto-detected if empty)

        Returns:
            CodeSolution with fixed code and explanation of changes
        """
        if not language:
            language = detect_language(code)
        print(f"🐛 Debugging {language} code...")

        prompt   = DEBUG_PROMPT.format(code=code, error=error, language=language)
        response = self.llm.invoke(prompt)

        fixed_code  = extract_code_block(response)
        root_cause  = extract_section(response, "ROOT CAUSE")
        changes     = extract_section(response, "CHANGES")
        explanation = f"Root cause: {root_cause}\n\nChanges made: {changes}"

        solution = CodeSolution(
            task        = f"Debug: {error[:60]}",
            code        = fixed_code,
            explanation = explanation,
            language    = language,
        )

        if save:
            solution.saved_path = save_solution(solution, self.output_dir)
            print(f"  ✅ Fixed code saved: {solution.saved_path}")

        # Always print the diagnosis
        if root_cause:
            print(f"\n  🔍 Root cause: {root_cause}")
        if changes:
            print(f"  🔧 Fix: {changes[:150]}...")

        return solution

    def explain(self, code: str) -> str:
        """
        Explain what a piece of code does in plain English.

        Args:
            code : the code to explain

        Returns:
            str — structured explanation
        """
        print("📖 Explaining code...")
        prompt = EXPLAIN_PROMPT.format(code=code)
        result = self.llm.invoke(prompt)
        print("\n" + "─" * 60)
        print(result)
        print("─" * 60)
        return result

    def review(self, code: str) -> str:
        """
        Perform a structured code review with specific improvement suggestions.

        Args:
            code : the code to review

        Returns:
            str — structured review with issues and recommendations
        """
        print("🔎 Reviewing code...")
        prompt = REVIEW_PROMPT.format(code=code)
        result = self.llm.invoke(prompt)
        print("\n" + "─" * 60)
        print(result)
        print("─" * 60)
        return result

    def generate(
        self,
        spec:     str,
        language: str = "python",
        save:     bool = True,
    ) -> CodeSolution:
        """
        Generate boilerplate code from a natural language specification.

        Args:
            spec     : description of what to build
            language : target programming language

        Returns:
            CodeSolution with complete generated code
        """
        print(f"⚙️  Generating {language} code from specification...")
        prompt   = GENERATE_PROMPT.format(spec=spec, language=language)
        response = self.llm.invoke(prompt)
        code     = extract_code_block(response)

        solution = CodeSolution(
            task        = spec,
            code        = code,
            explanation = "Generated from specification.",
            language    = language,
        )

        if save:
            solution.saved_path = save_solution(solution, self.output_dir)
            print(f"  ✅ Generated code saved: {solution.saved_path}")

        return solution

    def convert(
        self,
        code:        str,
        target_lang: str,
        source_lang: str = "",
        save:        bool = True,
    ) -> CodeSolution:
        """
        Convert code from one language to another.

        Args:
            code        : source code to convert
            target_lang : target language (e.g. "javascript")
            source_lang : source language (auto-detected if empty)

        Returns:
            CodeSolution with converted code
        """
        if not source_lang:
            source_lang = detect_language(code)
        print(f"🔄 Converting {source_lang} → {target_lang}...")

        prompt   = CONVERT_PROMPT.format(
            code=code, source_lang=source_lang, target_lang=target_lang
        )
        response = self.llm.invoke(prompt)
        converted = extract_code_block(response)

        solution = CodeSolution(
            task        = f"Convert {source_lang} to {target_lang}",
            code        = converted,
            explanation = f"Converted from {source_lang} to {target_lang}.",
            language    = target_lang,
        )

        if save:
            solution.saved_path = save_solution(solution, self.output_dir)
            print(f"  ✅ Converted code saved: {solution.saved_path}")

        return solution

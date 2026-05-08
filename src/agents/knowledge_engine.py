"""
Knowledge Engine Agent
======================
Answers questions on any topic using live web search + local LLM reasoning.
Combines real-time information retrieval with structured, grounded responses.

Capabilities:
    - ask(question)           → web-grounded answer on any topic
    - deep_dive(topic)        → comprehensive multi-source research summary
    - compare(a, b, context)  → structured comparison of two options
    - summarise(text)         → condense long text into key points
    - explain_concept(topic)  → clear technical explanation with examples
    - fact_check(claim)       → searches for evidence supporting or refuting a claim
"""

import logging
import datetime
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class KnowledgeResponse:
    question:  str
    answer:    str
    sources:   List[str] = field(default_factory=list)
    timestamp: str = ""
    mode:      str = "ask"

    def print(self):
        print("\n" + "═" * 65)
        print(f"  Q: {self.question}")
        print("═" * 65)
        print(self.answer)
        if self.sources:
            print("\n📚 Sources:")
            for i, s in enumerate(self.sources[:3], 1):
                print(f"  [{i}] {s}")
        print("═" * 65 + "\n")


# ── Web Search ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search DuckDuckGo and return a list of result dicts.
    Each dict has: title, url, snippet.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return []

    results = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
            for r in raw:
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        logger.warning(f"Web search failed for '{query}': {e}")
    return results


def format_search_context(results: List[dict]) -> str:
    """Format search results into a clean context string for the LLM prompt."""
    if not results:
        return "No search results available."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n{r['snippet']}\nSource: {r['url']}")
    return "\n\n".join(lines)


# ── Prompt Templates ──────────────────────────────────────────────────────────

ASK_PROMPT = """
You are a knowledgeable assistant. Answer the question below using the provided
web search context. Base your answer primarily on the context, but you may supplement
with your own knowledge when the context is insufficient.

If the context directly answers the question, cite which sources [1], [2], etc.
If the context is irrelevant, say so and answer from your own knowledge.

SEARCH CONTEXT:
{context}

QUESTION: {question}

Provide a clear, factual, well-structured answer. Be concise but complete.
"""

DEEP_DIVE_PROMPT = """
You are a research analyst. Based on the search results below, write a comprehensive
summary on the topic: "{topic}"

SEARCH RESULTS:
{context}

Structure your response as:
1. OVERVIEW (2-3 sentences)
2. KEY FINDINGS (bullet points — most important facts/insights)
3. CURRENT STATE (what is happening now / latest developments)
4. IMPLICATIONS (why this matters)

Be specific, cite sources where relevant [1][2][3], and avoid vague generalities.
"""

COMPARE_PROMPT = """
You are an expert analyst. Compare "{option_a}" vs "{option_b}" in the context of: {context}

Based on the following information:
{search_context}

Provide a structured comparison:

| Dimension | {option_a} | {option_b} |
|-----------|-----------|-----------|
[Fill in 5-7 relevant comparison dimensions]

VERDICT: Which is better for {context} and why? (2-3 sentences, be direct)
"""

FACT_CHECK_PROMPT = """
You are a fact-checker. Evaluate whether the following claim is true, false, or uncertain.

CLAIM: "{claim}"

EVIDENCE FROM WEB SEARCH:
{context}

Respond with:
VERDICT: [TRUE / FALSE / UNCERTAIN / PARTIALLY TRUE]
CONFIDENCE: [High / Medium / Low]
REASONING: [2-3 sentences explaining what the evidence shows]
NUANCE: [any important caveats or context]
"""

EXPLAIN_CONCEPT_PROMPT = """
You are an expert teacher. Explain the following concept clearly:

CONCEPT: {topic}

ADDITIONAL CONTEXT FROM WEB:
{context}

Structure your explanation as:
1. SIMPLE DEFINITION (1 sentence — explain it to a smart non-expert)
2. HOW IT WORKS (the core mechanism, step by step)
3. CONCRETE EXAMPLE (a specific, real-world example)
4. WHY IT MATTERS (practical applications or importance)
5. COMMON MISCONCEPTIONS (1-2 things people often get wrong)

Be technical enough to be useful, but clear enough to be understood.
"""


# ── Knowledge Engine Agent ────────────────────────────────────────────────────

class KnowledgeEngineAgent:
    """
    Answers questions on any topic using live web search + local LLM reasoning.
    All inference runs locally via Ollama.
    """

    def __init__(self, llm_client, ntfy_topic: str = ""):
        self.llm        = llm_client
        self.ntfy_topic = ntfy_topic

    def ask(self, question: str, num_sources: int = 5) -> KnowledgeResponse:
        """
        Answer any question using web search + LLM reasoning.

        Args:
            question    : the question to answer
            num_sources : number of web search results to use as context

        Returns:
            KnowledgeResponse with answer and sources
        """
        print(f"🔍 Searching for: {question}")
        results = web_search(question, max_results=num_sources)
        context = format_search_context(results)

        print("🧠 Reasoning with local LLM...")
        prompt = ASK_PROMPT.format(context=context, question=question)
        answer = self.llm.invoke(prompt)

        response = KnowledgeResponse(
            question  = question,
            answer    = answer,
            sources   = [r["url"] for r in results if r.get("url")],
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            mode      = "ask",
        )
        response.print()
        return response

    def deep_dive(self, topic: str, num_sources: int = 8) -> KnowledgeResponse:
        """
        Comprehensive multi-source research on a topic.

        Args:
            topic       : topic to research in depth
            num_sources : number of web sources to aggregate

        Returns:
            KnowledgeResponse with structured research summary
        """
        print(f"📚 Deep dive: {topic}")
        # Use multiple search angles for richer coverage
        queries = [topic, f"{topic} latest 2025 2026", f"{topic} explained"]
        all_results = []
        for q in queries:
            all_results.extend(web_search(q, max_results=3))

        # Deduplicate by URL
        seen, unique = set(), []
        for r in all_results:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)

        context = format_search_context(unique[:num_sources])
        prompt  = DEEP_DIVE_PROMPT.format(topic=topic, context=context)
        answer  = self.llm.invoke(prompt)

        response = KnowledgeResponse(
            question  = f"Deep dive: {topic}",
            answer    = answer,
            sources   = [r["url"] for r in unique[:num_sources]],
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            mode      = "deep_dive",
        )
        response.print()
        return response

    def compare(self, option_a: str, option_b: str, context: str = "general use") -> KnowledgeResponse:
        """
        Structured comparison of two options, technologies, or approaches.

        Args:
            option_a : first option to compare
            option_b : second option to compare
            context  : the use-case or context for the comparison

        Returns:
            KnowledgeResponse with comparison table and verdict
        """
        print(f"⚖️  Comparing: {option_a} vs {option_b}")
        query   = f"{option_a} vs {option_b} comparison {context}"
        results = web_search(query, max_results=6)
        context_str = format_search_context(results)

        prompt = COMPARE_PROMPT.format(
            option_a       = option_a,
            option_b       = option_b,
            context        = context,
            search_context = context_str,
        )
        answer = self.llm.invoke(prompt)

        response = KnowledgeResponse(
            question  = f"{option_a} vs {option_b} ({context})",
            answer    = answer,
            sources   = [r["url"] for r in results],
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            mode      = "compare",
        )
        response.print()
        return response

    def summarise(self, text: str, style: str = "bullet") -> str:
        """
        Summarise a long piece of text.

        Args:
            text  : text to summarise
            style : "bullet" for bullet points, "paragraph" for prose

        Returns:
            str — the summary
        """
        print("📝 Summarising text...")
        if style == "bullet":
            instruction = "Provide a bullet-point summary with the 5–7 most important points."
        else:
            instruction = "Write a 3-paragraph prose summary covering the main points."

        prompt = f"""Summarise the following text.

{instruction}

TEXT:
{text[:6000]}

SUMMARY:"""
        result = self.llm.invoke(prompt)
        print("\n" + "─" * 60)
        print(result)
        print("─" * 60)
        return result

    def explain_concept(self, topic: str) -> KnowledgeResponse:
        """
        Clear technical explanation of any concept with examples.

        Args:
            topic : concept or technology to explain

        Returns:
            KnowledgeResponse with structured explanation
        """
        print(f"💡 Explaining: {topic}")
        results = web_search(f"{topic} explained simply", max_results=4)
        context = format_search_context(results)
        prompt  = EXPLAIN_CONCEPT_PROMPT.format(topic=topic, context=context)
        answer  = self.llm.invoke(prompt)

        response = KnowledgeResponse(
            question  = f"Explain: {topic}",
            answer    = answer,
            sources   = [r["url"] for r in results],
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            mode      = "explain",
        )
        response.print()
        return response

    def fact_check(self, claim: str) -> KnowledgeResponse:
        """
        Check whether a claim is supported by web evidence.

        Args:
            claim : statement to fact-check

        Returns:
            KnowledgeResponse with verdict and reasoning
        """
        print(f"🔎 Fact-checking: {claim}")
        results = web_search(claim, max_results=6)
        context = format_search_context(results)
        prompt  = FACT_CHECK_PROMPT.format(claim=claim, context=context)
        answer  = self.llm.invoke(prompt)

        response = KnowledgeResponse(
            question  = f"Fact check: {claim}",
            answer    = answer,
            sources   = [r["url"] for r in results],
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            mode      = "fact_check",
        )
        response.print()
        return response

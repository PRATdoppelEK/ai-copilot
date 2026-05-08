"""
LLM client wrapper — unified interface for Ollama local models.
Handles retries, streaming, and structured JSON output parsing.
"""

import json
import re
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wrapper around OllamaLLM with retry logic, structured output parsing,
    and consistent prompt formatting.
    """

    def __init__(self, model: str = "llama3", temperature: float = 0.2):
        self.model       = model
        self.temperature = temperature
        self._llm        = None

    def _get_llm(self):
        if self._llm is None:
            try:
                from langchain_ollama import OllamaLLM
                self._llm = OllamaLLM(model=self.model, temperature=self.temperature)
            except ImportError:
                raise ImportError(
                    "langchain-ollama not installed. Run: pip install langchain-ollama"
                )
        return self._llm

    def invoke(self, prompt: str, retries: int = 2) -> str:
        """Call the LLM with automatic retry on failure."""
        llm = self._get_llm()
        for attempt in range(retries + 1):
            try:
                response = llm.invoke(prompt)
                return response.strip()
            except Exception as e:
                if attempt < retries:
                    wait = 2 ** attempt
                    logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM call failed after {retries+1} attempts: {e}")
                    raise

    def invoke_json(self, prompt: str) -> Optional[dict]:
        """
        Call the LLM expecting a JSON response.
        Strips markdown fences and parses safely.
        """
        json_prompt = prompt + "\n\nRespond with valid JSON only. No markdown, no explanation."
        raw = self.invoke(json_prompt)
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw response: {raw[:200]}")
            return None

    def invoke_table(self, prompt: str, separator: str = "|") -> list[list[str]]:
        """
        Call the LLM expecting pipe-separated table rows.
        Returns a list of row lists.
        """
        raw = self.invoke(prompt)
        rows = []
        for line in raw.split("\n"):
            if separator in line:
                parts = [p.strip() for p in line.split(separator)]
                parts = [p for p in parts if p]  # Remove empty strings
                if parts:
                    rows.append(parts)
        return rows

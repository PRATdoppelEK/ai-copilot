"""
Job Hunter Agent
================
Searches the live job market, scores opportunities against your CV,
generates tailored cover letters, and sends match alerts to your phone.

Pipeline:
    1. Live web search via DuckDuckGo (no API key)
    2. CV loaded into ChromaDB vector store for RAG-based matching
    3. Local LLM scores and ranks each opportunity (0–100)
    4. Cover letter generated as formatted PDF per top match
    5. Push notification sent with ranked summary
"""

import os
import re
import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class JobMatch:
    title:       str
    company:     str
    snippet:     str
    source_url:  str = ""
    match_score: int = 0
    reasoning:   str = ""
    action:      str = ""


@dataclass
class JobHunterResult:
    matches:       List[JobMatch] = field(default_factory=list)
    cover_letters: List[str]      = field(default_factory=list)  # PDF paths
    summary:       str = ""


# ── CV Vector Store ───────────────────────────────────────────────────────────

class CVKnowledgeBase:
    """Loads CV PDFs into a ChromaDB vector store for semantic retrieval."""

    def __init__(self, cv_files: List[str], persist_dir: str = "data/vectorstore/cv"):
        self.cv_files    = cv_files
        self.persist_dir = persist_dir
        self._store      = None

    def build(self) -> bool:
        """Build or reload the vector store. Returns True if successful."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_ollama import OllamaEmbeddings
            from langchain_community.vectorstores import Chroma

            # Reload if already built
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                logger.info("♻️  Reloading existing CV vector store")
                embeddings   = OllamaEmbeddings(model="nomic-embed-text")
                self._store  = Chroma(persist_directory=self.persist_dir,
                                      embedding_function=embeddings,
                                      collection_name="cv_store")
                return True

            all_pages = []
            for filepath in self.cv_files:
                if os.path.exists(filepath):
                    loader = PyPDFLoader(filepath)
                    all_pages.extend(loader.load())
                    logger.info(f"  Loaded CV: {filepath}")
                else:
                    logger.warning(f"  CV file not found: {filepath}")

            if not all_pages:
                logger.warning("No CV files found — will use profile summary only")
                return False

            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
            chunks   = splitter.split_documents(all_pages)

            embeddings  = OllamaEmbeddings(model="nomic-embed-text")
            self._store = Chroma.from_documents(
                documents         = chunks,
                embedding         = embeddings,
                collection_name   = "cv_store",
                persist_directory = self.persist_dir,
            )
            logger.info(f"  ✅ CV vector store built ({len(chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"Failed to build CV vector store: {e}")
            return False

    def get_relevant_context(self, query: str, k: int = 4) -> str:
        """Retrieve the most relevant CV sections for a given job query."""
        if self._store is None:
            return ""
        try:
            docs = self._store.similarity_search(query, k=k)
            return "\n\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.warning(f"Vector store retrieval failed: {e}")
            return ""


# ── Web Search ────────────────────────────────────────────────────────────────

def search_jobs(queries: List[str]) -> str:
    """Search DuckDuckGo for job listings. Returns aggregated raw text."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return ""

    all_results = []
    with DDGS() as ddgs:
        for query in queries:
            try:
                results = list(ddgs.text(query, max_results=5))
                for r in results:
                    all_results.append(
                        f"Title: {r.get('title', '')}\n"
                        f"URL: {r.get('href', '')}\n"
                        f"Snippet: {r.get('body', '')}\n"
                    )
                logger.info(f"  🔍 '{query}' → {len(results)} results")
            except Exception as e:
                logger.warning(f"  Search failed for '{query}': {e}")

    return "\n---\n".join(all_results)


# ── LLM Job Scoring ───────────────────────────────────────────────────────────

def score_and_rank_jobs(
    search_results: str,
    cv_context:     str,
    profile_summary: str,
    llm_client,
) -> List[JobMatch]:
    """Use the LLM to identify and score the top job matches."""

    prompt = f"""
You are a career advisor helping an ML engineer find jobs in Germany.

CANDIDATE PROFILE:
{profile_summary}

RELEVANT CV EXPERIENCE:
{cv_context}

LIVE JOB SEARCH RESULTS:
{search_results[:4000]}

TASK:
Identify the top 3 most relevant job opportunities from the results above.
For each, respond with a pipe-separated row in this exact format:
Job Title | Company | Match Score (0-100) | Why it fits (1 sentence) | Immediate action

Output ONLY 3 rows. No headers, no numbering, no extra text.
"""
    rows    = llm_client.invoke_table(prompt, separator="|")
    matches = []
    for row in rows[:3]:
        if len(row) >= 4:
            matches.append(JobMatch(
                title        = row[0] if len(row) > 0 else "Unknown",
                company      = row[1] if len(row) > 1 else "Unknown",
                match_score  = int(re.sub(r"\D", "", row[2])) if len(row) > 2 else 0,
                reasoning    = row[3] if len(row) > 3 else "",
                action       = row[4] if len(row) > 4 else "",
                snippet      = "",
            ))
    return matches


# ── Cover Letter Generator ────────────────────────────────────────────────────

def generate_cover_letter_pdf(
    match: JobMatch,
    candidate_name:  str,
    candidate_email: str,
    profile_summary: str,
    llm_client,
    output_dir: str = "outputs/cover_letters",
) -> str:
    """Generate a tailored cover letter PDF for a job match."""

    prompt = f"""
Write a professional cover letter for {candidate_name} applying for the role below.

CANDIDATE PROFILE:
{profile_summary}

TARGET ROLE:
Company: {match.company}
Title: {match.title}

INSTRUCTIONS:
- Write exactly 3 paragraphs. No salutation. No signature block. No headers.
- Paragraph 1: Why this specific company and role
- Paragraph 2: Connect 2 specific candidate skills/projects directly to this role
- Paragraph 3: Availability, enthusiasm, clear call to action
- Tone: Professional, confident, suitable for German engineering companies (in English)
- Do NOT use "passionate", "hard-working", "team player", or other clichés
"""
    letter_text = llm_client.invoke(prompt)

    # Build PDF
    os.makedirs(output_dir, exist_ok=True)
    safe_company = re.sub(r"[^\w]", "_", match.company)
    filepath = os.path.join(output_dir, f"CoverLetter_{safe_company}.pdf")

    c = canvas.Canvas(filepath, pagesize=A4)
    w, h = A4
    m    = 20 * mm

    # Header bar
    c.setFillColorRGB(0.01, 0.20, 0.40)
    c.rect(0, h - 28*mm, w, 28*mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(m, h - 12*mm, candidate_name)
    c.setFont("Helvetica", 9)
    c.drawString(m, h - 19*mm, f"Munich, Germany  ·  {candidate_email}")

    # Meta
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 10)
    c.drawString(m, h - 38*mm, f"Munich, {datetime.date.today().strftime('%d %B %Y')}")
    c.drawString(m, h - 44*mm, f"To: Hiring Team, {match.company}")
    c.drawString(m, h - 50*mm, f"Re: Application — {match.title}")

    # Divider
    c.setStrokeColorRGB(0.01, 0.20, 0.40)
    c.setLineWidth(1)
    c.line(m, h - 54*mm, w - m, h - 54*mm)

    # Salutation
    c.setFont("Helvetica-Bold", 11)
    c.drawString(m, h - 62*mm, "Dear Hiring Team,")

    # Body
    c.setFont("Helvetica", 10.5)
    txt = c.beginText(m, h - 72*mm)
    txt.setLeading(15)
    for para in letter_text.strip().split("\n"):
        para = para.strip()
        if not para:
            txt.textLine("")
            continue
        words, line = para.split(), ""
        for word in words:
            if len(line) + len(word) + 1 <= 97:
                line = (line + " " + word).strip()
            else:
                txt.textLine(line)
                line = word
        if line:
            txt.textLine(line)
        txt.textLine("")
    c.drawText(txt)

    # Footer
    c.setFont("Helvetica", 10.5)
    c.drawString(m, 38*mm, "Yours sincerely,")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(m, 31*mm, candidate_name)
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(0, 20*mm, w, 20*mm)
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawCentredString(w / 2, 13*mm, f"Generated by AI Copilot — {datetime.date.today()}")

    c.save()
    return filepath


# ── Main Agent ────────────────────────────────────────────────────────────────

class JobHunterAgent:
    """
    Autonomous job search agent.
    Searches → scores → generates cover letters → notifies.
    """

    def __init__(self, config, llm_client):
        self.config     = config
        self.llm        = llm_client
        self.cv_kb      = CVKnowledgeBase(
            cv_files    = config.candidate.cv_files,
            persist_dir = config.paths.get("vectorstore", "data/vectorstore/"),
        )

    def run(self, notify: bool = True) -> JobHunterResult:
        result = JobHunterResult()
        print("\n🔍 Job Hunter Agent starting...\n")

        # Step 1: Build CV knowledge base
        print("📄 Loading CV knowledge base...")
        self.cv_kb.build()

        # Step 2: Search job market
        print("\n📡 Scanning live job market...")
        raw_results = search_jobs(self.config.job_search.queries)
        if not raw_results:
            print("  ⚠️  No search results returned.")
            return result

        # Step 3: Score matches
        print("\n🧠 Scoring job matches with local LLM...")
        cv_context = self.cv_kb.get_relevant_context(raw_results[:300])
        result.matches = score_and_rank_jobs(
            raw_results,
            cv_context,
            self.config.candidate.profile_summary,
            self.llm,
        )

        # Step 4: Print ranked results
        print("\n" + "═" * 65)
        print("  TOP JOB MATCHES")
        print("═" * 65)
        for i, m in enumerate(result.matches, 1):
            print(f"  [{i}] {m.match_score}/100 — {m.title} @ {m.company}")
            print(f"       Why: {m.reasoning}")
            print(f"       Action: {m.action}\n")
        print("═" * 65)

        # Step 5: Generate cover letters for top matches
        print("\n✍️  Generating cover letters...")
        for m in result.matches[:2]:
            try:
                path = generate_cover_letter_pdf(
                    match            = m,
                    candidate_name   = self.config.candidate.name,
                    candidate_email  = self.config.candidate.email,
                    profile_summary  = self.config.candidate.profile_summary,
                    llm_client       = self.llm,
                    output_dir       = self.config.paths.get("cover_letters", "outputs/cover_letters/"),
                )
                result.cover_letters.append(path)
                print(f"  ✅ {path}")
            except Exception as e:
                logger.error(f"Cover letter generation failed for {m.company}: {e}")

        # Step 6: Notification
        if notify and self.config.notifications.enabled:
            from src.utils.notifier import send_notification
            summary_lines = [f"#{i} {m.match_score}/100 — {m.title} @ {m.company}"
                              for i, m in enumerate(result.matches, 1)]
            send_notification(
                message  = "\n".join(summary_lines),
                title    = "🚀 Job Matches Found",
                topic    = self.config.notifications.ntfy_topic,
                tags     = "briefcase",
                priority = "high",
            )

        result.summary = f"Found {len(result.matches)} matches, {len(result.cover_letters)} cover letters generated."
        print(f"\n✅ Job Hunter complete. {result.summary}")
        return result

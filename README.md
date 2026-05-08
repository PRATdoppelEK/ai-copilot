# AI Copilot — Personal Autonomous Assistant

A fully local, privacy-first AI assistant with three specialised engines:
**Job Hunter**, **Code Assistant**, and **Knowledge Engine**.
Runs entirely on your machine via Ollama — no cloud API, no subscription, no data leaving your device.

---

## Project overview

Most AI assistants are general-purpose chat interfaces. This copilot is purpose-built
for an engineer's actual workflow: finding relevant jobs in a specialised market,
solving domain-specific coding problems, and getting well-researched answers on technical topics.

All three engines share a single local LLM (Llama 3 via Ollama), a unified config file,
and a consistent CLI interface — so you interact with one tool, not three separate scripts.

**Why local LLM:** CV data, job search strategy, and code are sensitive.
Running everything locally via Ollama means zero exposure to third-party APIs —
your data stays on your machine.

---

## Three engines, one interface

### 🔍 Job Hunter
Autonomously searches the live job market, scores opportunities against your CV
using RAG, generates tailored cover letters as PDFs, and sends ranked alerts to your phone.

| Step | What happens |
|------|-------------|
| Search | DuckDuckGo queries for your target roles — no API key |
| CV matching | Your CV is loaded into ChromaDB; relevant sections retrieved per job via semantic search |
| Scoring | Local LLM scores each opportunity 0–100 with specific reasoning |
| Cover letters | PDF generated per top match — tailored to company and role |
| Notification | Push alert with ranked summary sent to phone via ntfy |

### 💻 Code Assistant
Solves coding problems, debugs errors, explains code, reviews quality,
generates boilerplate, and converts between languages — all saved to file.

| Command | What it does |
|---------|-------------|
| `solve` | Generates a working solution from a natural language problem description |
| `debug` | Diagnoses the root cause and returns fixed code with explanation |
| `explain` | Plain-English explanation of what code does and how |
| `review` | Structured code review: correctness, performance, readability, security |
| `generate` | Production-ready boilerplate from a specification |
| `convert` | Translates code between languages (e.g. MATLAB → Python) |

### 🧠 Knowledge Engine
Answers any question using live web search + local LLM reasoning.
Grounded answers with source citations — not hallucinated responses.

| Command | What it does |
|---------|-------------|
| `ask` | Answer any question using web-retrieved context |
| `dive` | Comprehensive multi-source research summary on a topic |
| `compare` | Structured comparison table + verdict for two options |
| `explain-concept` | Technical explanation with examples and common misconceptions |
| `fact` | Fact-check any claim against current web evidence |
| `sum` | Summarise any long text into bullet points |

---

## Project structure

```
ai-copilot/
│
├── copilot.py                        # Main entry point — interactive CLI or --mode flag
│
├── src/
│   ├── agents/
│   │   ├── job_hunter.py             # Job search, CV RAG, scoring, cover letter PDF, ntfy
│   │   ├── code_assistant.py         # Solve, debug, explain, review, generate, convert
│   │   └── knowledge_engine.py       # Ask, deep-dive, compare, summarise, fact-check
│   │
│   └── utils/
│       ├── config_loader.py          # YAML config → typed dataclasses
│       ├── llm_client.py             # Ollama wrapper with retry, JSON, table parsing
│       └── notifier.py               # ntfy push notification utility
│
├── configs/
│   └── config.yaml                   # Your personal config (CV paths, ntfy topic, queries)
│
├── data/
│   ├── cv/                           # Place your CV PDFs here (gitignored)
│   └── vectorstore/                  # ChromaDB store (auto-created, gitignored)
│
├── outputs/
│   ├── cover_letters/                # Generated PDF cover letters
│   └── code_solutions/               # Saved code outputs with timestamps
│
├── notebooks/
│   └── copilot_demo.ipynb            # Interactive walkthrough of all three engines
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

**Step 1 — Install Ollama and pull models:**
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
ollama pull nomic-embed-text
```

**Step 2 — Clone and install:**
```bash
git clone https://github.com/PRATdoppelEK/ai-copilot.git
cd ai-copilot
pip install -r requirements.txt
```

**Step 3 — Configure:**
```bash
# Edit configs/config.yaml:
# - Add your name, email, LinkedIn
# - Set paths to your CV PDFs
# - Set your job search queries
# - Set your ntfy topic (optional)
```

**Step 4 — Add CV files:**
```bash
# Place your CV PDFs at the paths set in config.yaml, e.g.:
data/cv/Lebenslauf.pdf
data/cv/Projekterfahrung.pdf
```

---

## Usage

**Interactive mode (recommended):**
```bash
python copilot.py
```

**Direct CLI flags:**
```bash
python copilot.py --mode job                        # Full job search run
python copilot.py --mode ask --input "What is RAG?" # Quick question
python copilot.py --mode solve --input "Write a Python function to parse ISO 8601 dates" --lang python
python copilot.py --mode explain --input "attention mechanism in transformers"
python copilot.py --mode dive --input "solid-state batteries 2025"
```

**Interactive session examples:**
```
Copilot › jobs
Copilot › solve Write an LSTM training loop in PyTorch for time-series regression
Copilot › ask What are the best vector databases for production RAG in 2026?
Copilot › compare FAISS vs ChromaDB
Copilot › explain-concept Kalman filter
Copilot › fact GPT-4 can process images natively
Copilot › debug
  [paste broken code, end with ---]
  [paste error message]
```

---

## Requirements

```
langchain-community>=0.4.0
langchain-ollama>=0.2.0
chromadb>=0.5.0
pypdf>=4.0.0
duckduckgo-search>=8.0.0
reportlab>=4.0.0
requests>=2.28.0
pyyaml>=6.0.0
```

---

## Key concepts

**Local-first architecture:** Every LLM call goes to Ollama running on `localhost`.
CV data, job search queries, and code snippets never leave your machine.

**RAG for CV matching:** Rather than including your full CV in every prompt,
ChromaDB retrieves only the most relevant sections per job listing — keeping
prompts focused and LLM responses precise.

**Structured prompting:** Each agent uses carefully designed prompt templates that
enforce specific output formats (pipe-separated tables, named sections) so responses
can be parsed reliably without fragile regex heuristics.

**Persistent vector store:** The CV vector store is built once and reloaded on
subsequent runs — no re-embedding on every session start.

**ntfy notifications:** A free, open-source push notification service.
No account, no app registration — subscribe to any topic string in the ntfy app
and the copilot pushes job match summaries to your phone automatically.

---

## Author

**Prateek Gaur** — Applied ML Engineer | Agentic AI | TU Berlin M.Sc.
[LinkedIn](https://www.linkedin.com/in/prateek-gaur-15a629b4) · [prateekgaur@gmx.de](mailto:prateekgaur@gmx.de)

---

## License

MIT License — free to use with attribution.

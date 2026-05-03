# Value Brief

Value Brief: Covering your assets. An automated daily digest that accumulates insights on your assets and tracks intrinsic value, margin of safety, and portfolio fundamentals.

## Why It Exists

**The Problem:** Rigorous fundamental analysis requires hours of manual data aggregation and modeling. Meanwhile, standard automated stock screeners lack the nuanced qualitative synthesis needed to determine a true margin of safety.

**The Solution:** Value Brief is an automated daily digest that acts as a personal team of investment analysts. By bridging the gap between generative AI and deterministic financial modeling, it translates complex, time-intensive research workflows into a streamlined pipeline that delivers actionable, fundamental-driven investment theses.

## Architecture

Value Brief is powered by an agentic research workflow orchestrated by LangGraph. It relies on specialised AI agents that operate iteratively and securely maintain state using Supabase PostgreSQL as a checkpointer.

- **Supervisor**: Controls workflow routing, validates inputs, and ensures all research parameters.
- **Bull Analyst**: Operates standalone with recursive web scraping and research tooling to synthesise growth catalysts, competitive moats, and upside cases.
- **Bear Analyst**: Utilizes matching toolsets independently to extract short theses, highlight speculative risks, downside scenarios, and mapping margin issues.
- **Judge Analyst**: Synthesizes conflicting fundamental researches, and leverages a configured `ValuationModel` to execute a multi-stage Discounted Cash Flow (DCF). Weighs probability models (Bear/Base/Bull), terminal value limits, and produces a reconciled final decision based strictly on margin of safety.
- **Report Generator**: Reconciles the output into isolated Markdown timelines (with segmented debug graphs vs presentation-ready final structures). Builds citation manifests and upserts valuations to Supabase.
- **Curator**: Post-run knowledge maintenance agent. Manages the hybrid RAG lifecycle — marks cited memories, prunes stale vectors, consolidates old vault files into monthly syntheses, tracks thesis drift, deduplicates content, and monitors vector storage health.

### Visualisation

```mermaid
graph TD
    S[Supervisor<br/>Validates inputs, fetches financial data<br/>Routes workflow, seeds vault]

    subgraph "Research Tools (available to Analysts)"
        WS[Web Search]
        PW[Parse Website]
        NS[News Search<br/>Last day]
    end

    subgraph "Independent Analysis"
        BA[Bull Analyst<br/>Recursive scraping and synthesis<br/>Growth catalysts, moats, upside]
        BE[Bear Analyst<br/>Recursive scraping and synthesis<br/>Short theses, risks, downside]
    end

    BA --> WS
    BA --> PW
    BA --> NS
    BE --> WS
    BE --> PW
    BE --> NS

    S --> BA
    S --> BE

    BA --> J
    BE --> J

    subgraph "Valuation and Reconciliation"
        J[Judge Analyst<br/>Synthesizes conflicting research<br/>Executes multi-stage DCF<br/>Weighs probability models<br/>Applies margin of safety]
        VM[ValuationModel<br/>Two-stage 10-year DCF<br/>Gordon Growth terminal value]
    end

    J --> VM
    VM --> J

    J --> R[Report Generator<br/>Assembles final + debug reports<br/>Builds citation manifest<br/>Upserts valuation to DB]

    R --> C[Curator<br/>Post-run knowledge maintenance<br/>Memory lifecycle management<br/>Thesis drift tracking<br/>Vault consolidation and pruning]

    subgraph "Hybrid RAG Storage"
        V[Local Vault<br/> Daily research documents<br/> Monthly syntheses<br/> Block-level citation IDs]
        VMEM[Vector Memory<br/> Supabase pgvector<br/> Semantic similarity search<br/> Priority-tiered insights]
    end

    C --> V
    C --> VMEM

    S --> V

    subgraph "Persistence Layer"
        DB[(Supabase PostgreSQL<br/> LangGraph checkpointer<br/> Valuations table<br/> Thesis drifts table<br/> Investment memories vector table)]
    end

    C --> DB
    R --> DB
    S --> DB

    S -.-> AV&YF[Alpha Vantage & yfinance<br/>direct router call]

    C --> Done[Final Output]
```

## Sample Output Report

> Completed: 2026-04-12T01:06:59.965264

### Investment Report: Generic Corp (GNC)

### Investment Thesis

**Verdict** — Strong Buy on weakness, targeting a probability-weighted intrinsic value of approximately $408 per share.

**Rationale** — Generic Corp’s core investment case remains anchored in its entrenched enterprise workflow dominance and commercially indemnified AI architecture... the qualitative reality points to a deliberate monetization pivot from volume-based subscriptions to value-driven AI credit consumption. The current disconnect between Generic Corp’s durable cash generation and its compressed valuation multiple represents a classic transitional mispricing.

### Key Risks

- Accelerated enterprise seat consolidation outpacing AI credit monetization.
- Prolonged leadership vacuum delaying strategic capital allocation.
- Competitive disruption from AI-native platforms capturing mid-market share.

### DCF Valuation

| Scenario | Probability | Intrinsic Value | Margin of Safety |
| :------- | :---------- | :-------------- | :--------------- |
| Bear     | 25%         | $225.56         | -10.1%           |
| **Base** | **50%**     | **$380.93**     | **34.8%**        |
| Bull     | 25%         | $643.82         | 61.4%            |

**Expected Intrinsic Value:** $407.81  
**Current Price:** $248.39  
**Expected 5-Year CAGR:** 16.4%  
**Recommendation:** Strong Buy

---

## 🔗 Sources

- https://example.com/financials/gnc
- https://example.com/news/gnc-upgrades

---

## Hybrid RAG: Insight Lifecycle Management

Value Brief maintains a dual-layer knowledge store that grows smarter with each run. Fresh research is written to both a **local Markdown vault** (cold storage) and a **Supabase pgvector table** (hot semantic storage). The Curator agent manages the full lifecycle — creation, citation, consolidation, pruning, and drift tracking — so the system self-maintains and avoids bloat.

### Storage Architecture

| Layer                   | Storage                                                                                    | Purpose                                                                                                         | Retention                                              |
| :---------------------- | :----------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- |
| **Vault** (cold)        | `data/vault/{TICKER}/` — Markdown files with YAML frontmatter and block-level citation IDs | Immutable, auditable record of every research document and monthly synthesis                                    | Indefinite (deduplicated by content hash)              |
| **Vector Memory** (hot) | `investment_memories` table — 1536-dim pgvector embeddings via `text-embedding-3-small`    | Semantic similarity search across ticker-scoped insights; priority-tiered (0=noise, 1=secondary, 2=first-party) | Actively pruned; uncited vectors deleted after 90 days |

### Lifecycle Stages

```mermaid
graph LR
    subgraph "1. Creation (per run)"
        SUP[Supervisor]
        ANALYSTS[Analysts]
    end

    subgraph "2. Curation (per run)"
        REP[Report Generator]
        CM[Citation Manifest]
        CUR[Curator]
    end

    subgraph "3. Consolidation (monthly)"
        OLD[Vault files 90 days old]
        GRP[Monthly Groups]
        SYNTH[(Monthly Synthesis)]
    end

    subgraph "4. Aggressive Pruning"
        OLDEST[Oldest summary vectors]
        HIST[(Historical Summary)]
    end

    subgraph "5. Drift Tracking"
        OLDV[Prior valuation]
        NEWV[Current judge decision]
        DRIFT[Delta Calculation]
    end

    subgraph "6. Deduplication"
        DUP[Duplicate vault files]
    end

    VAULT[(Local Vault)]
    VEC[(Vector Memory)]
    DRIFTTBL[(thesis_drifts)]

    SUP -->|fundamentals| VAULT
    SUP -->|embeddings| VEC
    ANALYSTS -->|research docs| VAULT
    ANALYSTS -->|insights| VEC
    REP -->|builds manifest| CM
    CM -->|resolves block IDs| VAULT
    CUR -->|mark cited, delete uncited| VEC
    OLD -->|group by month| GRP
    GRP -->|LLM synthesize| SYNTH
    SYNTH -->|archive sources| VAULT
    SYNTH -->|swap granular → summary| VEC
    OLDEST -->|LLM merge| HIST
    HIST -->|keep 3 most recent| VEC
    OLDV -->|compare verdict + EV| DRIFT
    NEWV --> DRIFT
    DRIFT -->|record δ% + key changes| DRIFTTBL
    DUP -->|keep earliest, remove rest| VAULT

    linkStyle 0 stroke:#3b82f6,stroke-width:2px
    linkStyle 1 stroke:#3b82f6,stroke-width:2px
    linkStyle 2 stroke:#000000,stroke-width:2px
    linkStyle 3 stroke:#000000,stroke-width:2px
    linkStyle 4 stroke:#f59e0b,stroke-width:2px
    linkStyle 5 stroke:#f59e0b,stroke-width:2px
    linkStyle 6 stroke:#f59e0b,stroke-width:2px
    linkStyle 7 stroke:#10b981,stroke-width:2px
    linkStyle 8 stroke:#10b981,stroke-width:2px
    linkStyle 9 stroke:#10b981,stroke-width:2px
    linkStyle 10 stroke:#10b981,stroke-width:2px
    linkStyle 11 stroke:#ef4444,stroke-width:2px
    linkStyle 12 stroke:#ef4444,stroke-width:2px
    linkStyle 13 stroke:#8b5cf6,stroke-width:2px
    linkStyle 14 stroke:#8b5cf6,stroke-width:2px
    linkStyle 15 stroke:#8b5cf6,stroke-width:2px
    linkStyle 16 stroke:#ef4444,stroke-width:2px
```

#### 1. Creation — Every Run

- **Supervisor** fetches financial data via Alpha Vantage / yfinance and writes a fundamentals snapshot to the vault (`VaultWriter`).
- Analyst research findings (Bull + Bear) are written to the vault as timestamped, content-hashed Markdown documents.
- Each paragraph in a vault document receives a block ID (`^block-xxxxxxxx`), enabling granular citation.
- Insights are embedded via `text-embedding-3-small` (1536-dim) and stored in the `investment_memories` vector table with ticker-scoped metadata and a `source_priority` tier.

#### 2. Curation — Every Run

After the report is generated, the **Curator** performs housekeeping:

1. **Citation Manifest**: Scans the final report for inline citation references (`(See: file.md#^block-id)`) and resolves each to its source paragraph in the vault.
2. **Mark Cited Memories**: Any vector memory referenced in the manifest is marked `is_cited = true` in Supabase. Citations act as a survival signal — cited memories are protected from pruning.
3. **Delete Uncited Memories**: Vectors created within the active window (default 90 days) that were _not_ cited are deleted, preventing stale or irrelevant embeddings from accumulating.

#### 3. Consolidation — Monthly

When vault files for a ticker exceed `CONSOLIDATION_CUTOFF_DAYS` (default 90 days), the Curator triggers consolidation:

- Groups vault files by month (`{YYYY-MM}`).
- Feeds each month's documents to the **curator LLM**, which synthesizes them into a structured `{YYYY-MM}_synthesis.md` file stored in the vault.
- Archives the original source files (marks them `archived: true` in frontmatter).
- **Atomic vector swap**: deletes all granular vector memories for that month and inserts a single summary vector with `source_priority = 2` and `is_cited = true`.

This ensures the vector index stays lean while preserving the full audit trail in cold storage.

#### 4. Aggressive Pruning — On Storage Pressure

When the `investment_memories` table exceeds the aggressive threshold (default 80% of 500 MB):

- The Curator identifies the oldest monthly summary vectors.
- Merges them via the curator LLM into broader historical summaries.
- Retains only the **3 most recent months** of summary vectors, deleting the rest.
- This is an emergency mechanism — it only fires when storage approaches the configured `DB_LIMIT_MB`.

#### 5. Thesis Drift Tracking

Every run that produces a new valuation is compared against the **prior valuation** stored in Supabase:

- Extracts the old and new verdicts (e.g., "Strong Buy" → "Buy").
- Calculates the delta percentage between old and new probability-weighted expected values.
- Extracts key risk changes from the judge's reconciliation output.
- Records the drift as a row in the `thesis_drifts` table: `(ticker, old_verdict, new_verdict, old_ev, new_ev, delta_pct, key_changes)`.

On a ticker's very first run, no prior valuation exists, so drift recording is skipped.

#### 6. Deduplication

The Curator scans the vault for files with identical SHA-256 content hashes. Within each duplicate group, the **earliest** file (by date in filename) is preserved and the rest are removed. This prevents redundant research documents from bloating the vault when the same URL is scraped across multiple runs.

### Citation System

Value Brief uses a lightweight, file-based citation scheme to connect assertions in the final report back to their source paragraphs:

- **In the vault**: Every paragraph in a Markdown document is tagged with a block ID: `^block-a1b2c3d4`.
- **In the report**: Agents reference sources inline using the pattern `(See: 2026-05-02_a1b2c3d4.md#^block-a1b2c3d4)`.
- **At curation time**: `build_citation_manifest()` parses the report text, resolves each block ID to the corresponding vault paragraph via `resolve_citation()`, and the Curator uses this manifest to protect cited vector memories.

This creates a **provenance chain**: report assertion → block ID → vault paragraph → vector embedding, enabling auditability while keeping the vector layer relevant.

## Getting Started

### Prerequisites

| Requirement                      | Version                                 |
| :------------------------------- | :-------------------------------------- |
| Python                           | `>= 3.14`                               |
| [uv](https://docs.astral.sh/uv/) | latest                                  |
| Supabase project                 | PostgreSQL (transaction pooler enabled) |
| LLM provider account             | OpenRouter / Google / Others            |

### 1. Clone the repository

```bash
git clone https://github.com/your-username/valuebrief.git
cd valuebrief
```

### 2. Install dependencies

Value Brief uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install all dependencies
uv sync
```

This resolves dependencies from `pyproject.toml` and the pinned `uv.lock` file.

### 3. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env-example .env
```

Edit `.env` with the following values:

| Variable                     | Description                                                   |
| :--------------------------- | :------------------------------------------------------------ |
| `SUPABASE_CONNECTION_STRING` | Your Supabase PostgreSQL transaction-pooler connection string |
| `GOOGLE_API_KEY`             | Google Gemini API key (if using `langchain-google-genai`)     |
| `DEEPSEEK_API_KEY`           | DeepSeek API key                                              |
| `OPENROUTER_API_KEY`         | OpenRouter API key (used as the default provider)             |
| `ALPHAVANTAGE_API_KEY`       | Alpha Vantage key for financial data                          |
| `LANGSMITH_API_KEY`          | LangSmith key for tracing (optional but recommended)          |
| `*_PROVIDER` / `*_MODEL`     | Per-agent LLM provider and model overrides                    |

> **Tip:** Each agent (Bull, Bear, Judge, Supervisor, Report Generator, Valuation) has its own `_PROVIDER`, `_MODEL`, and `_TEMPERATURE` variable.
>
> Frontier models are strongly recommended for **Bull, Bear, and Judge analysts** for the best web search, reasoning, and tool calling capabilities. Success has been found using `qwen/qwen3.6-plus` with `0.2` temperature for excellent reasoning while remaining cost-effective.

### 4. Set up your portfolio

Create a `portfolio.json` file in the project root listing the tickers you want to track.

**International Stocks:** Use the [Yahoo Finance convention](https://help.yahoo.com/kb/finance-for-web/SLN2310.html) (`TICKER.EXCHANGE`) for all tickers.

```json
{
  "tickers": ["AAPL", "MZH.SI", "9988.HK", "RY.TO"]
}
```

#### Exchange Mappings (Alpha Vantage)

Since Alpha Vantage uses different exchange suffixes than Yahoo Finance, Value Brief uses a mapping file to translate them during data retrieval. You can customise these mappings in `exchange_mappings.json`:

```json
{
  "yahoo_to_alphavantage": {
    ".SI": ".SIN",
    ".HK": ".HKG",
    ".TO": ".TRT"
  }
}
```

> [!NOTE]
> Alpha Vantage often lacks fundamental data (`OVERVIEW`) for international stocks. In such cases, Value Brief automatically falls back to `yfinance` to ensure your report remains complete.

See `example-portfolio.json` for reference.

### 5. Initialise the database

The checkpointer and valuation tables are created automatically on first run via `AsyncPostgresSaver.setup()`. Ensure your Supabase connection string points to a **transaction-pooler** endpoint (port `6543`) with `autocommit` enabled.

### 6. Run Value Brief

```bash
# Analyse tickers from portfolio.json
uv run python src/main.py

# Override tickers inline
uv run python src/main.py --tickers NVDA TSM ASML

# Point to a custom portfolio file
uv run python src/main.py --portfolio my-watchlist.json
```

Generated Markdown reports are written to the `logs/` directory.

### Scheduling (optional)

To run Value Brief as a daily digest, add a cron job:

```bash
# Example: run at 07:00 every day
0 7 * * * cd /path/to/valuebrief && uv run python src/main.py >> logs/cron.log 2>&1
```

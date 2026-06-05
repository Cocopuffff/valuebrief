---
name: sec-filings
description: Use this skill when a research task asks for recent SEC filings, 10-Ks, 10-Qs, 8-Ks, annual reports, quarterly reports, or filing freshness checks for a US-listed company.
---

# SEC Filings Discovery

Use official SEC filing data before relying on third-party summaries.

Workflow:

1. Call `get_sec_filings` with the ticker and relevant forms, usually `["10-K", "10-Q", "8-K"]`.
2. Prefer the most recent 10-K and most recent 10-Q. Include earnings releases or 8-Ks only when they materially update the financial picture.
3. For each filing, record form type, filing date, report date, accession number, and SEC URL.
4. Call `scrape_website` on the most relevant SEC filing URLs when the task needs source text.
5. Cite SEC URLs directly in factual claims.

Do not infer filing freshness from search snippets when official SEC records are available.

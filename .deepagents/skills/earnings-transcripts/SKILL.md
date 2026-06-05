---
name: earnings-transcripts
description: Use this skill when a research task asks for earnings calls, management commentary, prepared remarks, Q&A, transcripts, or investor-relations event materials.
---

# Earnings Transcript Discovery

Use transcript and investor-relations sources to understand management commentary and near-term business changes.

Workflow:

1. Call `discover_earnings_call_transcripts` with ticker and company.
2. Prefer company investor-relations pages, earnings releases, event pages, and official webcast/transcript pages.
3. If company pages are unavailable, use reputable transcript aggregators, but label them as third-party sources.
4. Call `scrape_website` on promising URLs before using their content.
5. Separate management claims from analyst interpretation.

When a transcript cannot be found, state that clearly and use available earnings releases or official shareholder materials as fallback evidence.

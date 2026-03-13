GV2-EDGE — Biotech Catalyst Radar

Specification Document for Claude Code

Objective

Build a biotech catalyst detection system for GV2-EDGE capable of identifying biotech stocks with a high probability of becoming momentum runners or top gainers.

The system must automatically:

1. Monitor biotech catalysts (FDA, clinical trials, filings, news)
2. Map Drug → Company → Ticker
3. Filter the biotech universe to focus on tradable, liquid small-caps
4. Combine catalysts with market data signals
5. Generate a Biotech Opportunity Score

The output must produce actionable alerts, not raw data.

---

1. System Architecture Overview

Data Sources
   │
   ├── FDA announcements
   ├── Clinical trials registry
   ├── SEC filings
   ├── News feeds
   └── Market data (volume, float, price)
        │
        ▼
Data Normalization Layer
        │
Drug → Company → Ticker Resolver
        │
Catalyst Classification Engine
        │
Market Signal Integration
        │
Biotech Opportunity Score
        │
Alert Engine

---

2. Biotech Universe Filtering

Monitoring all biotech companies is inefficient.

We first create a filtered universe of tradable biotech stocks.

2.1 Exchange Filter

Allowed exchanges:

- NASDAQ
- NYSE
- AMEX

Excluded:

- OTC
- Pink sheets

Reason:
OTC stocks have unreliable liquidity and are often manipulated.

---

2.2 Price Filter

1$ < Price < 20$

Rationale:

- Stocks under $20 move faster
- Still liquid enough for trading

---

2.3 Liquidity Filter

Average Daily Volume > 200,000 shares

Purpose:

- Remove illiquid biotech stocks
- Focus on tradable setups

---

2.4 Float Filter

Float < 100M
Ideal: Float < 50M
High momentum: Float < 20M

Lower float increases the probability of:

- squeezes
- fast momentum moves

---

2.5 Resulting Watchlist

After filtering:

200 – 350 biotech stocks

This list becomes the core universe monitored by the radar.

---

3. Catalyst Data Sources

The radar must ingest biotech catalysts from multiple sources.

3.1 Clinical Trials Registry

Data extracted:

- drug name
- sponsor company
- trial phase
- indication
- completion date

Important phases:

Phase 2
Phase 3

Ignore:

Phase 1
Early research

---

3.2 FDA Announcements

Important events:

PDUFA date
Drug approval
Advisory committee meeting
FDA rejection

These events can trigger large price moves.

---

3.3 SEC Filings

Relevant filings:

8-K → partnerships, licensing deals
S-1 / 424B → dilution / offerings
13D / 13G → institutional accumulation
Form 4 → insider buying

These filings can precede biotech momentum.

---

3.4 Press Releases

Typical catalysts:

drug partnership
licensing deal
trial success
trial failure
strategic collaboration

Press releases often appear before financial news platforms report them.

---

4. Drug → Company → Ticker Resolver

Biotech data often references drug names, not tickers.

Example:

Drug: ABC-101
Sponsor: Vertex Pharmaceuticals

But trading systems need:

Ticker: VRTX

---

4.1 Resolver Pipeline

Raw Catalyst Data
      │
Extract Drug Name
      │
Extract Sponsor Company
      │
Normalize Company Name
      │
Match Company → Ticker
      │
Store Catalyst Event

---

4.2 Normalization

Company names may appear in different forms:

Vertex Pharmaceuticals
Vertex Pharmaceuticals Inc
Vertex Pharma

Normalization rules:

- remove "Inc", "Corp", "Ltd"
- lowercase
- trim spaces

---

4.3 Fuzzy Matching

If direct match fails:

Use fuzzy string matching to identify the closest company.

Example libraries:

rapidfuzz
fuzzywuzzy

---

4.4 Internal Mapping Database

Maintain an internal table:

Drug → Company → Ticker

Example:

Drug| Company| Ticker
ABC-101| Vertex Pharmaceuticals| VRTX
BNT162b2| Pfizer| PFE
REGN-COV2| Regeneron| REGN

This table improves resolution speed over time.

---

5. Catalyst Classification

Each biotech catalyst must be classified by impact level.

Impact Levels

Event Type| Impact
Phase 1| Low
Phase 2| Medium
Phase 3| High
PDUFA| Very High
FDA Approval| Maximum

Example scoring:

Phase 2 → 0.5
Phase 3 → 0.8
PDUFA → 0.9
FDA Approval → 1.0

---

6. Catalyst Timing Score

The closer the catalyst date, the stronger the market reaction.

days_to_event = event_date - today

Scoring:

< 3 days  → 1.0
< 7 days  → 0.8
< 30 days → 0.5

---

7. Market Data Integration

Catalysts alone are not enough.

We combine catalysts with market signals.

Important metrics:

relative_volume
price_momentum
float
market_cap
short_interest

---

Key Signals

Relative Volume

Relative Volume > 2

Indicates speculative activity.

---

Float

Float < 30M

High potential for fast moves.

---

Short Interest

Short interest > 15%

Possible squeeze scenario.

---

8. Biotech Opportunity Score

Final score formula:

Biotech Score =
Event Importance
× Catalyst Proximity
× Relative Volume
× Float Factor
× Short Interest Factor

Example:

Phase 3 catalyst → 0.8
Days to event → 0.9
Relative volume → 3
Float factor → 0.8

Result:

High Opportunity Score

---

9. Alert Generation

Example alert format:

BIOTECH ALERT

Ticker: XYZ
Drug: ABC-101
Event: Phase 3 Results
Days to Event: 4
Float: 14M
Relative Volume: 3.8

Opportunity Score: 0.86

---

10. Update Frequency

System updates:

Watchlist update → weekly
Catalyst ingestion → daily
Market signals → real time

---

11. Final Pipeline

Biotech Universe Filter
        │
Catalyst Data Ingestion
        │
Drug → Company → Ticker Resolver
        │
Catalyst Classification
        │
Market Data Integration
        │
Biotech Opportunity Score
        │
Alert Engine

---

12. Key Principle

The system must not flood the user with raw data.

It must deliver:

high probability biotech opportunities

The radar's goal is to detect:

biotech stocks capable of 50% – 200% moves
around clinical catalysts

---

End of Specification

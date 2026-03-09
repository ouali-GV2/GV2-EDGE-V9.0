GV2-EDGE — SEC EDGAR Ticker Resolution

Specification for Claude Code

Objective

Fix the issue where GV2-EDGE displays SEC filings without tickers.

SEC EDGAR data does not include stock tickers directly.
Instead, it uses a unique company identifier called CIK (Central Index Key).

The system must therefore implement a CIK → Ticker resolution layer before processing filings.

---

1. Problem Overview

Current behavior:

SEC filings are ingested with data similar to:

{
 "cik": "0001682852",
 "form": "8-K",
 "filingDate": "2026-05-18"
}

But the trading engine requires:

{
 "ticker": "MRNA",
 "form": "8-K",
 "filingDate": "2026-05-18"
}

Because market data providers (price, volume, etc.) operate using tickers, not CIK identifiers.

Therefore, the system must convert CIK → ticker.

---

2. Official Mapping Source

The U.S. SEC publishes a public file containing the mapping:

https://www.sec.gov/files/company_tickers.json

This dataset contains:

- ticker
- company name
- CIK identifier

Example entry:

{
 "ticker": "MRNA",
 "cik_str": 1682852,
 "title": "Moderna Inc."
}

---

3. Data Normalization

SEC CIK values in filings are typically formatted with 10 digits.

Example:

CIK from filing: 1682852
Normalized CIK: 0001682852

Normalization rule:

CIK = str(cik_str).zfill(10)

---

4. Mapping Table Creation

At system startup, build a dictionary:

cik_to_ticker = {
  "0001682852": "MRNA",
  "0000320193": "AAPL"
}

Structure:

Key   → CIK (10 digit string)
Value → Ticker

This table will be used to resolve tickers for all incoming filings.

---

5. System Pipeline

The SEC ingestion pipeline must follow this structure:

SEC EDGAR Feed
      │
      ▼
Extract Filing
      │
      ▼
Extract CIK
      │
      ▼
CIK → Ticker Resolver
      │
      ▼
Normalized Filing Event
      │
      ▼
Market Data Integration
      │
      ▼
Catalyst Detection

---

6. Example Conversion

Raw Filing

{
 "cik": "0001682852",
 "form": "8-K",
 "filingDate": "2026-05-18"
}

After Resolver

{
 "ticker": "MRNA",
 "company": "Moderna Inc.",
 "form": "8-K",
 "filingDate": "2026-05-18"
}

Now GV2-EDGE can:

- fetch market price
- analyze volume
- trigger alerts

---

7. Implementation Example

Example Python code:

import requests

url = "https://www.sec.gov/files/company_tickers.json"
data = requests.get(url).json()

cik_to_ticker = {}

for item in data.values():
    cik = str(item["cik_str"]).zfill(10)
    ticker = item["ticker"]
    cik_to_ticker[cik] = ticker

def resolve_ticker(cik):
    return cik_to_ticker.get(cik)

---

8. Performance Optimization

To avoid latency:

The mapping table must be loaded once at system startup.

Recommended approach:

System Start
      │
Download ticker mapping
      │
Build dictionary
      │
Store in memory

This allows constant-time lookup:

O(1) lookup for every filing

---

9. Edge Cases

Possible issues:

Missing ticker

Some filings belong to:

- private companies
- subsidiaries
- shell entities

Solution:

if ticker not found:
   ignore filing

or mark as:

unmapped_company

---

10. Final Output Format

After normalization, every SEC event stored by GV2-EDGE must follow:

{
 "ticker": "XYZ",
 "company": "Example Biotech Inc.",
 "form": "8-K",
 "filingDate": "2026-06-01",
 "source": "sec_edgar"
}

This ensures compatibility with:

- market data feeds
- catalyst scoring engines
- alert systems

---

11. Expected Result

After implementing the resolver:

GV2-EDGE will:

- correctly display tickers for SEC filings
- integrate filings with market data
- detect catalyst-driven opportunities

This resolves the missing ticker issue in the SEC EDGAR pipeline.

---

End of Document

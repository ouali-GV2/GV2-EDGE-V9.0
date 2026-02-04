import json
from datetime import datetime

from config import GROK_API_KEY
from utils.api_guard import safe_post
from utils.logger import get_logger
from utils.data_validator import validate_event

logger = get_logger("NLP_EVENT_PARSER")

GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"


SYSTEM_PROMPT = """
You are a financial event extraction engine.

From the given text, extract ONLY real bullish catalysts for US stocks.

Event types allowed:
- FDA_APPROVAL
- FDA_TRIAL_RESULT
- MERGER_ACQUISITION
- EARNINGS_BEAT
- ANALYST_UPGRADE
- MAJOR_CONTRACT
- PARTNERSHIP
- GUIDANCE_RAISE
- BUYOUT_RUMOR

For each event return JSON list:

[
 {
  "ticker": "XYZ",
  "type": "EVENT_TYPE",
  "impact": 0.0 to 1.0,
  "date": "YYYY-MM-DD",
  "summary": "short explanation"
 }
]

Only output valid JSON.
No text outside JSON.
"""


def call_grok(text):
    payload = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    r = safe_post(GROK_ENDPOINT, json=payload, headers=headers)
    return r.json()


def parse_events_from_text(text):
    try:
        response = call_grok(text)

        content = response["choices"][0]["message"]["content"]

        events = json.loads(content)

        clean_events = []

        for e in events:
            if validate_event(e):
                clean_events.append(e)

        logger.info(f"NLP extracted {len(clean_events)} events")

        return clean_events

    except Exception as e:
        logger.error(f"NLP parse failed: {e}")
        return []


# ============================
# Batch helper
# ============================

def parse_many_texts(texts):
    all_events = []

    for t in texts:
        ev = parse_events_from_text(t)
        all_events.extend(ev)

    return all_events


if __name__ == "__main__":
    sample = """
    TCGL announces FDA approval for its new cancer drug.
    FEED receives major $200M contract with US government.
    """

    print(parse_events_from_text(sample))

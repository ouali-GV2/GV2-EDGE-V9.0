from utils.logger import get_logger
from utils.api_guard import safe_get
from utils.cache import Cache
from utils.time_utils import is_premarket

from config import FINNHUB_API_KEY, PM_MIN_VOLUME, USE_IBKR_DATA

from datetime import datetime

logger = get_logger("PM_SCANNER")

cache = Cache(ttl=30)

FINNHUB_QUOTE = "https://finnhub.io/api/v1/quote"

# IBKR connector with lazy loading
_ibkr_connector = None
_ibkr_init_attempted = False


def _get_ibkr_connector():
    """
    Lazy loading for IBKR connector.
    Only initializes once, caches result (including failures).
    """
    global _ibkr_connector, _ibkr_init_attempted

    if _ibkr_init_attempted:
        return _ibkr_connector

    _ibkr_init_attempted = True

    if not USE_IBKR_DATA:
        return None

    try:
        from src.ibkr_connector import get_ibkr
        _ibkr_connector = get_ibkr()
        if _ibkr_connector and _ibkr_connector.connected:
            logger.info("PM Scanner: IBKR connector initialized")
        else:
            logger.debug("PM Scanner: IBKR not connected, using Finnhub fallback")
            _ibkr_connector = None
    except ImportError:
        logger.debug("PM Scanner: IBKR connector not available")
        _ibkr_connector = None
    except Exception as e:
        logger.debug(f"PM Scanner: IBKR init failed: {e}")
        _ibkr_connector = None

    return _ibkr_connector


# ============================
# Fetch live quote (IBKR or Finnhub)
# ============================

def fetch_quote(ticker):
    """Fetch quote from IBKR (if available) or Finnhub fallback"""

    # Try IBKR first (lazy load)
    ibkr = _get_ibkr_connector()
    if ibkr and ibkr.connected:
        try:
            quote = ibkr.get_quote(ticker)
            if quote:
                # Convert IBKR format to Finnhub-like format
                return {
                    "o": quote.get("open"),
                    "h": quote.get("high"),
                    "l": quote.get("low"),
                    "c": quote.get("last"),
                    "pc": quote.get("close"),  # previous close
                    "v": quote.get("volume")
                }
        except Exception as e:
            logger.debug(f"IBKR quote failed for {ticker}: {e}")
    
    # Fallback to Finnhub
    params = {
        "symbol": ticker,
        "token": FINNHUB_API_KEY
    }

    r = safe_get(FINNHUB_QUOTE, params=params)
    return r.json()


# ============================
# PM Metrics
# ============================

def compute_pm_metrics(ticker):
    if not is_premarket():
        return None

    cached = cache.get(f"pm_{ticker}")
    if cached:
        return cached

    try:
        q = fetch_quote(ticker)

        pm_open = q.get("o")
        pm_high = q.get("h")
        pm_low = q.get("l")
        last = q.get("c")
        volume = q.get("v", 0)

        if not pm_open or not last:
            return None

        gap_pct = (last - pm_open) / pm_open

        momentum = (pm_high - pm_low) / pm_low if pm_low else 0

        liquid = volume >= PM_MIN_VOLUME

        metrics = {
            "gap_pct": gap_pct,
            "pm_high": pm_high,
            "pm_low": pm_low,
            "pm_momentum": momentum,
            "pm_volume": volume,
            "pm_liquid": liquid
        }

        cache.set(f"pm_{ticker}", metrics)

        return metrics

    except Exception as e:
        logger.error(f"PM scan error {ticker}: {e}")
        return None


# ============================
# Batch scan
# ============================

def scan_premarket(tickers, limit=None):
    results = {}

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        m = compute_pm_metrics(t)
        if m:
            results[t] = m

    logger.info(f"PM scanned {len(results)} tickers")

    return results


if __name__ == "__main__":
    print(compute_pm_metrics("AAPL"))

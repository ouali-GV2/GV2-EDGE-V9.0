import json
import os

from utils.logger import get_logger
from utils.cache import Cache

from src.event_engine.event_hub import get_events_by_ticker
from src.feature_engine import compute_features
from src.pm_scanner import compute_pm_metrics

# NEW: Import advanced intelligence modules
from src.historical_beat_rate import get_earnings_probability
from src.social_buzz import get_buzz_signal

# Options flow (optional - requires IBKR subscription)
try:
    from src.options_flow_ibkr import get_options_flow_score
    OPTIONS_FLOW_AVAILABLE = True
except:
    OPTIONS_FLOW_AVAILABLE = False

# Extended hours boost (optional - requires IBKR subscription)
try:
    from src.extended_hours_quotes import get_extended_hours_boost
    EXTENDED_HOURS_AVAILABLE = True
except:
    EXTENDED_HOURS_AVAILABLE = False

from config import ADVANCED_MONSTER_WEIGHTS  # ✅ Use optimized weights

logger = get_logger("MONSTER_SCORE")

cache = Cache(ttl=30)

WEIGHT_FILE = "data/monster_score_weights.json"


# ============================
# Load weights (auto-tuning ready)
# ============================

def load_weights():
    """
    Load weights from file or use ADVANCED_MONSTER_WEIGHTS as default
    
    Priority:
    1. Custom weights from weekly_audit auto-tuning (if exists)
    2. ADVANCED_MONSTER_WEIGHTS from config (optimized)
    """
    if os.path.exists(WEIGHT_FILE):
        try:
            with open(WEIGHT_FILE) as f:
                custom_weights = json.load(f)
                logger.info("Using custom auto-tuned weights")
                return custom_weights
        except:
            pass
    
    # Use optimized weights from config
    logger.info("Using ADVANCED_MONSTER_WEIGHTS from config")
    return ADVANCED_MONSTER_WEIGHTS.copy()


def save_weights(w):
    os.makedirs("data", exist_ok=True)
    with open(WEIGHT_FILE, "w") as f:
        json.dump(w, f, indent=2)


# ============================
# Normalize helpers
# ============================

def clamp(x, lo=0, hi=1):
    return max(lo, min(hi, x))


def normalize(x, scale=1):
    try:
        return clamp(x / scale)
    except:
        return 0


# ============================
# Score computation
# ============================

def compute_event_score(ticker):
    events = get_events_by_ticker(ticker)

    if not events:
        return 0

    return max(e["boosted_impact"] for e in events)


def compute_monster_score(ticker, use_advanced=True):
    """
    Compute Monster Score with optional advanced patterns
    
    Args:
        ticker: stock ticker
        use_advanced: if True, use pattern_analyzer and pm_transition
    
    Returns: dict with score and components
    """
    cached = cache.get(f"score_{ticker}")
    if cached:
        return cached

    weights = load_weights()

    # ===== EVENTS =====
    event_score = compute_event_score(ticker)

    # ===== FEATURES =====
    feats = compute_features(ticker, include_advanced=use_advanced)

    if not feats:
        return None

    momentum = normalize(abs(feats["momentum"]), 0.2)
    volume = normalize(feats["volume_spike"], 5)
    squeeze = normalize(feats["squeeze_proxy"], 10)

    # ===== PREMARKET =====
    pm = compute_pm_metrics(ticker)
    
    # ===== ADVANCED PATTERNS (if enabled) =====
    pattern_score = 0
    pm_transition_score = 0
    
    if use_advanced:
        try:
            # Get full dataframe for pattern analysis
            from src.feature_engine import fetch_candles
            df = fetch_candles(ticker)
            
            if df is not None and len(df) >= 20:
                # Pattern Analyzer
                from src.pattern_analyzer import compute_pattern_score as calc_pattern
                pattern_data = calc_pattern(ticker, df, pm)
                pattern_score = pattern_data.get("pattern_score", 0)
                
                # PM Transition Analyzer
                if pm:
                    from src.pm_transition import compute_pm_transition_score
                    transition_data = compute_pm_transition_score(ticker, df, pm)
                    pm_transition_score = transition_data.get("pm_transition_score", 0)
        
        except Exception as e:
            logger.warning(f"Advanced scoring error for {ticker}: {e}")
            # Continue with basic scoring
    
    # ===== INTELLIGENCE MODULES (V4 - INSTITUTIONAL GRADE) =====
    beat_rate_boost = 0
    social_buzz_boost = 0
    options_boost = 0
    extended_hours_boost = 0
    
    try:
        # Check if event is earnings-related
        events = get_events_by_ticker(ticker)
        has_earnings = any("earning" in e.get("type", "").lower() for e in events)
        
        if has_earnings:
            # Historical beat rate analysis
            earnings_prob = get_earnings_probability(ticker)
            
            if earnings_prob and earnings_prob > 0.6:
                # High probability of beat = boost score
                beat_rate_boost = (earnings_prob - 0.5) * 0.4  # Max +0.2 boost
                logger.info(f"{ticker} beat probability: {earnings_prob*100:.0f}% → +{beat_rate_boost:.2f}")
        
        # Social buzz detection
        buzz = get_buzz_signal(ticker)
        buzz_score = buzz.get("buzz_score", 0)
        
        if buzz_score > 0.5:
            # Abnormal buzz = boost
            social_buzz_boost = (buzz_score - 0.5) * 0.2  # Max +0.1 boost
            logger.info(f"{ticker} buzz spike: {buzz_score:.2f} → +{social_buzz_boost:.2f}")
        
        # Options flow (if available)
        if OPTIONS_FLOW_AVAILABLE:
            options_score, options_details = get_options_flow_score(ticker)
            
            if options_score > 0.6:
                options_boost = (options_score - 0.5) * 0.2  # Max +0.1 boost
                logger.info(f"{ticker} options flow: {options_score:.2f} → +{options_boost:.2f}")
        
        # Extended hours boost (if available - after-hours/pre-market gaps)
        if EXTENDED_HOURS_AVAILABLE:
            eh_boost, eh_details = get_extended_hours_boost(ticker)
            
            if eh_boost > 0:
                extended_hours_boost = eh_boost  # Max +0.22
                logger.info(f"{ticker} extended hours: +{eh_boost:.2f} ({eh_details.get('session', 'N/A')})")
    
    except Exception as e:
        logger.warning(f"Intelligence modules error for {ticker}: {e}")

    # ===== FINAL SCORE (V5 WITH INTELLIGENCE) =====
    # Base score from technical/fundamental
    base_score = (
        weights.get("event", 0.30) * event_score +
        weights.get("volume", 0.20) * volume +
        weights.get("pattern", 0.20) * pattern_score +
        weights.get("pm_transition", 0.15) * pm_transition_score +
        weights.get("momentum", 0.10) * momentum +
        weights.get("squeeze", 0.05) * squeeze
    )
    
    # Add intelligence boosts
    total_boost = beat_rate_boost + social_buzz_boost + options_boost + extended_hours_boost
    final_score = base_score + total_boost
    
    # Clamp 0-1
    final_score = clamp(final_score)

    details = {
        "monster_score": final_score,
        "base_score": base_score,
        "intelligence_boost": total_boost,
        "components": {
            "event": event_score,
            "volume": volume * weights.get("volume", 0.20),
            "momentum": momentum * weights.get("momentum", 0.10),
            "squeeze": squeeze * weights.get("squeeze", 0.05),
            "pattern": pattern_score * weights.get("pattern", 0.20),
            "pm_transition": pm_transition_score * weights.get("pm_transition", 0.15),
            "beat_rate_boost": beat_rate_boost,
            "social_buzz_boost": social_buzz_boost,
            "options_boost": options_boost,
            "extended_hours_boost": extended_hours_boost
        }
    }

    cache.set(f"score_{ticker}", details)

    return details


# ============================
# Batch helper
# ============================

def score_many(tickers, limit=None):
    results = {}

    for i, t in enumerate(tickers):
        if limit and i >= limit:
            break

        s = compute_monster_score(t)
        if s:
            results[t] = s

    logger.info(f"Scored {len(results)} tickers")

    return results


if __name__ == "__main__":
    print(compute_monster_score("AAPL"))

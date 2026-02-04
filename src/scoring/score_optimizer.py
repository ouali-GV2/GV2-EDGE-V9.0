import json
import os

from utils.logger import get_logger

WEIGHT_FILE = "data/monster_score_weights.json"
PERF_FILE = "data/backtest_reports/performance_attribution.json"

logger = get_logger("SCORE_OPTIMIZER")


# ============================
# Default weights (fallback)
# ============================

DEFAULT_WEIGHTS = {
    "event": 0.35,
    "momentum": 0.2,
    "volume": 0.15,
    "vwap": 0.1,
    "squeeze": 0.1,
    "pm_gap": 0.1
}


# ============================
# Load / Save
# ============================

def load_weights():
    if os.path.exists(WEIGHT_FILE):
        try:
            with open(WEIGHT_FILE) as f:
                return json.load(f)
        except:
            return DEFAULT_WEIGHTS.copy()

    return DEFAULT_WEIGHTS.copy()


def save_weights(weights):
    os.makedirs("data", exist_ok=True)

    with open(WEIGHT_FILE, "w") as f:
        json.dump(weights, f, indent=2)


# ============================
# Load performance attribution
# ============================

def load_performance_data():
    if not os.path.exists(PERF_FILE):
        logger.warning("No performance attribution file yet")
        return None

    try:
        with open(PERF_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Perf load error: {e}")
        return None


# ============================
# Safe optimizer logic
# ============================

def optimize_weights(learning_rate=0.05, max_change=0.1):
    """
    Small nudges only (safe mode)
    """

    weights = load_weights()
    perf = load_performance_data()

    if not perf:
        logger.warning("No perf data - skipping optimization")
        return weights

    # perf example:
    # {"event": 0.12, "momentum": 0.05, ...} average return contribution

    total_perf = sum(abs(v) for v in perf.values())

    if total_perf == 0:
        logger.warning("Zero perf data")
        return weights

    new_weights = weights.copy()

    for component, contribution in perf.items():

        if component not in weights:
            continue

        direction = 1 if contribution > 0 else -1

        delta = learning_rate * direction * abs(contribution) / total_perf

        delta = max(-max_change, min(max_change, delta))

        new_weights[component] += delta

        # prevent negative or crazy
        new_weights[component] = max(0.01, min(1.0, new_weights[component]))

    # normalize weights to sum to 1
    s = sum(new_weights.values())
    for k in new_weights:
        new_weights[k] /= s

    save_weights(new_weights)

    logger.info(f"Weights optimized: {new_weights}")

    return new_weights


# ============================
# Manual trigger
# ============================

if __name__ == "__main__":
    print("Before:", load_weights())
    print("After:", optimize_weights())

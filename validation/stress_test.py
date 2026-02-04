import os
import pandas as pd
import numpy as np

from utils.logger import get_logger

BACKTEST_DIR = "data/backtest_reports"

logger = get_logger("STRESS_TEST")


# ============================
# Load all backtest trades
# ============================

def load_backtests():
    dfs = []

    for f in os.listdir(BACKTEST_DIR):
        if f.startswith("backtest_") and f.endswith(".csv"):
            df = pd.read_csv(f"{BACKTEST_DIR}/{f}")
            dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


# ============================
# Drawdown calculation
# ============================

def compute_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak
    return drawdown.min()


# ============================
# Stress scenarios
# ============================

def run_stress_test():

    trades = load_backtests()

    if trades is None or trades.empty:
        logger.warning("No backtest data found")
        return None

    pnl = trades["pnl"].values

    # Equity curve
    equity = np.cumsum(pnl)

    max_dd = compute_drawdown(pd.Series(equity))

    # Worst losing streak
    losses = pnl < 0
    max_streak = 0
    current = 0

    for l in losses:
        if l:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    # Shock simulation (random reorder)
    simulations = []

    for _ in range(500):
        shuffled = np.random.permutation(pnl)
        eq = np.cumsum(shuffled)
        dd = compute_drawdown(pd.Series(eq))
        simulations.append(dd)

    shock_dd_95 = np.percentile(simulations, 95)

    results = {
        "max_drawdown_real": round(float(max_dd), 2),
        "worst_losing_streak": int(max_streak),
        "shock_drawdown_95pct": round(float(shock_dd_95), 2),
        "total_trades": len(pnl)
    }

    out_path = f"{BACKTEST_DIR}/stress_test_results.json"

    pd.Series(results).to_json(out_path, indent=2)

    logger.info(f"Stress test saved: {out_path}")
    logger.info(f"Results: {results}")

    return results


# ============================
# CLI
# ============================

if __name__ == "__main__":
    print(run_stress_test())

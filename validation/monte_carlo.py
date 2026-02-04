import os
import numpy as np
import pandas as pd

from utils.logger import get_logger

BACKTEST_DIR = "data/backtest_reports"

logger = get_logger("MONTE_CARLO")


# ============================
# Load backtest trades
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
# Monte Carlo simulation
# ============================

def run_monte_carlo(n_simulations=2000):

    trades = load_backtests()

    if trades is None or trades.empty:
        logger.warning("No backtest data found")
        return None

    pnl = trades["pnl"].values

    n_trades = len(pnl)

    final_equities = []
    max_drawdowns = []

    for _ in range(n_simulations):

        shuffled = np.random.choice(pnl, size=n_trades, replace=True)
        equity = np.cumsum(shuffled)

        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak

        final_equities.append(equity[-1])
        max_drawdowns.append(drawdown.min())

    results = {
        "simulations": n_simulations,
        "avg_final_equity": round(float(np.mean(final_equities)), 2),
        "median_final_equity": round(float(np.median(final_equities)), 2),
        "best_case": round(float(np.max(final_equities)), 2),
        "worst_case": round(float(np.min(final_equities)), 2),
        "avg_max_drawdown": round(float(np.mean(max_drawdowns)), 2),
        "worst_drawdown": round(float(np.min(max_drawdowns)), 2),
        "ruin_probability": round(
            float(np.mean(np.array(final_equities) <= 0)), 3
        )
    }

    out_path = f"{BACKTEST_DIR}/monte_carlo_results.json"

    pd.Series(results).to_json(out_path, indent=2)

    logger.info(f"Monte Carlo saved: {out_path}")
    logger.info(f"Results: {results}")

    return results


# ============================
# CLI
# ============================

if __name__ == "__main__":
    print(run_monte_carlo())

"""Command-line training and reproducible result export."""
import argparse
import json
from pathlib import Path
import pandas as pd
from market_predictor.data import clean_prices, demo_prices, download_prices
from market_predictor.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--ticker")
    source.add_argument("--csv", type=Path)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=str(pd.Timestamp.today().date()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--capital", type=float, default=10000.)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--max-orders", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error("epochs must be between 1 and 100")
    try:
        frame = (demo_prices() if args.demo else clean_prices(pd.read_csv(args.csv))
                 if args.csv else download_prices(args.ticker, args.start, args.end))
        result = run_experiment(frame, args.lookback, args.epochs, args.horizon,
                                args.capital, args.fee, args.max_orders)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    result.model.save(args.output / "lstm.keras")
    for name, table in [("predictions", result.evaluation), ("metrics", result.scores),
                         ("forecast", result.future), ("strategy", result.plan.ledger),
                         ("backtest", result.backtest)]:
        table.to_csv(args.output / f"{name}.csv")
    metadata = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "scaler": {"minimum": result.windows.minimum, "span": result.windows.span},
                "split": {"train_target_count": result.windows.train_end,
                          "validation_target_end": result.windows.validation_end},
                "history": result.history, "summary": result.summary}
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result.scores.to_string())
    print(json.dumps(result.summary, indent=2))
    print(f"Saved model, scaler metadata, and CSV reports to {args.output.resolve()}")


if __name__ == "__main__":
    main()

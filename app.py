"""
UTAYA BTC HILO - CLI Backtest / Paper Trading
判定方式: エントリー価格から ±50 USD の先着
配当: 勝ち +88%、負け -100%
注意: 1分足による近似検証。実取引は行わない。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


SYMBOL = "BTC-USD"
TARGET_USD = 50.0
PAYOUT_RATE = 0.88
STAKE_USD = 5.0
HOLD_BARS = 1
DATA_DIR = Path("utaya_data")
DATA_DIR.mkdir(exist_ok=True)


def get_data(period: str = "7d") -> pd.DataFrame:
    """1分足データを取得し、OHLCを整える。"""
    df = yf.download(
        SYMBOL,
        period=period,
        interval="1m",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            "価格データを取得できませんでした。"
            "時間をおいて再実行してください。"
        )

    # yfinanceのバージョンによるMultiIndex列に対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"必要な列がありません: {missing}")

    df = df[required].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if len(df) < 100:
        raise RuntimeError(
            f"データ不足です。取得件数: {len(df)}"
        )

    return df


def make_signal(df: pd.DataFrame, i: int) -> str:
    """
    i番目の足の終値までを使って方向を判定する。
    未来の価格は参照しない。
    """
    close = df["Close"].iloc[: i + 1]

    if len(close) < 30:
        return "SKIP"

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])

    sma10 = float(close.tail(10).mean())
    sma30 = float(close.tail(30).mean())

    # 直近の値動きが小さすぎる場合は見送る
    momentum = last - prev
    deviation = last - sma10

    if momentum > 0 and last > sma30 and deviation > 0:
        return "HIGH"

    if momentum < 0 and last < sma30 and deviation < 0:
        return "LOW"

    return "SKIP"


def judge_first_touch(
    entry: float,
    direction: str,
    future: pd.DataFrame,
    target: float = TARGET_USD,
) -> tuple[str, str]:
    """
    上下の目標価格への先着をOHLCで近似判定する。

    1分足の同一バーで上下両方に触れた場合、
    先着順が不明なのでAMBIGUOUSとする。
    """
    upper = entry + target
    lower = entry - target

    for _, candle in future.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper
        hit_down = low <= lower

        if hit_up and hit_down:
            return "AMBIGUOUS", "同一足で上下両方に到達"

        if hit_up:
            result = "WIN" if direction == "HIGH" else "LOSS"
            return result, "上目標に到達"

        if hit_down:
            result = "WIN" if direction == "LOW" else "LOSS"
            return result, "下目標に到達"

    return "NO_TOUCH", "判定期間内に目標未到達"


def run_backtest(
    df: pd.DataFrame,
    target: float,
    stake: float,
    payout: float,
    hold_bars: int,
) -> pd.DataFrame:
    """時系列順に先着判定を行う。"""
    records = []

    # 予測にはi番目まで、判定にはその後の足だけを使用
    for i in range(30, len(df) - hold_bars):
        direction = make_signal(df, i)

        if direction == "SKIP":
            continue

        entry = float(df["Close"].iloc[i])
        future = df.iloc[i + 1 : i + 1 + hold_bars]

        result, reason = judge_first_touch(
            entry, direction, future, target
        )

        if result == "WIN":
            pnl = stake * payout
        elif result == "LOSS":
            pnl = -stake
        else:
            pnl = 0.0

        records.append({
            "signal_time": df.index[i],
            "entry_price": entry,
            "direction": direction,
            "upper_target": entry + target,
            "lower_target": entry - target,
            "result": result,
            "reason": reason,
            "pnl_usd": pnl,
        })

    return pd.DataFrame(records)


def print_report(results: pd.DataFrame) -> None:
    """勝率・損益・連敗を表示する。"""
    if results.empty:
        print("シグナルがありません。")
        return

    wins = int((results["result"] == "WIN").sum())
    losses = int((results["result"] == "LOSS").sum())
    ambiguous = int((results["result"] == "AMBIGUOUS").sum())
    no_touch = int((results["result"] == "NO_TOUCH").sum())

    decided = wins + losses
    win_rate = wins / decided * 100 if decided else 0.0
    pnl = float(results["pnl_usd"].sum())

    # 勝ち・負けが確定した取引だけで最大連敗を計算
    max_losing_streak = 0
    current_streak = 0

    for result in results["result"]:
        if result == "LOSS":
            current_streak += 1
            max_losing_streak = max(
                max_losing_streak, current_streak
            )
        elif result == "WIN":
            current_streak = 0

    print("\n========== UTAYA BTC HILO ==========")
    print(f"総シグナル数       : {len(results)}")
    print(f"勝ち               : {wins}")
    print(f"負け               : {losses}")
    print(f"確定取引の勝率     : {win_rate:.2f}%")
    print(f"判定不能           : {ambiguous}")
    print(f"目標未到達         : {no_touch}")
    print(f"最大連敗           : {max_losing_streak}")
    print(f"損益合計           : ${pnl:+.2f}")
    print("====================================")
    print("※勝率は勝敗が確定した取引のみで計算")
    print("※判定不能・未到達を勝ち負けに含めていません")


def backtest_mode(args: argparse.Namespace) -> None:
    print("BTC 1分足データを取得中...")
    df = get_data(args.period)

    print(f"データ件数: {len(df)}")
    print(f"開始: {df.index[0]}")
    print(f"終了: {df.index[-1]}")
    print("バックテスト実行中...")

    results = run_backtest(
        df=df,
        target=args.target,
        stake=args.stake,
        payout=args.payout / 100.0,
        hold_bars=args.hold_bars,
    )

    print_report(results)

    if not results.empty:
        output = DATA_DIR / "backtest_results.csv"
        results.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"\n結果を保存しました: {output.resolve()}")


def paper_mode(args: argparse.Namespace) -> None:
    """新しい確定足ごとに方向を表示する。注文は出さない。"""
    last_seen = None
    print("ペーパーモード開始。Ctrl+Cで終了します。")
    print("このモードは予測表示のみで、取引を行いません。")

    try:
        while True:
            df = get_data("1d")

            # 最新足は未確定の可能性があるため、
            # ひとつ前の足を使ってシグナルを計算
            if len(df) < 32:
                time.sleep(10)
                continue

            i = len(df) - 2
            bar_time = df.index[i]

            if last_seen != bar_time:
                last_seen = bar_time
                direction = make_signal(df, i)
                price = float(df["Close"].iloc[i])

                print(
                    f"{bar_time} | BTC=${price:,.2f} | "
                    f"予測={direction}"
                )

                row = pd.DataFrame([{
                    "time": bar_time,
                    "price": price,
                    "signal": direction,
                }])

                log_path = DATA_DIR / "paper_signals.csv"
                row.to_csv(
                    log_path,
                    mode="a",
                    header=not log_path.exists(),
                    index=False,
                    encoding="utf-8-sig",
                )

            time.sleep(10)

    except KeyboardInterrupt:
        print("\nペーパーモードを終了しました。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UTAYA BTC HILO CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper"],
        default="backtest",
        help="実行モード",
    )
    parser.add_argument(
        "--period",
        default="7d",
        help="バックテスト期間（例: 7d）",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=50.0,
        help="上下目標幅 USD",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=5.0,
        help="1回の仮想賭け金 USD",
    )
    parser.add_argument(
        "--payout",
        type=float,
        default=88.0,
        help="勝利時の利益率（%）",
    )
    parser.add_argument(
        "--hold-bars",
        type=int,
        default=1,
        help="エントリー後の判定足数。1なら次の1分足",
    )

    args = parser.parse_args()

    if args.target <= 0 or args.stake <= 0:
        parser.error("targetとstakeは正の値にしてください")
    if not 0 <= args.payout <= 500:
        parser.error("payoutは0～500の範囲にしてください")
    if args.hold_bars < 1:
        parser.error("hold-barsは1以上にしてください")

    try:
        if args.mode == "backtest":
            backtest_mode(args)
        else:
            paper_mode(args)
    except Exception as exc:
        print(f"\nエラー: {exc}")
        print(
            "通信状況、データ取得制限、"
            "引数を確認してください。"
        )


if __name__ == "__main__":
    main()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from datetime import datetime
import os

# =========================================================
# UTAYA グローバル・インベストメンツ (v17 創発・先回り戦闘エンジン)
# 思想: 群集の先回り(ボラティリティスクイーズ・プレッシャー蓄積)、勝敗DNAの即時学習、絶対生存サーキットブレーカー
# =========================================================

st.set_page_config(
    page_title="UTAYA Global Investments - v17 Emergent Anticipation Engine",
    page_icon="🔥",
    layout="wide",
)

SYMBOLS_CONFIG = {
    "BTC (暗号資産)": {"ticker": "BTC-USD", "quick_range_usd": 50.0},
    "ETH (暗号資産)": {"ticker": "ETH-USD", "quick_range_usd": 15.0},
    "SOL (暗号資産)": {"ticker": "SOL-USD", "quick_range_usd": 2.0},
    "XRP (暗号資産)": {"ticker": "XRP-USD", "quick_range_usd": 0.05},
    "BNB (暗号資産)": {"ticker": "BNB-USD", "quick_range_usd": 5.0},
    "WTI原油 (エネルギー)": {"ticker": "CL=F", "quick_range_usd": 0.40},
    "金スポット (貴金属)": {"ticker": "GC=F", "quick_range_usd": 5.0},
    "日経平均 (日本株)": {"ticker": "^N225", "quick_range_usd": 100.0},
    "S&P 500 (米国株)": {"ticker": "^GSPC", "quick_range_usd": 15.0},
    "Nasdaq 100 (米国株)": {"ticker": "^NDX", "quick_range_usd": 50.0},
    "USD/JPY (為替)": {"ticker": "USDJPY=X", "quick_range_usd": 0.15},
    "米10年債利回り (金利)": {"ticker": "^TNX", "quick_range_usd": 0.05},
}

PAYOUT_RATE = 0.88  
DATA_DIR = Path("utaya_data_v17")
DATA_DIR.mkdir(exist_ok=True)

GLOBAL_HISTORY_DIR = DATA_DIR / "history"
GLOBAL_HISTORY_DIR.mkdir(exist_ok=True)
ANALYSIS_LOG_FILE = DATA_DIR / "analysis_log_v17.csv"

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    .combat-banner {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 20px;
        border-radius: 12px;
        color: #00ffcc;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #00ffcc;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.2);
    }
    .panel-anticipation {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
    }
    .side-high {
        flex: 1;
        background: rgba(0, 255, 204, 0.1);
        border: 2px solid #00ffcc;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
    }
    .side-low {
        flex: 1;
        background: rgba(255, 75, 75, 0.1);
        border: 2px solid #ff4b4b;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
    }
    .circuit-lock {
        background: rgba(255, 0, 0, 0.2);
        border-left: 6px solid #ff0000;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 15px;
        color: #ffcccc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔥 UTAYA v17 創発・先回り戦闘エンジン")
st.caption(
    "群集の先回り検出 (ボラティリティスクイーズ) | 失敗DNA即時学習 | 絶対生存サーキットブレーカー"
)


# -------------------- ① データ同期エンジン --------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_single_market_data(ticker, period="60d", interval="1h"):
    file_path = GLOBAL_HISTORY_DIR / f"{ticker.replace('^', '').replace('=', '_')}_{interval}.csv"
    try:
        df_new = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df_new is not None and not df_new.empty:
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            cols = ["Open", "High", "Low", "Close", "Volume"] if "Volume" in df_new.columns else ["Open", "High", "Low", "Close"]
            if all(c in df_new.columns for c in ["Open", "High", "Low", "Close"]):
                df_new = df_new[cols].copy()
                if "Volume" not in df_new.columns:
                    df_new["Volume"] = np.nan
                df_new = df_new.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
                df_new.index = pd.to_datetime(df_new.index)
                if df_new.index.tz is not None:
                    df_new.index = df_new.index.tz_localize(None)
                df_new = df_new[(df_new["Close"] > 0) & (df_new["High"] >= df_new["Low"])]
                df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()

                if file_path.exists():
                    try:
                        df_old = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        df_old.index = pd.to_datetime(df_old.index)
                        if df_old.index.tz is not None:
                            df_old.index = df_old.index.tz_localize(None)
                        df_combined = pd.concat([df_old, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                        df_combined.to_csv(file_path, encoding="utf-8-sig")
                    except Exception:
                        df_new.to_csv(file_path, encoding="utf-8-sig")
                else:
                    df_new.to_csv(file_path, encoding="utf-8-sig")

        if file_path.exists():
            df_stored = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df_stored.index = pd.to_datetime(df_stored.index)
            if df_stored.index.tz is not None:
                df_stored.index = df_stored.index.tz_localize(None)
            return df_stored.sort_index()
        return df_new
    except Exception:
        if file_path.exists():
            df_stored = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df_stored.index = pd.to_datetime(df_stored.index)
            if df_stored.index.tz is not None:
                df_stored.index = df_stored.index.tz_localize(None)
            return df_stored.sort_index()
        return None


def fetch_all_markets_data(period="60d", interval="1h"):
    all_dfs = {}
    for name, cfg in SYMBOLS_CONFIG.items():
        df = fetch_single_market_data(cfg["ticker"], period=period, interval=interval)
        if df is not None and not df.empty:
            all_dfs[name] = df
    return all_dfs


# -------------------- ② 先回り特徴量 & 厳格ハイロー直接学習 --------------------

def build_v17_combat_features_and_labels(target_name, all_dfs, fixed_range_usd, hold_bars=5):
    if target_name not in all_dfs:
        return pd.DataFrame(), pd.Series(dtype=int)

    df_target = all_dfs[target_name].copy()
    df_target.index = pd.to_datetime(df_target.index)
    if df_target.index.tz is not None:
        df_target.index = df_target.index.tz_localize(None)

    open_p = df_target["Open"].astype(float)
    high = df_target["High"].astype(float)
    low = df_target["Low"].astype(float)
    close = df_target["Close"].astype(float)
    volume = df_target["Volume"].astype(float) if "Volume" in df_target.columns else pd.Series(np.nan, index=df_target.index)

    returns = close.pct_change()
    delta = close.diff()
    
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(losses != 0, 100.0)
    rsi = rsi.where(gains != 0, 50.0)

    # 1. ボラティリティ・スクイーズ（暴風雨前の静けさ＝先回りエントリーの核心）
    rolling_std = returns.rolling(14).std()
    squeeze_index = rolling_std / rolling_std.rolling(30).mean().replace(0, 0.0001)

    # 2. オーダー・プレッシャー（上下のヒゲと実体の比率から大口の仕掛けを検出）
    body_size = (close - open_p).abs()
    candle_range = (high - low).replace(0, 0.0001)
    body_ratio = body_size / candle_range
    upper_pressure = (high - np.maximum(open_p, close)) / candle_range
    lower_pressure = (np.minimum(open_p, close) - low) / candle_range

    features = pd.DataFrame(index=df_target.index)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["sma_ratio_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean()
    features["rsi"] = rsi
    features["squeeze_index"] = squeeze_index  # 圧縮度（先回り指標）
    features["body_ratio"] = body_ratio
    features["pressure_diff"] = lower_pressure - upper_pressure  # 買い圧力 vs 売り圧力
    features["volatility"] = rolling_std

    # ハイロー値幅到達ラベル（未来リーク完全排除）
    labels = []
    for idx in range(len(df_target) - hold_bars):
        entry_price = float(df_target["Open"].iloc[idx + 1])
        future_window = df_target.iloc[idx + 1 : idx + 1 + hold_bars]
        
        upper_target = entry_price + fixed_range_usd
        lower_target = entry_price - fixed_range_usd
        
        lbl = np.nan
        for _, candle in future_window.iterrows():
            h = float(candle["High"])
            l = float(candle["Low"])
            hit_up = h >= upper_target
            hit_down = l <= lower_target
            
            if hit_up and not hit_down:
                lbl = 1
                break
            elif hit_down and not hit_up:
                lbl = 0
                break
            elif hit_up and hit_down:
                break
        labels.append(lbl)

    features_trimmed = features.iloc[:len(labels)].copy()
    features_trimmed["target"] = labels
    features_trimmed = features_trimmed.replace([np.inf, -np.inf], np.nan).dropna()

    if features_trimmed.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    return features_trimmed.drop(columns=["target"]), features_trimmed["target"].astype(int)


# -------------------- ③ 評価・戦闘判定関数 --------------------

def evaluate_combat_outcome(entry_price, direction, future_candles, fixed_range_usd, hold_bars=5):
    if future_candles is None or len(future_candles) == 0:
        return "PENDING", "未来足の到着待ち", "PENDING", None

    upper_target = float(entry_price) + fixed_range_usd
    lower_target = float(entry_price) - fixed_range_usd

    for idx, candle in future_candles.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])

        hit_up = high >= upper_target
        hit_down = low <= lower_target

        if hit_up and hit_down:
            return "LOSS", "乱高下によるノイズ被弾", "NOISE_HIT", str(idx)
        if hit_up:
            if direction == "HIGH":
                return "WIN", "上方ターゲット先着", "SUCCESS_HIGH", str(idx)
            else:
                return "LOSS", "上方逆行被弾", "REVERSE_HIGH", str(idx)
        if hit_down:
            if direction == "LOW":
                return "WIN", "下方ターゲット先着", "SUCCESS_LOW", str(idx)
            else:
                return "LOSS", "下方逆行被弾", "REVERSE_LOW", str(idx)

    if len(future_candles) >= hold_bars:
        return "TIMEOUT", "膠着・ターゲット未達", "TIMEOUT", str(future_candles.index[-1])

    return "PENDING", "交戦中", "PENDING", None


def classify_combat_tier(feat_row, p_high, p_low, current_drawdown_pct):
    # 🚨 絶対生存サーキットブレーカー（資金が15%以上削られたら戦闘強制停止）
    if current_drawdown_pct >= 0.15:
        return "絶対生存・緊急停止", "🚨 危険水域到達。資本防衛のため全システム戦闘停止。"

    squeeze = float(feat_row.get("squeeze_index", 1.0))
    probability_gap = abs(p_high - p_low)

    # 嵐の前の静けさ（スクイーズ状態）かつ確率ギャップがある場合は先回り攻撃
    if squeeze < 0.8 and probability_gap >= 0.03:
        return "先回り特攻候補", f"⚡ スクイーズ検知（圧縮度 {squeeze:.2f}）＆ギャップ {probability_gap:.3f}。先回り奇襲の好機。"

    if probability_gap >= 0.04:
        return "通常戦闘候補", f"🔥 確率優位確立（ギャップ {probability_gap:.3f}）。通常迎撃態勢。"

    return "待伏せ・条件監視", f"👀 エッジ蓄積中（ギャップ {probability_gap:.3f}, スクイーズ {squeeze:.2f}）。次のシグナルを待伏せ。"


# -------------------- ④ ウォークフォワード戦闘検証 --------------------

def run_v17_combat_backtest(target_name, all_dfs, stake_usd, hold_bars, fixed_range_usd):
    df_target = all_dfs.get(target_name)
    if df_target is None or len(df_target) < 300:
        return pd.DataFrame()

    features_df, labels_s = build_v17_combat_features_and_labels(target_name, all_dfs, fixed_range_usd, hold_bars)
    if len(features_df) < 100:
        return pd.DataFrame()

    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
    scaler = StandardScaler()

    init_end = int(len(features_df) * 0.3)
    X_init = features_df.iloc[:init_end].values
    y_init = labels_s.iloc[:init_end].values

    scaler.fit(X_init)
    model.partial_fit(scaler.transform(X_init), y_init, classes=np.array([0, 1]))

    records = []
    current_capital = stake_usd
    peak_capital = stake_usd

    for i in range(init_end, len(features_df) - hold_bars):
        row_time = features_df.index[i]
        feat_row = features_df.iloc[i]

        X_pred = scaler.transform(np.array([feat_row.values]))
        try:
            probs = model.predict_proba(X_pred)[0]
            classes_list = list(model.classes_)
            p_low = probs[classes_list.index(0)] if 0 in classes_list else probs[0]
            p_high = probs[classes_list.index(1)] if 1 in classes_list else probs[1]
        except Exception:
            p_low, p_high = 0.5, 0.5

        direction = "HIGH" if p_high >= p_low else "LOW"
        
        current_dd = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0.0
        tier, tier_reason = classify_combat_tier(feat_row, p_high, p_low, current_dd)

        try:
            entry_idx_loc = df_target.index.get_loc(row_time) + 1
            entry_price = float(df_target["Open"].iloc[entry_idx_loc])
            future_window = df_target.iloc[entry_idx_loc : entry_idx_loc + hold_bars]
        except Exception:
            continue

        result, reason, category, settle_time = evaluate_combat_outcome(
            entry_price, direction, future_window, fixed_range_usd, hold_bars=hold_bars
        )
        
        status = "SETTLED" if result != "PENDING" else "PENDING"
        pnl = 0.0
        if status == "SETTLED":
            if "緊急停止" in tier:
                pnl = 0.0
            elif tier in ["先回り特攻候補", "通常戦闘候補"]:
                pnl = stake_usd * PAYOUT_RATE if result == "WIN" else -stake_usd
                current_capital += pnl
                if current_capital > peak_capital:
                    peak_capital = current_capital

        # 失敗DNAの即時学習（負けからも勝ちからも直ちにモデルを適応）
        actual_label = int(labels_s.loc[row_time])
        try:
            model.partial_fit(X_pred, np.array([actual_label]), classes=np.array([0, 1]))
        except ValueError:
            model.partial_fit(X_pred, np.array([actual_label]))

        records.append({
            "time": str(row_time),
            "symbol": target_name,
            "direction": direction,
            "p_high": p_high,
            "p_low": p_low,
            "tier": tier,
            "result": result,
            "pnl_usd": pnl,
            "capital": current_capital,
        })

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df.to_csv(ANALYSIS_LOG_FILE, index=False, encoding="utf-8-sig")
    return res_df


# -------------------- ⑤ ダッシュボード・UI描画 --------------------

st.sidebar.header("🔥 v17 戦闘司令室")
selected_symbol = st.sidebar.selectbox("🎯 ターゲット銘柄", options=list(SYMBOLS_CONFIG.keys()), index=0)

cfg = SYMBOLS_CONFIG[selected_symbol]
default_range = cfg["quick_range_usd"]

fixed_range_usd = st.sidebar.number_input("📏 目標値幅 (±$)", 0.001, 10000.0, float(default_range), 0.01, format="%.2f")
stake_usd = st.sidebar.number_input("💵 参戦資金 ($)", 1.0, 10000.0, 10.0, 1.0)
hold_bars = st.sidebar.slider("⏱️ 判定期間 (本)", 1, 20, 5)

if st.sidebar.button("🔄 戦闘ログ・記憶リセット", use_container_width=True):
    if ANALYSIS_LOG_FILE.exists():
        ANALYSIS_LOG_FILE.unlink()
    st.sidebar.success("初期化完了。新たな戦いへ。")
    st.rerun()

tab1, tab2 = st.tabs(["⚡ 個別実戦・先回りパネル", "🌌 全市場・一斉急襲スキャナー"])

with tab1:
    if "v17_active" not in st.session_state:
        st.session_state.v17_active = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ 先回り戦闘AI起動", use_container_width=True):
            st.session_state.v17_active = True
    with c2:
        if st.button("⏹ 緊急停戦", use_container_width=True):
            st.session_state.v17_active = False

    if st.session_state.v17_active:
        st.success(f"戦闘態勢稼働中 [{selected_symbol}]：スクイーズ検出＆即時適応中")
    else:
        st.info("待機中。起動ボタンを押してください。")

    @st.fragment(run_every="30s")
    def combat_loop():
        if not st.session_state.v17_active:
            return

        all_dfs = fetch_all_markets_data(period="60d", interval="1h")
        df_target = all_dfs.get(selected_symbol)
        if df_target is None or len(df_target) < 50:
            st.warning("データ同期中...")
            return

        features_df, labels_s = build_v17_combat_features_and_labels(selected_symbol, all_dfs, fixed_range_usd, hold_bars)
        if len(features_df) < 10:
            return

        signal_i = len(features_df) - 1
        feat_row = features_df.iloc[signal_i]
        row_time = features_df.index[signal_i]

        X_live = features_df.values
        y_live = labels_s.values
        scaler = StandardScaler().fit(X_live)
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        model.partial_fit(scaler.transform(X_live), y_live, classes=np.array([0, 1]))

        probs = model.predict_proba(scaler.transform(np.array([feat_row.values])))[0]
        p_low, p_high = probs[0], probs[1]
        direction = "HIGH" if p_high >= p_low else "LOW"
        
        tier, tier_reason = classify_combat_tier(feat_row, p_high, p_low, 0.0)
        signal_price = float(df_target["Close"].loc[row_time]) if row_time in df_target.index else float(df_target["Close"].iloc[-1])

        # ⚡ 先回り戦闘パネル
        st.markdown(
            f"""
            <div class="panel-anticipation">
                <div class="side-high">
                    <h3>▲ 昇華・HIGH予測</h3>
                    <h1 style="color: #00ffcc; margin: 10px 0;">{p_high:.1%}</h1>
                    <p style="font-size: 0.8rem; color: #ccc;">上方ターゲット: +${fixed_range_usd}</p>
                </div>
                <div class="side-low">
                    <h3>▼ 降下・LOW予測</h3>
                    <h1 style="color: #ff4b4b; margin: 10px 0;">{p_low:.1%}</h1>
                    <p style="font-size: 0.8rem; color: #ccc;">下方ターゲット: -${fixed_range_usd}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "緊急停止" in tier:
            st.markdown(f'<div class="circuit-lock"><h3>{tier}</h3><p>{tier_reason}</p></div>', unsafe_allow_html=True)
        else:
            st.info(f"🎯 現在の戦闘ステージ: 【 {tier} 】\n\n{tier_reason} (基準価格: ${signal_price:,.2f})")

    combat_loop()

    st.markdown("---")
    st.subheader("🚀 過去戦歴の検証と生存確認 (バックテスト)")
    if st.button("▶️ v17 厳格バックテスト実行", use_container_width=True):
        with st.spinner("全履歴をスキャンし、勝敗DNAの即時学習と資金曲線検証中..."):
            all_dfs = fetch_all_markets_data(period="60d", interval="1h")
            res_df = run_v17_combat_backtest(selected_symbol, all_dfs, stake_usd, hold_bars, fixed_range_usd)
            if not res_df.empty:
                total_trades = len(res_df[res_df["result"].isin(["WIN", "LOSS"])])
                wins = len(res_df[res_df["result"] == "WIN"])
                win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0
                final_cap = res_df["capital"].iloc[-1]
                
                st.markdown(f"### 📊 戦果報告: 勝率 {win_rate:.2f}% ({wins}/{total_trades}勝) | 最終資金: ${final_cap:.2f} (初期 ${stake_usd})")
                st.line_chart(res_df.set_index("time")[["capital"]])
                st.dataframe(res_df.tail(20), use_container_width=True, hide_index=True)
            else:
                st.warning("有効な戦闘データがありません。")

with tab2:
    st.subheader("🌌 全12市場 先回り急襲スキャナー")
    if st.button("🔍 全市場の歪みを一斉スキャン", use_container_width=True):
        with st.spinner("全12市場のボラティリティ・スクイーズを解析中..."):
            all_dfs = fetch_all_markets_data(period="60d", interval="1h")
            scan_res = []
            for name, cfg in SYMBOLS_CONFIG.items():
                df = all_dfs.get(name)
                rng = cfg["quick_range_usd"]
                if df is None or len(df) < 50:
                    continue
                try:
                    f_df, l_s = build_v17_combat_features_and_labels(name, all_dfs, rng, 5)
                    if len(f_df) < 10:
                        continue
                    feat_row = f_df.iloc[-1]
                    scaler = StandardScaler().fit(f_df.values)
                    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42).partial_fit(scaler.transform(f_df.values), l_s.values, classes=np.array([0, 1]))
                    probs = model.predict_proba(scaler.transform(np.array([feat_row.values])))[0]
                    p_l, p_h = probs[0], probs[1]
                    tier, _ = classify_combat_tier(feat_row, p_h, p_l, 0.0)
                    scan_res.append({"銘柄": name, "ステージ": tier, "HIGH確率": f"{p_h:.1%}", "LOW確率": f"{p_l:.1%}", "ギャップ": round(abs(p_h - p_l), 3)})
                except Exception:
                    pass
            if scan_res:
                st.dataframe(pd.DataFrame(scan_res), use_container_width=True, hide_index=True)
            else:
                st.warning("スキャンデータ不足。")

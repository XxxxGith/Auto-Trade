import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
import time

# ================================================================
# Utility: auto-detect close/open/volume columns
# ================================================================
def detect_close_column(df):
    cols = [c for c in df.columns if c.startswith("close")]
    if cols:
        return cols[0]
    cols = [c for c in df.columns if "close" in c.lower()]
    if not cols:
        raise ValueError(f"No CLOSE-like column found! Columns = {df.columns}")
    return cols[0]

def detect_open_column(df):
    cols = [c for c in df.columns if c.startswith("open")]
    if cols:
        return cols[0]
    cols = [c for c in df.columns if "open" in c.lower()]
    return cols[0] if cols else None

def detect_volume_column(df):
    vols = [c for c in df.columns if "vol" in c.lower()]
    if not vols:
        raise ValueError(f"No VOLUME column found! Columns = {df.columns}")
    return vols[0]

def detect_high_column(df):
    highs = [c for c in df.columns if c.startswith("high")]
    if highs:
        return highs[0]
    highs = [c for c in df.columns if "high" in c.lower()]
    return highs[0] if highs else None

def detect_low_column(df):
    lows = [c for c in df.columns if c.startswith("low")]
    if lows:
        return lows[0]
    lows = [c for c in df.columns if "low" in c.lower()]
    return lows[0] if lows else None

# ================================================================
# Normalize yfinance dataframe
# ================================================================
def normalize_df(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    df.columns = df.columns.str.lower()
    df.reset_index(inplace=True)

    # datetime
    datetime_col = None
    for col in df.columns:
        if col.lower() in ["datetime", "date", "index"]:
            datetime_col = col
            break

    df["Datetime"] = pd.to_datetime(df[datetime_col])

    # price & volume
    df["close"] = pd.to_numeric(df[detect_close_column(df)], errors="coerce")
    df["open"] = pd.to_numeric(df[detect_open_column(df)], errors="coerce")
    df["high"] = pd.to_numeric(df[detect_high_column(df)], errors="coerce")
    df["low"] = pd.to_numeric(df[detect_low_column(df)], errors="coerce")
    df["volume"] = pd.to_numeric(df[detect_volume_column(df)], errors="coerce")

    return df

# ================================================================
# STRATEGIES (same as before)
# ================================================================
def strat_ma(df, fast=5, slow=20):
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()
    df["signal"] = 0
    df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
    df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1
    return df

def strat_rsi(df, length=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["signal"] = 0
    df.loc[df["rsi"] < 30, "signal"] = 1
    df.loc[df["rsi"] > 70, "signal"] = -1
    return df

def strat_macd(df):
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal_line"] = df["macd"].ewm(span=9).mean()
    df["hist"] = df["macd"] - df["signal_line"]
    df["signal"] = (df["macd"] > df["signal_line"]).astype(int) - (df["macd"] < df["signal_line"]).astype(int)
    return df

def strat_bbands(df, length=20, mult=2):
    df["ma"] = df["close"].rolling(length).mean()
    df["std"] = df["close"].rolling(length).std()
    df["upper"] = df["ma"] + mult * df["std"]
    df["lower"] = df["ma"] - mult * df["std"]
    df["signal"] = 0
    df.loc[df["close"] < df["lower"], "signal"] = 1
    df.loc[df["close"] > df["upper"], "signal"] = -1
    return df

def strat_breakout(df, length=20):
    df["recent_high"] = df["close"].rolling(length).max()
    df["recent_low"] = df["close"].rolling(length).min()
    df["signal"] = 0
    df.loc[df["close"] > df["recent_high"], "signal"] = 1
    df.loc[df["close"] < df["recent_low"], "signal"] = -1
    return df

def strat_vwap(df):
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_pv"] = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]
    df["signal"] = (df["close"] > df["vwap"]).astype(int) - (df["close"] < df["vwap"]).astype(int)
    return df

def strat_momentum(df, length=10):
    df["mom"] = df["close"].diff(length)
    df["signal"] = (df["mom"] > 0).astype(int) - (df["mom"] < 0).astype(int)
    return df

STRATEGY_MAP = {
    "MA Crossover": strat_ma,
    "RSI": strat_rsi,
    "MACD": strat_macd,
    "Bollinger Bands": strat_bbands,
    "Breakout": strat_breakout,
    "VWAP": strat_vwap,
    "Momentum": strat_momentum,
}

# ================================================================
# Streamlit UI
# ================================================================
st.set_page_config(page_title="Real-Time Strategy Dashboard", layout="wide")

st.sidebar.title("⚙ Settings")
ticker = st.sidebar.text_input("Ticker:", "AAPL").upper()
interval = st.sidebar.selectbox("Interval:", ["1m", "5m", "15m", "30m", "60m", "1d"])
period = st.sidebar.selectbox("History Range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
strategy_name = st.sidebar.selectbox("Strategy:", list(STRATEGY_MAP.keys()))
refresh_sec = st.sidebar.slider("Auto-refresh (seconds):", 2, 60, 10)

run_btn = st.sidebar.button("▶ START AUTO REFRESH")
stop_btn = st.sidebar.button("⏹ STOP")

if "running" not in st.session_state:
    st.session_state.running = False

if run_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

st.title(f"📈 Live Trading Dashboard — {ticker} ({strategy_name})")

# ================================================================
# MAIN LOOP
# ================================================================
# def run_cycle():
#     try:
#         df = normalize_df(yf.download(ticker, period=period, interval=interval))
#         df = STRATEGY_MAP[strategy_name](df)
#
#         latest = df.iloc[-1]["signal"]
#         if latest == 1:
#             st.subheader("📢 Signal: **BUY** 🟢")
#         elif latest == -1:
#             st.subheader("📢 Signal: **SELL** 🔴")
#         else:
#             st.subheader("📢 Signal: **HOLD** 🟡")
#
#         # ============================================================
#         # PRICE CHART (CANDLESTICK)
#         # ============================================================
#         # candle_base = alt.Chart(df)
#         #
#         # candle = candle_base.mark_bar().encode(
#         #     x=alt.X("Datetime:T"),
#         #     y="low:Q",
#         #     y2="high:Q",
#         #     color=alt.condition("datum.open <= datum.close",
#         #                         alt.value("#26a69a"),  # green
#         #                         alt.value("#ef5350"))  # red
#         # )
#         #
#         # candle_body = candle_base.mark_bar(size=5).encode(
#         #     x="Datetime:T",
#         #     y="open:Q",
#         #     y2="close:Q",
#         #     color=alt.condition("datum.open <= datum.close",
#         #                         alt.value("#26a69a"),
#         #                         alt.value("#ef5350"))
#         # )
#         #
#         # price_chart = candle + candle_body
#
#         # ===========================
#         # PRICE CANDLESTICK CHART
#         # ===========================
#         y_min = float(df["low"].min()) * 0.995
#         y_max = float(df["high"].max()) * 1.005
#         price_scale = alt.Scale(domain=[y_min, y_max])
#
#         candle_base = alt.Chart(df)
#
#         # Wicks
#         candle_wick = candle_base.mark_rule().encode(
#             x="Datetime:T",
#             y=alt.Y("low:Q", scale=price_scale),
#             y2="high:Q",
#             color=alt.condition("datum.open <= datum.close",
#                                 alt.value("#26a69a"),
#                                 alt.value("#ef5350"))
#         )
#
#         # Candle body
#         candle_body = candle_base.mark_bar(size=5).encode(
#             x="Datetime:T",
#             y=alt.Y("open:Q", scale=price_scale),
#             y2="close:Q",
#             color=alt.condition("datum.open <= datum.close",
#                                 alt.value("#26a69a"),
#                                 alt.value("#ef5350"))
#         )
#
#         price_chart = candle_wick + candle_body
#
#         # overlays
#         if strategy_name == "MA Crossover":
#             price_chart += candle_base.mark_line(color="orange").encode(y="ma_fast:Q")
#             price_chart += candle_base.mark_line(color="red").encode(y="ma_slow:Q")
#
#         if strategy_name == "Bollinger Bands":
#             price_chart += candle_base.mark_line(color="yellow").encode(y="upper:Q")
#             price_chart += candle_base.mark_line(color="yellow").encode(y="lower:Q")
#
#         if strategy_name == "VWAP":
#             price_chart += candle_base.mark_line(color="purple").encode(y="vwap:Q")
#
#         # containers fix layout
#         price_container = st.container()
#         indicator_container = st.container()
#
#         with price_container:
#             st.altair_chart(price_chart.properties(height=380), use_container_width=True)
#
#         # ============================================================
#         # INDICATOR SUBPLOTS
#         # ============================================================
#         with indicator_container:
#             if strategy_name == "RSI":
#                 st.line_chart(df[["rsi"]])
#
#             if strategy_name == "MACD":
#                 st.line_chart(df[["macd", "signal_line", "hist"]])
#
#             if strategy_name == "Momentum":
#                 st.line_chart(df[["mom"]])
#
#     except Exception as e:
#         st.error(f"❌ Error: {e}")
#
# run_cycle()
#
# if st.session_state.running:
#     time.sleep(refresh_sec)
#     st.rerun()


def run_cycle():
    try:
        df = normalize_df(yf.download(ticker, period=period, interval=interval))
        df = STRATEGY_MAP[strategy_name](df)

        # Determine Signal
        latest = df.iloc[-1]["signal"]
        if latest == 1:
            st.subheader("📢 Signal: **BUY** 🟢")
        elif latest == -1:
            st.subheader("📢 Signal: **SELL** 🔴")
        else:
            st.subheader("📢 Signal: **HOLD** 🟡")

        # ============================================================
        # CREATE BUY / SELL MARKERS
        # ============================================================
        df["buy"] = df["signal"].apply(lambda x: 1 if x == 1 else None)
        df["sell"] = df["signal"].apply(lambda x: 1 if x == -1 else None)

        df["buy_price"] = df["high"] * 1.002
        df["sell_price"] = df["low"] * 0.998

        # ============================================================
        # PRICE CANDLESTICK CHART
        # ============================================================
        y_min = float(df["low"].min()) * 0.995
        y_max = float(df["high"].max()) * 1.005
        price_scale = alt.Scale(domain=[y_min, y_max])

        candle_base = alt.Chart(df)

        # Wick lines
        candle_wick = candle_base.mark_rule().encode(
            x="Datetime:T",
            y=alt.Y("low:Q", scale=price_scale),
            y2="high:Q",
            color=alt.condition("datum.open <= datum.close",
                                alt.value("#26a69a"),
                                alt.value("#ef5350"))
        )

        # Candle body
        candle_body = candle_base.mark_bar(size=5).encode(
            x="Datetime:T",
            y=alt.Y("open:Q", scale=price_scale),
            y2="close:Q",
            color=alt.condition("datum.open <= datum.close",
                                alt.value("#26a69a"),
                                alt.value("#ef5350"))
        )

        price_chart = candle_wick + candle_body

        # Strategy overlays
        if strategy_name == "MA Crossover":
            price_chart += candle_base.mark_line(color="orange").encode(y="ma_fast:Q")
            price_chart += candle_base.mark_line(color="red").encode(y="ma_slow:Q")

        if strategy_name == "Bollinger Bands":
            price_chart += candle_base.mark_line(color="yellow").encode(y="upper:Q")
            price_chart += candle_base.mark_line(color="yellow").encode(y="lower:Q")

        if strategy_name == "VWAP":
            price_chart += candle_base.mark_line(color="purple").encode(y="vwap:Q")

        # ============================================================
        # BUY / SELL ARROWS
        # ============================================================
        buy_marks = alt.Chart(df[df["buy"] == 1]).mark_text(
            text="↑", color="#00e676", fontSize=18, dy=-10
        ).encode(
            x="Datetime:T",
            y="buy_price:Q"
        )

        sell_marks = alt.Chart(df[df["sell"] == -1]).mark_text(
            text="↓", color="#ff1744", fontSize=18, dy=10
        ).encode(
            x="Datetime:T",
            y="sell_price:Q"
        )

        price_chart = price_chart + buy_marks + sell_marks

        # ============================================================
        # VOLUME BARS
        # ============================================================
        volume_chart = alt.Chart(df).mark_bar().encode(
            x="Datetime:T",
            y=alt.Y("volume:Q", axis=alt.Axis(title="Volume")),
            color=alt.condition(
                "datum.close >= datum.open",
                alt.value("#26a69a"),   # green
                alt.value("#ef5350")    # red
            )
        ).properties(height=120)

        # Combine price + volume vertically
        full_chart = alt.vconcat(
            price_chart.properties(height=380),
            volume_chart
        ).resolve_scale(x='shared')

        # Output charts
        price_container = st.container()
        indicator_container = st.container()

        with price_container:
            st.altair_chart(full_chart, use_container_width=True)

        # ============================================================
        # INDICATORS
        # ============================================================
        with indicator_container:
            if strategy_name == "RSI":
                st.line_chart(df[["rsi"]])

            if strategy_name == "MACD":
                st.line_chart(df[["macd", "signal_line", "hist"]])

            if strategy_name == "Momentum":
                st.line_chart(df[["mom"]])

    except Exception as e:
        st.error(f"❌ Error: {e}")


run_cycle()

if st.session_state.running:
    time.sleep(refresh_sec)
    st.rerun()
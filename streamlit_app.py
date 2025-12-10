import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
import time

# ================================================================
# Utility: auto-detect close/open columns
# ================================================================
def detect_close_column(df):
    cols = [c for c in df.columns if "close" in c.lower()]
    if not cols:
        raise ValueError(f"No CLOSE-like column found! Columns = {df.columns}")
    return cols[0]


def detect_open_column(df):
    cols = [c for c in df.columns if "open" in c.lower()]
    return cols[0] if cols else None


# ================================================================
# Normalize yfinance dataframe
# ================================================================
def normalize_df(df):
    df = df.copy()

    # Flatten MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    df.columns = df.columns.str.lower()
    df.reset_index(inplace=True)  # YFinance returns DatetimeIndex → becomes "Datetime"

    # ---- FIX: robust datetime detection ----
    datetime_col = None
    for col in df.columns:
        if col.lower() in ["datetime", "date", "index"]:
            datetime_col = col
            break

    if datetime_col:
        df["Datetime"] = pd.to_datetime(df[datetime_col])
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    else:
        raise ValueError(f"❌ Could not find datetime column. Columns = {df.columns}")

    # detect price fields
    close_col = detect_close_column(df)
    df["close"] = df[close_col].astype(float)

    open_col = detect_open_column(df)
    if open_col:
        df["open"] = df[open_col].astype(float)
    else:
        df["open"] = df["close"]  # fallback

    return df


# ================================================================
# Strategy: Moving Average
# ================================================================
def apply_ma_strategy(df, fast=5, slow=20):
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()

    df["signal"] = 0
    df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
    df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1
    df["signal"] = df["signal"].ffill().fillna(0)

    return df


# ================================================================
# Streamlit UI
# ================================================================
st.set_page_config(page_title="Real-Time Strategy Dashboard", layout="wide")

st.sidebar.title("⚙ Settings")

ticker = st.sidebar.text_input("Ticker:", "AAPL").upper()
interval = st.sidebar.selectbox("Interval:", ["1m", "5m", "15m", "30m", "60m", "1d"])
period = st.sidebar.selectbox("History Range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

fast = st.sidebar.number_input("Fast MA:", 1, 200, 5)
slow = st.sidebar.number_input("Slow MA:", 2, 300, 20)

refresh_sec = st.sidebar.slider("Auto-refresh (seconds):", 2, 60, 10)

run_btn = st.sidebar.button("▶ START AUTO REFRESH")
stop_btn = st.sidebar.button("⏹ STOP")

# ========== Maintain session state ==========
if "running" not in st.session_state:
    st.session_state.running = False

if run_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

st.title(f"📈 Live Trading Dashboard — {ticker}")


# ================================================================
# Fetch + Plot Function
# ================================================================
def run_cycle():
    try:
        df_raw = yf.download(ticker, period=period, interval=interval)

        if df_raw.empty:
            st.error("❌ No data received from yfinance.")
            return

        df = normalize_df(df_raw)
        df = apply_ma_strategy(df, fast, slow)

        latest = df.iloc[-1]
        signal = latest["signal"]

        # ===== Signal Display =====
        if signal == 1:
            st.subheader("📢 Signal: **BUY** 🟢")
        elif signal == -1:
            st.subheader("📢 Signal: **SELL** 🔴")
        else:
            st.subheader("📢 Signal: **HOLD** 🟡")

        # ===== Chart =====
        y_min = df["close"].min() * 0.98
        y_max = df["close"].max() * 1.02

        base = alt.Chart(df).encode(
            x=alt.X("Datetime:T", axis=alt.Axis(title="Time"))
        )

        price_line = base.mark_line(color="white").encode(
            y=alt.Y("close:Q",
                    scale=alt.Scale(domain=[y_min, y_max]),
                    axis=alt.Axis(title="Price", grid=True))
        )
        ma_fast_line = base.mark_line(color="orange").encode(y="ma_fast:Q")
        ma_slow_line = base.mark_line(color="red").encode(y="ma_slow:Q")

        chart = (price_line + ma_fast_line + ma_slow_line).properties(height=450)

        st.altair_chart(chart, use_container_width=True)

        st.write("Latest row:", latest)

    except Exception as e:
        st.error(f"❌ Error: {e}")


# ================================================================
# Loop Refresh Mode
# ================================================================
run_cycle()

if st.session_state.running:
    st.sidebar.info("🟢 Auto-refresh running...")
    time.sleep(refresh_sec)
    st.rerun()
else:
    st.sidebar.warning("⏹ Auto-refresh stopped.")

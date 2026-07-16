
# =====================================================
# AI Trading Assistant
# =====================================================

# -----------------------------
# Import Libraries
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ta
import shap
import matplotlib.pyplot as plt

from twelvedata import TDClient
from groq import Groq

# -----------------------------
# API Keys
# -----------------------------
TWELVE_DATA_API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# -----------------------------
# Initialize API
# -----------------------------
td = TDClient(apikey=TWELVE_DATA_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("trade_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("📈 AI Trading Assistant")

st.markdown(
"""
Predict whether a trade is likely to succeed using
Machine Learning, Technical Indicators,
SHAP Explainability and AI.
"""
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Trading Settings")

# -----------------------------
# Select Company
# -----------------------------
stock_options = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Meta": "META",
    "Tesla": "TSLA",
    "AMD": "AMD",
    "Alphabet (Google)": "GOOGL",
    "Netflix": "NFLX"
}

selected_company = st.sidebar.selectbox(
    "Select Company",
    list(stock_options.keys())
)

ticker = stock_options[selected_company]


interval = st.sidebar.selectbox(
    "Interval",
    [
        "1min",
        "5min",
        "15min"
    ]
)

predict_button = st.sidebar.button(
    "Predict Trade"
)

# -----------------------------
# Model Loaded
# -----------------------------
st.success("Model Loaded Successfully")

# -----------------------------
# Live Data
# -----------------------------
if predict_button:

    try:

        with st.spinner("Downloading live data..."):

            ts = td.time_series(
                symbol=ticker,
                interval=interval,
                outputsize=200
            )

            live_df = ts.as_pandas()

            live_df = (
                live_df
                .reset_index()
            )

            live_df.columns = [
                c.lower()
                for c in live_df.columns
            ]

            live_df = (
                live_df
                .sort_values("datetime")
                .reset_index(drop=True)
            )

        st.subheader("Latest Market Data")

        st.dataframe(
            live_df.tail()
        )




        # -----------------------------
        # Latest Market Price
        # -----------------------------
        latest_price = live_df["close"].iloc[-1]

        st.metric(
            label="Latest Price",
            value=f"${latest_price:.2f}"
        )




        # -----------------------------
        # Feature Engineering
        # -----------------------------
        with st.spinner("Calculating technical indicators..."):

            # Daily Return
            live_df["Daily_Return"] = live_df["close"].pct_change()

            # Price Change
            live_df["Price_Change"] = (
                live_df["close"] -
                live_df["open"]
            )

            # Moving Averages
            live_df["MA10"] = (
                live_df["close"]
                .rolling(10)
                .mean()
            )

            live_df["MA20"] = (
                live_df["close"]
                .rolling(20)
                .mean()
            )

            # Exponential Moving Average
            live_df["EMA10"] = (
                live_df["close"]
                .ewm(span=10, adjust=False)
                .mean()
            )

            # Volatility
            live_df["Volatility"] = (
                live_df["Daily_Return"]
                .rolling(10)
                .std()
            )

            # RSI
            live_df["RSI"] = ta.momentum.RSIIndicator(
                close=live_df["close"],
                window=14
            ).rsi()

            # Trend
            live_df["Trend"] = (
                live_df["MA10"] >
                live_df["MA20"]
            ).astype(int)

            # MACD
            macd = ta.trend.MACD(
                close=live_df["close"]
            )

            live_df["MACD"] = macd.macd()
            live_df["MACD_Signal"] = macd.macd_signal()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(
                close=live_df["close"],
                window=20
            )

            live_df["BB_Upper"] = bb.bollinger_hband()
            live_df["BB_Lower"] = bb.bollinger_lband()

            # ATR
            atr = ta.volatility.AverageTrueRange(
                high=live_df["high"],
                low=live_df["low"],
                close=live_df["close"],
                window=14
            )

            live_df["ATR"] = atr.average_true_range()

            # Momentum
            live_df["Momentum"] = (
                ta.momentum.ROCIndicator(
                    close=live_df["close"],
                    window=10
                ).roc()
            )

            # Volume Features
            live_df["Volume_MA20"] = (
                live_df["volume"]
                .rolling(20)
                .mean()
            )

            live_df["Relative_Volume"] = (
                live_df["volume"] /
                live_df["Volume_MA20"]
            )

            # Candlestick Features
            live_df["Candle_Body"] = (
                live_df["close"] -
                live_df["open"]
            ).abs()

            live_df["Upper_Shadow"] = (
                live_df["high"] -
                live_df[["open", "close"]].max(axis=1)
            )

            live_df["Lower_Shadow"] = (
                live_df[["open", "close"]].min(axis=1) -
                live_df["low"]
            )

            live_df["High_Low_Range"] = (
                live_df["high"] -
                live_df["low"]
            )

            # Remove rows with missing values
            live_df.dropna(inplace=True)

        st.success("Feature Engineering Completed")

        # -----------------------------
        # Prepare Latest Trade
        # -----------------------------
        feature_order = list(
            scaler.feature_names_in_
        )

        latest_trade = (
            live_df[feature_order]
            .iloc[[-1]]
        )

        st.subheader("Latest Trade Features")


        st.dataframe(latest_trade)

        # -----------------------------
        # Scale Latest Trade
        # -----------------------------
        latest_trade_scaled = scaler.transform(
            latest_trade
        )

        # -----------------------------
        # Predict Trade
        # -----------------------------
        prediction = model.predict(
            latest_trade_scaled
        )[0]

        success_probability = model.predict_proba(
            latest_trade_scaled
        )[0][1]

        # -----------------------------
        # Risk Assessment
        # -----------------------------
        volatility = latest_trade["Volatility"].iloc[0]

        if success_probability >= 0.70 and volatility < 0.01:

            risk = "🟢 Low Risk"

            recommendation = "Consider Trade"

        elif success_probability >= 0.50:

            risk = "🟡 Medium Risk"

            recommendation = "Wait"

        else:

            risk = "🔴 High Risk"

            recommendation = "Avoid"

        # -----------------------------
        # Prediction Results
        # -----------------------------
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prediction",
                "Successful"
                if prediction == 1
                else "Unsuccessful"
            )

        with col2:

            st.metric(
                "Success Probability",
                f"{success_probability:.2%}"
            )

        with col3:

            st.metric(
                "Risk",
                risk
            )

        st.info(
            f"Recommendation: **{recommendation}**"
        )



        # -----------------------------
        # Save Prediction
        # -----------------------------
        prediction_df = pd.DataFrame({

            "Ticker": [ticker],

            "Prediction": [
                "Successful"
                if prediction == 1
                else "Unsuccessful"
            ],

            "Success Probability (%)": [
                round(success_probability * 100, 2)
            ],

            "Risk": [risk],

            "Recommendation": [
                recommendation
            ]

        })

        prediction_df.to_csv(

            "live_trade_prediction.csv",

            index=False

        )



        st.download_button(

            label="📥 Download Prediction",

            data=prediction_df.to_csv(index=False),

            file_name="live_trade_prediction.csv",

            mime="text/csv"



        )
        # -----------------------------
        # SHAP Explainability
        # -----------------------------
        with st.spinner("Generating SHAP Explanation..."):

            background_data = scaler.transform(
                live_df[feature_order]
            )

            explainer = shap.LinearExplainer(
                model,
                background_data
            )

            shap_values = explainer.shap_values(
                latest_trade_scaled
            )


        st.subheader("SHAP Feature Importance")

        shap_df = pd.DataFrame({

            "Feature": feature_order,

            "Impact": shap_values[0]

        })

        shap_df["Absolute Impact"] = (
            shap_df["Impact"].abs()
        )

        shap_df = (
            shap_df
            .sort_values(
                "Absolute Impact",
                ascending=False
            )
        )

        st.dataframe(
            shap_df[["Feature", "Impact"]]
        )



        # -----------------------------
        # Save SHAP Importance
        # -----------------------------
        shap_df.to_csv(

            "live_shap_feature_importance.csv",

            index=False

        )

        st.download_button(

            label="📥 Download SHAP Report",

            data=shap_df.to_csv(index=False),

            file_name="live_shap_feature_importance.csv",

            mime="text/csv"

        )

        # -----------------------------
        # SHAP Bar Chart
        # -----------------------------
        fig, ax = plt.subplots(figsize=(8,6))

        (
            shap_df
            .head(10)
            .sort_values(
                by="Absolute Impact",
                ascending=True
            )
            .plot.barh(
                x="Feature",
                y="Absolute Impact",
                ax=ax,
                legend=False
            )
        )

        ax.set_title("Top 10 Important Features")

        st.pyplot(fig)


        # -----------------------------
        # AI Trade Analysis
        # -----------------------------
        st.subheader("AI Trade Analysis")

        prompt = f"""
You are an experienced stock market analyst.

Ticker : {ticker}

Prediction : {"Successful" if prediction == 1 else "Unsuccessful"}

Success Probability : {success_probability:.2%}

Risk Level : {risk}

Top Features:

{shap_df.head(5).to_string(index=False)}

Give a concise analysis including:

1. Market outlook
2. Why the model predicted this
3. Main risks
4. Final recommendation

Limit the answer to about 150 words.
"""

        with st.spinner("Generating AI Analysis..."):

            response = client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                messages=[

                    {
                        "role": "user",
                        "content": prompt
                    }

                ]

            )

        ai_response = (
            response
            .choices[0]
            .message
            .content
        )

        st.markdown(ai_response)
    except Exception as e:

        st.error(f"Error : {e}")


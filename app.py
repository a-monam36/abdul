import streamlit as st
import stocks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

import google.generativeai as genai

from google.api_core import exceptions

try:
    GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
except:
    st.error("API Key not found! Make sure it is in .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash"
gen_model = genai.GenerativeModel(MODEL_NAME)


def run_ai_advisor(final_data, weights, budget, risk, max_stocks):
    try:
        # 1. Prepare data (Keep it light to save tokens)
        latest_weights = weights.iloc[0].sort_values(ascending=False).head(max_stocks)
        tickers = latest_weights.index.tolist()

        last_date = final_data.index.get_level_values('date')[-1]
        # Rounding to 2 decimal places saves significant "tokens"
        context_df = final_data.xs(last_date, level=0).loc[tickers][['rsi', 'Mkt-RF', 'SMB', 'HML']].round(2)

        # 2. Construct Prompt
        prompt = f"""
        Analyze this Cluster 0 portfolio for a ${budget} investment ({risk} risk).
        Weights: {latest_weights.to_dict()}
        Technical Context: {context_df.to_dict()}
        
        Provide a 3-bullet point investment plan and a risk warning.
        """

        # 3. Call the model
        response = gen_model.generate_content(prompt)
        return response.text

    except exceptions.ResourceExhausted:
        # This catches the 429 error specifically
        return "⚠️ **Rate Limit Reached:** Google's free tier is busy. Please wait 60 seconds and try again."
    
    except exceptions.ServiceUnavailable:
        return "⚠️ **Server Overloaded:** Gemini is temporarily unavailable. Try again in a moment."
    
    except Exception as e:
        # Catch-all for other issues (like internet connection)
        return f"❌ **An unexpected error occurred:** {e}"
    

if 'final_data' not in st.session_state:
    st.session_state.final_data = None
if 'all_charts' not in st.session_state:
    st.session_state.all_charts = None

if 'final_comparison' not in st.session_state:
    st.session_state.final_comparison = None



st.set_page_config(page_title="Quant Strategy Sandbox", layout= "wide")
st.title("🛠️ Strategy Feature Engine")

st.sidebar.header("Toggle Indicators")

use_gk = st.sidebar.toggle("Garman-Klass Volatility", value=True)

use_rsi = st.sidebar.toggle("RSI (20 Day)", value= True )

use_bb = st.sidebar.toggle("Bollinger Bands", value= True)

use_atr = st.sidebar.toggle("ATR (14 Day)", value= True)

use_macd = st.sidebar.toggle("MACD (20 Day)", value=True)

use_ff = st.sidebar.toggle("Fama-French 5-Factors", value= True, help="Calculates rolling risk exposures (Market, Size, Value, Profitability, Investment")

st.sidebar.divider()

st.sidebar.subheader("Backtest Settings")

max_weight = st.sidebar.slider("Max Stock Weight (%)", 5, 50, 10) # description, min, max, default

lookback_months = st.sidebar.select_slider("Optimization Lookback (Months)", options=[6, 12, 24], value=12)



benchmark_options = {
    "S&P 500 (Overall Market)": "SPY",
    "Nasdaq 100 (Tech Growth)": "QQQ",
    "Dow Jones (Blue Chip Giants)": "DIA",
    "Russell 2000 (Small Companies)": "IWM"
}

select_label = st.sidebar.selectbox("Select Benchmark", options=list(benchmark_options.keys()), help="Comparing against the S&P 500 shows if you're beating the average. Comparing against the Nasdaq shows if you're beating the tech leaders.")
benchmark_ticker = benchmark_options[select_label]





st.sidebar.divider()
st.sidebar.subheader("🤖 AI Advisor Settings")
user_budget = st.sidebar.number_input("Investment Budget ($)", min_value=1000, value=10000, step=1000)
user_risk = st.sidebar.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)


run_pipeline = st.sidebar.button("Execute Pipeline")


if run_pipeline:
    with st.status("Processing Data...", expanded=True) as status:
        st.write("Step 1: Downloading raw data...")

        raw_df = stocks.get_sp500_data()

        st.write("Step 2: Calculating selected indicators...")

        featured_df = stocks.calculate_metrics(raw_df, 
            use_rsi=use_rsi, 
            use_bb=use_bb, 
            use_atr=use_atr, 
            use_macd=use_macd, 
            use_gk=use_gk)
        
        st.write("Step 3: Filtering for top 150 liquid stocks...")
        filtered_df = stocks.top_150_stocks(featured_df)

        st.write("Step 4: Calculating monthly momentum returns...")
        final_data = stocks.momentum(filtered_df)
        if use_ff:

            st.write("Step 5: Estimating Fama-French Factor Betas...")

            ff_factors = stocks.get_fama_french_factors(start_date='2010-01-01')

            final_data = stocks.calculate_rolling_betas(final_data, ff_factors)

        
        st.write("Step 6: Running K-Means Clustering...")
        
        final_data = stocks.calculate_clusters(final_data) 

        st.write("Step 7: Generating Charts...")
        all_charts = stocks.plot_all_clusters(final_data)
        # save data in the memory
        st.session_state.final_data = final_data
        st.session_state.all_charts = all_charts

        st.write("Step 8: Selecting Stocks from Cluster 0...")
        fixed_dates = stocks.select_stocks(final_data)

        st.write("Step 9: Optimizing Portfolio & Backtesting...")
        portfolio_results, latest_weights = stocks.portfolio_optimization(final_data, fixed_dates, max_weight = max_weight /100, lookback = lookback_months)

        st.session_state.latest_weights = latest_weights 
        st.session_state.final_comparison = stocks.portfolio_visual(portfolio_results, benchmark_ticker)

        st.write("Step 10: Comparing against Benchmark...")
        final_comparison = stocks.portfolio_visual(portfolio_results, benchmark_ticker)
        st.session_state.final_comparison = final_comparison
        st.session_state.latest_weights = latest_weights



        st.session_state.final_data = final_data
        st.session_state.all_charts = all_charts
        st.session_state.portfolio_results = portfolio_results
        st.session_state.final_comparison = final_comparison




        
        
        status.update(label="Completed!", state="complete", expanded=False)


if st.session_state.final_data is not None:
    st.divider()
    st.subheader("📊 Cluster Analysis & Stock Selection")
    
    # 4.1 Cluster Slider
    all_charts = st.session_state.all_charts
    chart_index = st.slider("Historical Cluster Evolution", 0, len(all_charts)-1, len(all_charts)-1)
    st.pyplot(all_charts[chart_index])

    # 4.2 Performance Graph
    if st.session_state.final_comparison is not None:
        st.divider()
        st.subheader(f"📈 Performance: Strategy vs. {select_label}")

        with st.container(border=True):

            fig = stocks.plot_port(st.session_state.final_comparison)
            st.pyplot(fig)
        
        # perf_data = st.session_state.final_comparison
        # # Undo Log Returns for Cumulative Growth
        # cumulative_ret = np.exp(perf_data.cumsum()) - 1
        
        # fig, ax = plt.subplots(figsize=(16, 7))
        # cumulative_ret.plot(ax=ax)
        
        # ax.set_title(f"Cumulative Return History", fontsize=14)
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        # ax.set_ylabel("Growth (%)")
        # st.pyplot(fig)


    st.divider()
    st.subheader("🤖 AI Portfolio Commentary")

    with st.expander("Click to generate AI Investment Plan", expanded=True):
        if st.button("Generate Advice"):
            with st.spinner("Gemini is analyzing your engine's output..."):
            # We use the session state data so we don't have to rerun the whole math engine
                advice = run_ai_advisor(
                    st.session_state.final_data, 
                    st.session_state.latest_weights, # This contains the strategy returns/weights
                    user_budget, 
                    user_risk, 
                    max_stocks=5 # Or use a fixed number of stocks
                )
                st.markdown(advice)

    # 4.3 Data and Download
    st.divider()
    st.subheader("Raw Data Preview")
    st.dataframe(st.session_state.final_data.tail(50), use_container_width=True)
    
    csv = st.session_state.final_data.to_csv().encode('utf-8')
    st.download_button(label="📥 Download Data", data=csv, file_name='strategy_data.csv')

else:
    st.warning("👈 Adjust settings and click 'Execute Pipeline' in the sidebar to begin.")
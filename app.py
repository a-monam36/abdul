import streamlit as st
import stocks



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
        status.update(label="Completed!", state="complete", expanded=False)


if st.session_state.final_data is not None:
    
    st.divider()


    st.subheader("Results: Top 150 Most Liquid Stocks")
    # Show the tail of the data from memory
    st.dataframe(st.session_state.final_data.tail(50), use_container_width=True)

    st.subheader("📅 Historical Cluster Evolution")
    
    # Use the charts from our memory
    all_charts = st.session_state.all_charts
    
    # Now when you move this slider, the app reruns, but finds the data in session_state!
    chart_index = st.slider("Move slider to see clusters over time", 0, len(all_charts)-1, len(all_charts)-1)
    st.pyplot(all_charts[chart_index])

    # Download Button also stays visible
    csv = st.session_state.final_data.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Processed Data",
        data=csv,
        file_name='quant_momentum_data.csv',
        mime='text/csv',
    )
else:
    # If the user hasn't clicked 'Execute' yet, show this warning
    st.warning("Click the 'Execute Pipeline' button in the sidebar to begin.")

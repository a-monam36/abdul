import streamlit as st
import momentum_engine as engine  # Importing your separate logic file

# The slider stays here in the UI file
years_back = st.sidebar.slider('Select Lookback Period (Years)', 2, 12, 8)

# When the user clicks a button, call the logic from the other file
if st.button('Analyze Stocks'):
    # We pass the 'years_back' variable into the engine function
    df = engine.get_market_data(years_back)
    st.dataframe(df)



st.toggle("formula control", value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible", width="content")


import streamlit as st

# Page Configuration
st.set_page_config(
     page_title="🚧 Coming Soon",
     page_icon="🚧",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "features"
     }
)

st.title("🚧 Coming Soon 🚧")

st.markdown("""
    Please suggest any features you would like to see in this app! Go here to submit a feature request: [GitHub](https://github.com/Rhyzhang/IP-DALLE/issues)
""")

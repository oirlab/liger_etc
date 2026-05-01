import streamlit as st

st.set_page_config(
    page_title="Liger and Iris Exposure Time Calculator",
    layout="wide",
)

def etc_page():
    from pages.etc import ETCPage
    ETCPage()

def user_guide_page():
    from pages.user_guide import UserGuidePage
    UserGuidePage()

pg = st.navigation([
    st.Page(etc_page, title="ETC", icon=":material/calculate:"),
    st.Page(user_guide_page, title="User Guide", icon=":material/menu_book:")
])

pg.run()
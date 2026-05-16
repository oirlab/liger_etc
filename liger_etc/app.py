# from pathlib import Path
# import os
# from liger_iris_drp_resources import download

# BASE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
# CACHE_DIR = BASE / "liger_iris"
# READY = CACHE_DIR / ".ready"

# os.environ["LIGER_IRIS_DRP_RESOURCE_DIR"] = str(CACHE_DIR)

# def ensure_resources():
#     if READY.exists():
#         return

#     if os.environ.get("LIGER_IRIS_DOWNLOAD_RESOURCES", "0") != "1":
#         raise RuntimeError("Resources not present and downloads are disabled")

#     CACHE_DIR.mkdir(parents=True, exist_ok=True)
#     download()
#     READY.touch()

# ensure_resources()

import streamlit as st

st.set_page_config(
    page_title="Liger Exposure Time Calculator",
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
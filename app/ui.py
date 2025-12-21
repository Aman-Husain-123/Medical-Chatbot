import streamlit as st

def pdf_uploader():
    uploaded_files = st.file_uploader("Upload your medical documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files is not None:
        st.success("File uploaded successfully")
    return uploaded_files


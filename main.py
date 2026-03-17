import streamlit as st
import tempfile
import os
from rag import process_pdf, generate_answer

st.title("Research Tool")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

status_placeholder = st.empty()
process_pdf_button = st.sidebar.button("Process PDFs")

if process_pdf_button:
    if not uploaded_files:
        status_placeholder.text("You must upload at least one PDF file.")
    else:
        tmp_paths = []
        tmp_dir = tempfile.mkdtemp()

        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            tmp_paths.append(tmp_path)

        for status in process_pdf(tmp_paths):
            status_placeholder.text(status)

        status_placeholder.text("PDFs processed successfully!")

query = st.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            with st.container(border=True):
                if isinstance(sources, str):
                    sources = [sources]
                for i, source in enumerate(sources, 1):
                    st.markdown(f"{os.path.basename(source)}")
    except RuntimeError:
        status_placeholder.text("You must process the PDFs first.")
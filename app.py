import streamlit as st
from rag_pipeline import process_pdf, build_qa_chain

st.title("📄 RAG PDF Question Answering System")

st.write("Upload a PDF and ask questions about it.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    st.success("PDF uploaded successfully!")

    # Process PDF only once
    if "vector_store" not in st.session_state:

        with st.spinner("Processing document..."):
            vector_store = process_pdf(uploaded_file)

        st.session_state.vector_store = vector_store

    # Build QA chain
    qa_chain = build_qa_chain(st.session_state.vector_store)

    question = st.text_input("Ask a question about the document")

    if st.button("Ask"):

        if question:
            # testing if RAG pipeline is workiing by displaying the chunks retrieved
            docs = st.session_state.vector_store.similarity_search(question)
            for d in docs:
                st.write(d.page_content[:200])
                
            with st.spinner("Generating answer..."):
                response = qa_chain.run(question)

            st.subheader("Answer")
            st.write(response)

        else:
            st.warning("Please enter a question.")
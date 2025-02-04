import streamlit as st
import os
import tempfile
import zipfile
import pathlib

# Import LangChain tools
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Helper function to extract a zip file
def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# Function to load repository files.
# Here we only consider a few common file types (Python, JS, Java, etc.).
def load_codebase(repo_path, allowed_extensions=None):
    documents = []
    if allowed_extensions is None:
        allowed_extensions = [
            ".py", ".js", ".java", ".ts", ".cpp", ".c", ".md", ".txt"
        ]
    for file_path in pathlib.Path(repo_path).rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            try:
                content = file_path.read_text(errors="ignore")
                if content.strip():
                    documents.append({
                        "source": str(file_path),
                        "content": content
                    })
            except Exception as e:
                st.write(f"Error reading {file_path}: {e}")
    return documents

def main():
    st.title("Codebase Documentation Chatbot")
    st.write(
        """
        This app lets you navigate and understand your codebase. Upload your repository as a ZIP file 
        (containing your code and documentation), and then ask questions such as:
        - "Where is the function that handles authentication?"
        - "What does this module do?"
        """
    )

    # Upload a zipped repository
    uploaded_file = st.file_uploader("Upload repository ZIP file", type=["zip"])
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            st.info("Extracting repository...")
            zip_path = os.path.join(tmpdirname, "repo.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            extract_zip(zip_path, tmpdirname)

            st.info("Scanning repository for code files...")
            docs = load_codebase(tmpdirname)
            if not docs:
                st.error("No valid code files were found in the repository.")
                return
            st.success(f"Loaded {len(docs)} documents from the repository.")

            # Prepare documents for embedding
            texts = [doc["content"] for doc in docs]
            metadatas = [{"source": doc["source"]} for doc in docs]

            # Split the text into manageable chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splitted_texts = []
            splitted_metadata = []
            for text, meta in zip(texts, metadatas):
                splits = text_splitter.split_text(text)
                splitted_texts.extend(splits)
                splitted_metadata.extend([meta] * len(splits))

            st.info("Generating embeddings and building vector store index...")
            embeddings = OpenAIEmbeddings()  # Requires OPENAI_API_KEY in your environment
            vectorstore = FAISS.from_texts(splitted_texts, embeddings, metadatas=splitted_metadata)

            # Create a QA chain that will use the indexed documents
            qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(temperature=0),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            st.success("Repository processed. You can now ask questions about your codebase!")

            # User question input
            user_query = st.text_input("Ask a question about your codebase documentation:", "")
            if st.button("Get Answer") and user_query:
                with st.spinner("Generating answer..."):
                    answer = qa_chain.run(user_query)
                st.subheader("Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()

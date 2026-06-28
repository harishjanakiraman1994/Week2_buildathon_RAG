# 1. ALWAYS LOAD DOTENV FIRST (Before any LangChain imports)
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page configuration for a professional layout
st.set_page_config(
    page_title="TN Agriculture Scheme Assistant",
    page_icon="🌾",
    layout="centered"
)


# App Custom Styling/Branding
st.title("🌾 TN Agriculture Scheme Assistant")
st.markdown(
    "Welcome! This AI-powered assistant helps farmers and citizens search through official "
    "**Tamil Nadu Agriculture & Farmers Welfare Department** schemes dynamically using RAG."
)
st.divider()

# Cache the RAG Chain loading so it runs lightning-fast on user interactions
@st.cache_resource
def initialize_rag_chain():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load your existing FAISS index
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 matching contexts
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt_template = """You are an expert assistant helping farmers find information about Tamil Nadu Government Schemes.
Use the following retrieved context pieces to answer the question. If you don't know the answer, say that you don't know. Keep your answer brief, clear, and structured.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    # Build the chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# Initialize backend components
if not os.environ.get("OPENAI_API_KEY"):
    st.error("🔑 Please set your `OPENAI_API_KEY` in the `.env` file to start the application.")
else:
    with st.spinner("🔄 Loading local knowledge base (FAISS Index)..."):
        rag_chain, retriever = initialize_rag_chain()

    # --- UI SEARCH INTERFACE ---
    user_query = st.text_input(
        "💡 Ask a question about the agriculture schemes:",
        placeholder="e.g., What is the funding pattern for Training to Farmers?"
    )

    if user_query:
        with st.spinner("🤖 Searching documents and generating response..."):
            try:
                # 1. Get answer from the LLM
                answer = rag_chain.invoke(user_query)
                
                # 2. Get reference documents for transparency
                referenced_docs = retriever.invoke(user_query)
                
                # Display the primary answer
                st.subheader("📝 Answer:")
                st.info(answer)
                
                # Display source citations beautifully inside an expandable section
                with st.expander("🔍 View Verified Source Information"):
                    for idx, doc in enumerate(referenced_docs):
                        title = doc.metadata.get('title', 'Unknown Scheme')
                        source_url = doc.metadata.get('source', '#')
                        st.markdown(f"**Source {idx+1}: [{title}]({source_url})**")
                        st.caption(f"**Extracted Content Chunk:**\n{doc.page_content[:250]}...")
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
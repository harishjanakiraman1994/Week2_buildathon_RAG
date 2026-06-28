# 🌾 Tamil Nadu Agriculture Scheme Assistant (RAG)

An AI-powered Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS**, and **Streamlit** to help farmers and citizens seamlessly query and explore official agricultural welfare schemes from the Government of Tamil Nadu. 

The application implements an **offline document parsing pipeline** paired with an **online generation loop** utilizing OpenAI's models and tracked through LangSmith telemetry.

---

## 🏗️ Architectural Blueprint

The application architecture is decoupled into two primary core systems:



1. **Offline Ingestion Engine:** Dynamically scrapes the TN Government Scheme Portal, segments raw content into text blocks, and vectorizes data into a local high-performance similarity index.
2. **Online Query Execution:** Pulls relevant context pieces from the database matching the user's input, formats a structured system prompt, invokes `gpt-4o-mini`, and exposes response metrics via LangSmith logs.

---

## 🛠️ Project Directory Tree

```text
├── faiss_index/              # Local binary vector database (auto-generated)
│   ├── index.faiss
│   └── index.pkl
├── .env                      # Application API secrets (must be created manually)
├── app.py                    # Streamlit frontend & LangChain Expression Language (LCEL) chain
├── load_data.py              # Multi-level web scraping module (BeautifulSoup backend)
├── build_full_knowledge_base.py # Ingests, chunks, and creates the bulk vector repository
└── README.md                 # Project technical documentation# Week2_buildathon_RAG

"""
Clean, configurable UI for Medical AI Assistant.
- Ollama is default model
- Simple controls for RAG and performance tuning
"""

import streamlit as st
import requests
import time
from typing import Dict

API_BASE_URL = "http://localhost:8000"

# ---- Page config ----
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
)

# ---- Header ----
st.title("🏥 Medical AI Assistant")
st.caption(
    "Ask medical questions and get AI-assisted answers backed by a medical knowledge base. "
    "This is for educational use only, not medical advice."
)

st.info(
    "⚠️ Medical disclaimer: This tool does not diagnose or treat. "
    "Always consult a licensed healthcare professional."
)

# ---- Sidebar: configuration ----
with st.sidebar:
    st.header("Configuration")

    # Model selection (Ollama default)
    st.subheader("Model")
    model_choice = st.radio(
        "Model",
        options=["ollama", "auto", "openai", "gemini"],
        index=0,  # Ollama by default
        help="Ollama runs locally. Auto will choose the best available provider.",
    )

    # Ollama model & generation controls
    if model_choice in ["ollama", "auto"]:
        st.subheader("Ollama settings")
        ollama_model = st.selectbox(
            "Ollama model",
            options=[
                "llama3.2:3b",
                "llama3.2:latest",
            ],
            index=0,
            help="Smaller models (3b) are faster, larger ones are more capable.",
        )
        max_tokens = st.slider(
            "Max response tokens",
            min_value=128,
            max_value=2048,
            value=512,
            step=64,
            help="Lower values are faster. This limits how long the answer can be.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Lower is more factual and focused; higher is more creative.",
        )
    else:
        ollama_model = None
        max_tokens = None
        temperature = None

    # RAG settings
    st.subheader("Knowledge base (RAG)")
    use_rag = st.checkbox(
        "Use knowledge base",
        value=True,
        help="If disabled, the model will answer from its own knowledge only.",
    )
    top_k = st.slider(
        "Number of retrieved documents (top_k)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Fewer documents = faster, more focused context.",
    )

    st.subheader("Optional API keys")
    openai_key = st.text_input(
        "OpenAI API key (optional)",
        type="password",
        placeholder="sk-...",
    )
    gemini_key = st.text_input(
        "Google Gemini API key (optional)",
        type="password",
        placeholder="AIza...",
    )

    st.subheader("System status")
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_response.status_code == 200:
            health = health_response.json()
            st.write("Backend:", "✅", health.get("status", "unknown").title())
            st.write("Redis:", "✅" if health.get("redis_connected") else "❌")
            st.write("ChromaDB:", "✅" if health.get("chroma_initialized") else "⚠️ (not initialized)")
        else:
            st.write("Backend:", "❌ error")
    except Exception:
        st.write("Backend:", "❌ offline")

    try:
        metrics_response = requests.get(f"{API_BASE_URL}/api/v1/metrics", timeout=2)
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            if metrics.get("total_queries", 0) > 0:
                st.subheader("Session stats")
                st.metric("Total queries", metrics["total_queries"])
                st.metric("Success rate", f"{metrics['success_rate']:.1f}%")
    except Exception:
        pass

# ---- Main area: query + answer ----
st.subheader("Ask a medical question")

query = st.text_area(
    "Your question",
    placeholder="For example: What are the symptoms of diabetes?",
    height=120,
)

col_btn, col_empty = st.columns([1, 3])
with col_btn:
    submit = st.button("🔍 Get answer", type="primary", use_container_width=True)

if submit and not query.strip():
    st.warning("Please enter a question first.")

if submit and query.strip():
    with st.spinner("Processing your question…"):
        try:
            # Prepare API keys
            api_keys: Dict[str, str] = {}
            if openai_key:
                api_keys["openai"] = openai_key
            if gemini_key:
                api_keys["gemini"] = gemini_key

            # Extra options to send to backend for performance tuning
            # You will need to modify your backend to read these from the request if you want them applied.
            extra_options = {
                "ollama_model": ollama_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }

            payload = {
                "query": query,
                "model_choice": model_choice,
                "use_rag": use_rag,
                "api_keys": api_keys or None,
                "extra_options": extra_options,
            }

            submit_response = requests.post(
                f"{API_BASE_URL}/api/v1/query",
                json=payload,
                timeout=300,
            )

            if submit_response.status_code != 200:
                st.error(f"Failed to submit query: {submit_response.text}")
            else:
                task_id = submit_response.json().get("task_id")
                progress = st.progress(0)
                status_placeholder = st.empty()
                result = None

                max_attempts = 300  # up to 5 minutes
                for i in range(max_attempts):
                    time.sleep(1)
                    progress.progress((i + 1) / max_attempts)

                    r = requests.get(
                        f"{API_BASE_URL}/api/v1/task/{task_id}",
                        timeout=30,
                    )
                    if r.status_code != 200:
                        status_placeholder.write("Waiting for worker…")
                        continue

                    result = r.json()
                    status = result.get("status")

                    if status == "SUCCESS":
                        break
                    elif status == "FAILURE":
                        break
                    else:
                        status_placeholder.write(f"Status: {status}…")

                progress.empty()
                status_placeholder.empty()

                if not result:
                    st.warning("No response from backend. Please try again.")
                elif result["status"] == "FAILURE":
                    st.error(f"Query failed: {result.get('error', 'Unknown error')}")
                elif result["status"] == "SUCCESS":
                    st.success("Answer generated successfully")

                    st.markdown("### 📝 Answer")
                    st.write(result.get("answer", ""))

                    st.divider()
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Model", result.get("model_used", "N/A"))
                    with c2:
                        latency = result.get("latency_ms") or 0
                        st.metric("Latency", f"{latency:.0f} ms")
                    with c3:
                        st.metric("RAG", "On" if result.get("rag_used") else "Off")
                    with c4:
                        tokens = result.get("token_count")
                        st.metric("Tokens", tokens if tokens is not None else "N/A")

                    docs = result.get("retrieved_docs") or []
                    if docs:
                        st.divider()
                        st.markdown("### 📚 Retrieved knowledge")
                        with st.expander(f"View {len(docs)} retrieved entries"):
                            for i, doc in enumerate(docs, start=1):
                                st.markdown(f"**Source {i}**")
                                if doc.get("question"):
                                    st.caption(f"Q: {doc['question']}")
                                if doc.get("answer"):
                                    st.caption(f"A: {doc['answer']}")
                                tags = doc.get("tags")
                                if tags:
                                    if isinstance(tags, list):
                                        tag_str = ", ".join(tags)
                                    else:
                                        tag_str = str(tags)
                                    st.caption(f"Tags: {tag_str}")
                                st.markdown("---")

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Medical AI Assistant • Default model: Ollama • Tune performance via sidebar controls.")

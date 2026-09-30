import streamlit as st
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="University Student Support", page_icon="🎓")

@st.cache_resource
def load_data():
    base = "data"
    with open(os.path.join(base, "chatbot_data.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(base, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(base, "tfidf_vectors.pkl"), "rb") as f:
        vectors = pickle.load(f)
    return df, vectorizer, vectors

def get_response(q, df, v, x, threshold=0.20):
    qv = v.transform([q])
    sim = cosine_similarity(qv, x)
    i = sim.argmax()
    score = float(sim[0][i])
    if score < threshold:
        return {"response":"Sorry, I don't have enough information to answer that.",
                "intent":"Unknown", "topic":"Unknown", "score":score}
    row = df.iloc[i]
    return {"response":row["Bot Response"], "intent":row["Intent"],
            "topic":row["Topic"], "score":score}

st.title("🎓 University Student Support Chatbot")
st.write("Ask about registration, exams, financial aid, IT support and other university services.")

try:
    df, vectorizer, vectors = load_data()
except Exception as e:
    st.error(f"Could not load chatbot model files: {e}")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(f"Knowledge base: **{len(df)} questions**")
    st.write("Model: **TF-IDF + Cosine Similarity**")
    st.write("Model files: **PKL artifacts**")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

q = st.chat_input("Ask your question...")
if q:
    with st.chat_message("user"):
        st.write(q)
    st.session_state.messages.append({"role":"user", "content":q})

    r = get_response(q, df, vectorizer, vectors)
    meta = None if r["intent"] == "Unknown" else f"Topic: {r['topic']} | Intent: {r['intent']} | Similarity: {r['score']:.2f}"

    with st.chat_message("assistant"):
        st.write(r["response"])
        if meta:
            st.caption(meta)

    st.session_state.messages.append({"role":"assistant", "content":r["response"], "meta":meta})

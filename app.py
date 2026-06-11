import streamlit as st
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# CONFIG GEMINI
# =========================

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# LOAD VECTOR DATABASE
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "sbs_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# =========================
# UI
# =========================

st.set_page_config(
    page_title="Pensyarah AI SBS10022",
    page_icon="📘"
)

st.title("📘 Pensyarah AI SBS10022")
st.caption("Sains | Kolej Komuniti")

# Butang clear chat

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Papar sejarah chat

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# INPUT SOALAN
# =========================

if prompt_user := st.chat_input("Tanya soalan anda..."):

    # Papar soalan pengguna

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt_user
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt_user)

    # =========================
    # CARI MAKLUMAT DALAM MODUL
    # =========================

    docs = db.similarity_search(
        prompt_user,
        k=3
    )

    context = ""

    for doc in docs:
        context += doc.page_content + "\n\n"

    # =========================
    # PROMPT
    # =========================

    prompt = f"""
Anda ialah Pensyarah SBS10022 Sains.

MAKLUMAT MODUL:
{context}

SOALAN:
{prompt_user}

ARAHAN:

1. Cari jawapan dalam maklumat modul terlebih dahulu.

2. Jika jawapan ditemui dalam modul,
mulakan dengan:

📘 Jawapan Berdasarkan Modul SBS10022

3. Jika maklumat modul mencukupi,
JANGAN tambah maklumat AI.

4. Jika maklumat modul tidak lengkap,
tambahkan:

⚠️ Maklumat Tambahan AI
(Tidak terdapat secara langsung dalam modul SBS10022)

5. Jika langsung tiada maklumat dalam modul,
nyatakan bahawa jawapan berikut adalah daripada pengetahuan umum AI.

6. Gunakan Bahasa Melayu yang mudah difahami pelajar.

7. Jangan ulang isi yang sama.
"""

    # =========================
    # GEMINI RESPONSE
    # =========================

  with st.chat_message("assistant"):

    with st.spinner("Sedang mencari jawapan..."):

        try:
            response = model.generate_content(prompt)

            jawapan = response.text

            st.markdown(jawapan)

        except Exception:

            jawapan = """
⚠️ Sistem AI sedang sibuk atau had penggunaan telah dicapai.

Sila cuba semula dalam beberapa minit.
"""

            st.markdown(jawapan)

    # =========================
    # SIMPAN JAWAPAN
    # =========================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": jawapan
        }
    )

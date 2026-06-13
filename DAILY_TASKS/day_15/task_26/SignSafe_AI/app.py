import fitz
import streamlit as st
from dotenv import load_dotenv

from rag import create_vector_store, ask_question
from analyzer import analyze_agreement

load_dotenv()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="SignSafe AI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0F172A;
    color: white;
}

/* Remove Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Container */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* Dashboard Cards */
.metric-card {
    background: #1E293B;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #94A3B8;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}

/* Summary Cards */
.summary-card {
    background: #1E293B;
    padding: 18px;
    border-radius: 15px;
    margin-bottom: 15px;
    border: 1px solid #334155;
}

/* Section Headers */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Risk Box */
.risk-box {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

/* Chat */
.chat-container {
    background: #1E293B;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "document_text" not in st.session_state:
    st.session_state.document_text = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_text(uploaded_file):

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🛡️ SignSafe AI")

    st.caption(
        "Contract & Agreement Intelligence Platform"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Agreement PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(uploaded_file.name)

        if st.button(
            "🚀 Analyze Agreement",
            use_container_width=True
        ):

            with st.spinner("Reading Agreement..."):

                text = extract_pdf_text(uploaded_file)

                st.session_state.document_text = text

            with st.spinner("Building Vector Database..."):

                chunk_count = create_vector_store(text)

            with st.spinner("Running AI Analysis..."):

                analysis = analyze_agreement(text)

                st.session_state.analysis = analysis

                st.session_state.processed = True

                st.session_state.chunk_count = chunk_count

            st.success(
                f"Processed successfully ({chunk_count} chunks)"
            )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
# 🛡️ SignSafe AI

### Contract & Agreement Intelligence Platform
""")

st.markdown("---")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if st.session_state.processed:

    analysis = st.session_state.analysis

    clauses = analysis.get("clauses", [])

    risk_data = analysis.get(
        "risk_assessment",
        {}
    )

    risk_level = risk_data.get(
        "score",
        "Unknown"
    )

    clause_count = len(clauses)

    status = "Ready"

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                Risk Level
            </div>
            <div class="metric-value">
                {risk_level}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                Clauses Found
            </div>
            <div class="metric-value">
                {clause_count}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                Document Status
            </div>
            <div class="metric-value">
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Summary",
        "⚠️ Risk Analysis",
        "📚 Clauses",
        "🤖 AI Assistant"
    ])

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    with tab1:

        st.markdown(
            '<div class="section-title">Agreement Summary</div>',
            unsafe_allow_html=True
        )

        summary = analysis.get(
            "summary",
            {}
        )

        for title, value in summary.items():

            st.markdown(f"""
            <div class="summary-card">
                <h4>{title.title()}</h4>
                <p>{value}</p>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------
    # RISK
    # --------------------------------------------------

    with tab2:

        st.markdown(
            '<div class="section-title">Risk Assessment</div>',
            unsafe_allow_html=True
        )

        if risk_level.lower() == "high":
            st.error(
                f"🔴 HIGH RISK AGREEMENT"
            )

        elif risk_level.lower() == "medium":
            st.warning(
                f"🟠 MEDIUM RISK AGREEMENT"
            )

        else:
            st.success(
                f"🟢 LOW RISK AGREEMENT"
            )

        st.subheader("Detected Risks")

        risks = risk_data.get(
            "risks",
            []
        )

        if risks:

            for risk in risks:

                st.warning(
                    f"**{risk.get('type')}**\n\n"
                    f"{risk.get('reason')}"
                )

        else:

            st.success(
                "No major risks detected."
            )

    # --------------------------------------------------
    # CLAUSES
    # --------------------------------------------------

    with tab3:

        st.markdown(
            '<div class="section-title">Key Clauses</div>',
            unsafe_allow_html=True
        )

        if clauses:

            for clause in clauses:

                with st.expander(
                    clause.get(
                        "name",
                        "Clause"
                    )
                ):

                    st.write(
                        clause.get(
                            "content",
                            ""
                        )
                    )

        else:

            st.info(
                "No clauses extracted."
            )

    # --------------------------------------------------
    # CHAT
    # --------------------------------------------------

    with tab4:

        st.markdown(
            '<div class="section-title">Agreement Assistant</div>',
            unsafe_allow_html=True
        )

        for role, msg in st.session_state.chat_history:

            with st.chat_message(role):

                st.write(msg)

        question = st.chat_input(
            "Ask about this agreement..."
        )

        if question:

            st.session_state.chat_history.append(
                ("user", question)
            )

            with st.chat_message("user"):
                st.write(question)

            with st.spinner("Analyzing..."):

                answer, sources = ask_question(
                    question
                )

            st.session_state.chat_history.append(
                ("assistant", answer)
            )

            with st.chat_message("assistant"):

                st.write(answer)

                with st.expander(
                    "📌 Retrieved Sources"
                ):

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.markdown(
                            f"**Chunk {i}**"
                        )

                        st.write(source)

                        st.divider()

else:

    st.info(
        "Upload an agreement PDF from the sidebar and click 'Analyze Agreement'."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "SignSafe AI • Agreement Intelligence using RAG + Gemini"
)
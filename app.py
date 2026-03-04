import os
import re
import pdfplumber
import chromadb
import streamlit as st
import plotly.graph_objects as go
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Load models ───────────────────────────────────────────────────────────────
# We now use OpenAI embeddings via API to avoid large model downloads on deployment.

# ── ChromaDB ──────────────────────────────────────────────────────────────────
# Use OpenAI embedding function to avoid loading SentenceTransformers locally
from chromadb.utils import embedding_functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_collection(name="resume_collection_v3", embedding_function=openai_ef)
except:
    collection = chroma_client.create_collection(
        name="resume_collection_v3",
        embedding_function=openai_ef
    )

# ── OpenAI client ─────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ── Section headers to detect ─────────────────────────────────────────────────
SECTION_PATTERNS = [
    "skills", "technical skills", "core competencies",
    "experience", "work experience", "professional experience", "internship",
    "projects", "personal projects", "academic projects",
    "education", "qualifications",
    "certifications", "achievements", "awards",
    "summary", "objective", "profile",
]

SECTION_REGEX = re.compile(
    r"(?im)^(" + "|".join(re.escape(s) for s in SECTION_PATTERNS) + r")\s*[:\-]?\s*$"
)


# ── Helper functions ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def section_aware_chunks(text: str) -> list[dict]:
    """
    Split text into sections based on common resume headings.
    Returns list of dicts: {section, text, chunk_id}
    Falls back to generic chunking if no sections detected.
    """
    lines = text.split("\n")
    sections = []
    current_section = "General"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if SECTION_REGEX.match(stripped):
            if current_lines:
                sections.append({
                    "section": current_section,
                    "text": " ".join(current_lines).strip()
                })
            current_section = stripped.title()
            current_lines = []
        else:
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        sections.append({"section": current_section, "text": " ".join(current_lines).strip()})

    # If no sections were detected, fall back to word-level chunking
    if len(sections) <= 1:
        words = text.split()
        chunk_size, overlap = 300, 50
        sections = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            if chunk:
                sections.append({"section": "General", "text": chunk})

    # Add IDs
    for i, s in enumerate(sections):
        s["chunk_id"] = f"id_{i}"

    return sections


def embed_text(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI API."""
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [data.embedding for data in response.data]


def store_in_chroma(section_chunks: list[dict]) -> None:
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [s["text"] for s in section_chunks]
    embeddings = embed_text(texts)
    metadatas = [{"section": s["section"]} for s in section_chunks]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[s["chunk_id"] for s in section_chunks],
        metadatas=metadatas,
    )


def retrieve_relevant_chunks(job_description: str, top_k: int = 5) -> list[dict]:
    jd_embedding = embed_text([job_description])
    results = collection.query(
        query_embeddings=jd_embedding,
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "section": meta.get("section", "General")})
    return chunks


def analyze_resume(jd: str, retrieved_chunks: list[dict]) -> str:
    context_parts = [f"[{c['section']}]\n{c['text']}" for c in retrieved_chunks]
    context = "\n\n".join(context_parts)

    prompt = f"""You are an AI career assistant.

Job Description:
{jd}

Relevant Resume Sections:
{context}

Please provide:
1. Match Score (0-100%)
2. Missing Skills
3. Key Strengths
4. Suggested Improvements
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def rewrite_resume(jd: str, section_chunks: list[dict]) -> str:
    """
    Rewrite the resume sections to better align with the given job description.
    Returns the full rewritten resume as a string.
    """
    # Build a structured representation of the resume sections
    resume_sections = ""
    for chunk in section_chunks:
        resume_sections += f"\n### {chunk['section']}\n{chunk['text']}\n"

    prompt = f"""You are an expert resume writer and career coach.

You have been given a candidate's resume (split into sections) and a job description.
Your task is to REWRITE the resume to maximize alignment with the job description.

Rules:
- Keep the candidate's real experience, skills, and education — do NOT fabricate anything
- Rephrase bullet points to use keywords and language from the JD
- Strengthen weak or vague descriptions to be more impactful and quantifiable
- Add relevant skills that the candidate likely has based on their experience (only if plausible)
- Maintain a professional, clean format
- Return the full rewritten resume, section by section

Job Description:
{jd}

Candidate's Current Resume:
{resume_sections}

Rewritten Resume:
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.choices[0].message.content


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords (skip stopwords and short words)."""
    stopwords = {
        "the", "and", "for", "with", "are", "have", "has", "that", "this",
        "will", "from", "our", "you", "your", "can", "all", "any", "they",
        "their", "been", "not", "should", "must", "also", "such", "able",
        "into", "each", "more", "well", "work", "role", "team", "new", "its",
        "was", "we", "a", "an", "in", "of", "to", "or", "on", "at", "is",
        "be", "by", "it", "as", "if", "we", "use"
    }
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]*\b", text.lower())
    return [w for w in words if len(w) > 2 and w not in stopwords]


def build_keyword_heatmap(jd: str, resume_text: str):
    """
    Build a keyword heatmap comparing JD keywords vs Resume keywords.
    Returns a plotly figure.
    """
    jd_keywords = Counter(extract_keywords(jd))
    resume_keywords = Counter(extract_keywords(resume_text))

    # Top JD keywords
    top_jd = dict(jd_keywords.most_common(25))

    keywords = list(top_jd.keys())
    jd_counts = [top_jd.get(k, 0) for k in keywords]
    resume_counts = [resume_keywords.get(k, 0) for k in keywords]

    # Labels for hover/display
    labels = ["MISSING" if resume_keywords.get(k, 0) == 0 else f"Found: {resume_keywords[k]}" for k in keywords]
    # Colour: green if present in resume, red if missing
    colors = ["#2ecc71" if resume_keywords.get(k, 0) > 0 else "#e74c3c" for k in keywords]

    fig = go.Figure()
    
    # Add a background "shadow" bar for keywords missing in resume to show red
    # We use a small constant height so the red color is actually visible
    missing_indicator_height = [0.2 if resume_keywords.get(k, 0) == 0 else 0 for k in keywords]
    
    fig.add_trace(go.Bar(
        name="Job Description",
        x=keywords,
        y=jd_counts,
        marker_color="#3498db",
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>JD Frequency: %{y}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        name="Your Resume",
        x=keywords,
        y=[max(c, 0.2) if resume_keywords.get(k, 0) == 0 else c for k, c in zip(keywords, resume_counts)],
        marker_color=colors,
        text=labels,
        textposition="none", # Keep it clean, use hover
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Status: %{text}<br>Resume Frequency: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="🔥 Keyword Heatmap – JD vs Resume",
        barmode="group",
        xaxis_title="Keywords",
        yaxis_title="Frequency",
        xaxis_tickangle=-40,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eee"),
    )
    return fig


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="centered")
st.title("AI Resume + Job Description Matcher")
st.markdown("Upload your **PDF resume** and paste a job description to get an AI-powered analysis.")

# Initialize session state for persistence
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = None
if "chunks" not in st.session_state:
    st.session_state["chunks"] = None
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "detected_sections" not in st.session_state:
    st.session_state["detected_sections"] = None
if "analyzed_jd" not in st.session_state:
    st.session_state["analyzed_jd"] = None
if "retrieved_chunks" not in st.session_state:
    st.session_state["retrieved_chunks"] = None

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
jd = st.text_area("Paste Job Description", height=200)

if st.button("🔍 Analyze", width="stretch"):
    if not uploaded_file:
        st.warning("Please upload a PDF resume.")
    elif not jd.strip():
        st.warning("Please paste a job description.")
    elif not OPENAI_API_KEY:
        st.error("OpenAI API key is not set in the code.")
    else:
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            st.session_state["resume_text"] = resume_text

        if not resume_text.strip():
            st.error("Could not extract text from the PDF. Make sure it's not a scanned image-only PDF.")
        else:
            with st.spinner("Splitting into sections..."):
                chunks = section_aware_chunks(resume_text)
                st.session_state["chunks"] = chunks
                st.session_state["detected_sections"] = list(dict.fromkeys(c["section"] for c in chunks))

            with st.spinner("Storing in ChromaDB..."):
                store_in_chroma(chunks)

            with st.spinner("Retrieving relevant sections..."):
                retrieved = retrieve_relevant_chunks(jd)
                st.session_state["retrieved_chunks"] = retrieved

            with st.spinner("Analyzing with AI..."):
                result = analyze_resume(jd, retrieved)
                st.session_state["analysis_result"] = result
                st.session_state["analyzed_jd"] = jd
            
            # Clear previous rewrite if any
            if "rewritten_resume" in st.session_state:
                del st.session_state["rewritten_resume"]

# ── Display Results from Session State ────────────────────────────────────────

if st.session_state["analysis_result"]:
    resume_text = st.session_state["resume_text"]
    chunks = st.session_state["chunks"]
    detected = st.session_state["detected_sections"]
    retrieved = st.session_state["retrieved_chunks"]
    result = st.session_state["analysis_result"]
    analyzed_jd = st.session_state["analyzed_jd"]

    st.info(f"📂 Detected sections: **{', '.join(detected)}**")

    # ── Keyword Heatmap ───────────────────────────────────────────────
    st.subheader("🔥 Keyword Heatmap")
    fig = build_keyword_heatmap(analyzed_jd, resume_text)
    st.plotly_chart(fig, width="stretch")
    st.caption("🟢 Green = keyword found in your resume &nbsp;|&nbsp; 🔴 Red = missing from your resume")

    # ── AI Analysis ───────────────────────────────────────────────────
    st.subheader("📊 AI Analysis Result")
    st.markdown(result)

    # ── Resume Rewriting ──────────────────────────────────────────────
    st.divider()
    st.subheader("✍️ AI Resume Rewriter")
    st.markdown(
        "Click below to get a **tailored rewrite** of your resume that better matches this job description."
    )

    if st.button("✍️ Rewrite My Resume for This JD", width="stretch"):
        with st.spinner("Rewriting your resume to match the JD..."):
            rewritten = rewrite_resume(jd, chunks)
            st.session_state["rewritten_resume"] = rewritten

    # Show rewritten resume if it exists in session
    if "rewritten_resume" in st.session_state:
        st.subheader("📄 Rewritten Resume")
        st.text_area(
            label="Your tailored resume (copy or download below)",
            value=st.session_state["rewritten_resume"],
            height=500,
        )
        st.download_button(
            label="⬇️ Download Rewritten Resume (.txt)",
            data=st.session_state["rewritten_resume"],
            file_name="rewritten_resume.txt",
            mime="text/plain",
            width="stretch",
        )

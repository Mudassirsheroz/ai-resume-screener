import streamlit as st
from groq import Groq
import PyPDF2
import docx
import json
import pandas as pd
import os
from supabase import create_client

# ── Config ────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

@st.cache_resource
def get_groq():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── CV Reading ────────────────────────────────────────
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def extract_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_cv(uploaded):
    if uploaded.name.endswith(".pdf"):
        return extract_pdf(uploaded)
    elif uploaded.name.endswith(".docx"):
        return extract_docx(uploaded)
    return uploaded.read().decode("utf-8")

# ── Groq Analysis ─────────────────────────────────────
def analyze_with_groq(cv_text, jd_text, name):
    client = get_groq()
    prompt = f"""You are an expert HR recruiter. Analyze this CV against the Job Description.

JOB DESCRIPTION:
{jd_text}

CANDIDATE: {name}
CV:
{cv_text}

Respond ONLY in this exact JSON format, no extra text:
{{
  "score": <number 0-100>,
  "verdict": "<SUITABLE|MAYBE|NOT SUITABLE>",
  "strengths": ["<point1>", "<point2>", "<point3>"],
  "weaknesses": ["<point1>", "<point2>"],
  "summary": "<2 sentence summary>",
  "interview_questions": ["<q1>", "<q2>", "<q3>"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── UI Setup ──────────────────────────────────────────
st.set_page_config(page_title="AI Resume Screener Pro", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.big-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
.subtitle  { color: #888; margin-bottom: 1.5rem; font-size: 1rem; }
.score-card { padding: 1.2rem; border-radius: 12px; text-align: center; margin-bottom: 0.5rem; }
.suitable  { background: #d4edda; border: 2px solid #28a745; }
.maybe     { background: #fff3cd; border: 2px solid #ffc107; }
.nofit     { background: #f8d7da; border: 2px solid #dc3545; }
.login-box { max-width: 400px; margin: 5rem auto; text-align: center; padding: 2rem;
             border-radius: 16px; border: 1px solid #e0e0e0; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

# ── Google Login ──────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# Check URL for auth token
if st.session_state.user is None:
    try:
        supabase = get_supabase()
        params = st.query_params
        if "code" in params:
            code = params["code"]
            res = supabase.auth.exchange_code_for_session({"auth_code": code})
            st.session_state.user = res.user
            st.query_params.clear()
            st.rerun()
    except:
        pass

# ── Login Page ────────────────────────────────────────
if st.session_state.user is None:
    st.markdown("""
    <div class="login-box">
        <div style="font-size:3rem">🧠</div>
        <div style="font-size:1.8rem;font-weight:700;margin:0.5rem 0">AI Resume Screener Pro</div>
        <div style="color:#888;margin-bottom:1.5rem">Login karo aur CVs analyze karo</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔐 Login with Google", use_container_width=True, type="primary"):
            try:
                supabase = get_supabase()
                app_url = "https://cv-screener-pro.streamlit.app"
                res = supabase.auth.sign_in_with_oauth({
                    res = supabase.auth.sign_in_with_oauth({
    "provider": "google",
    "options": {"redirect_to": app_url, "scopes": "email profile"}
})
st.write(res)
            except Exception as e:
                st.error(f"Error: {e}")

# ── Main App ──────────────────────────────────────────
else:
    user = st.session_state.user
    
    # Header
    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="big-title">🧠 AI Resume Screener Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Groq AI se multiple CVs analyze karo</div>', unsafe_allow_html=True)
    with col_user:
        st.markdown(f"👤 **{user.email}**")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Job Description")
        jd_text = st.text_area("JD paste karo", height=300,
            placeholder="e.g. Python developer needed with Django, REST APIs, 3+ years exp...")

    with col2:
        st.subheader("📁 CVs Upload Karo")
        uploaded_files = st.file_uploader(
            "Multiple CVs upload karo (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} CV(s) ready!")
            for f in uploaded_files:
                st.caption(f"• {f.name}")

    st.divider()

    if st.button("🚀 Analyze All CVs", use_container_width=True, type="primary"):
        if not jd_text.strip():
            st.warning("⚠️ Job Description likhna zaroori hai!")
        elif not uploaded_files:
            st.warning("⚠️ Kam se kam ek CV upload karo!")
        else:
            results = []
            progress = st.progress(0, text="AI analyze kar raha hai...")

            for i, uploaded in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files),
                                  text=f"Analyzing: {uploaded.name}...")
                cv_text = read_cv(uploaded)
                name = uploaded.name.rsplit(".", 1)[0]
                try:
                    result = analyze_with_groq(cv_text, jd_text, name)
                    result["name"] = name
                    results.append(result)
                except Exception as e:
                    st.error(f"❌ Error — {uploaded.name}: {e}")

            progress.empty()
            results.sort(key=lambda x: x["score"], reverse=True)

            st.subheader("🏆 Rankings")
            num_cols = min(len(results), 4) if results else 1
            cols = st.columns(num_cols)
            for i, res in enumerate(results):
                with cols[i % num_cols]:
                    verdict = res["verdict"]
                    css = "suitable" if verdict == "SUITABLE" else ("maybe" if verdict == "MAYBE" else "nofit")
                    medal = "🥇" if i==0 else ("🥈" if i==1 else ("🥉" if i==2 else "🎖️"))
                    st.markdown(f"""
<div class="score-card {css}">
  <div style="font-size:2rem;font-weight:700">{res["score"]}%</div>
  <div style="font-size:1.1rem;font-weight:600">{medal} {res["name"]}</div>
  <div style="font-size:0.85rem;margin-top:4px">{verdict}</div>
</div>""", unsafe_allow_html=True)

            st.divider()
            st.subheader("📊 Score Comparison Chart")
            chart_data = pd.DataFrame({
                "Candidate": [r["name"] for r in results],
                "Score": [r["score"] for r in results]
            })
            st.bar_chart(chart_data.set_index("Candidate")["Score"], color="#4A90D9", height=350)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("✅ Suitable", sum(1 for r in results if r["verdict"] == "SUITABLE"))
            col_b.metric("⚠️ Maybe", sum(1 for r in results if r["verdict"] == "MAYBE"))
            col_c.metric("❌ Not Suitable", sum(1 for r in results if r["verdict"] == "NOT SUITABLE"))

            st.divider()
            st.subheader("📋 Detailed Analysis")

            for i, res in enumerate(results):
                medal = "🥇" if i==0 else ("🥈" if i==1 else ("🥉" if i==2 else "📄"))
                with st.expander(f"{medal} #{i+1} — {res['name']} ({res['score']}% — {res['verdict']})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**✅ Strengths**")
                        for s in res["strengths"]:
                            st.markdown(f"- {s}")
                    with c2:
                        st.markdown("**❌ Weaknesses**")
                        for w in res["weaknesses"]:
                            st.markdown(f"- {w}")
                    st.markdown("**📝 AI Summary**")
                    st.info(res["summary"])
                    st.markdown("**❓ Interview Questions**")
                    for q in res["interview_questions"]:
                        st.markdown(f"- {q}")

            st.divider()
            report = "AI RESUME SCREENER PRO — RESULTS\n" + "="*50 + "\n\n"
            report += f"JD:\n{jd_text[:300]}...\n\n"
            report += "RANKINGS:\n"
            for i, res in enumerate(results):
                report += f"\n#{i+1} {res['name']} — {res['score']}% — {res['verdict']}\n"
                report += f"  Summary: {res['summary']}\n"
                report += f"  Strengths: {', '.join(res['strengths'])}\n"
                report += f"  Weaknesses: {', '.join(res['weaknesses'])}\n"
                report += f"  Interview Qs: {', '.join(res['interview_questions'])}\n"

            st.download_button(
                label="📥 Full Report Download Karo",
                data=report,
                file_name="resume_report.txt",
                mime="text/plain",
                use_container_width=True
            )

import streamlit as st
from groq import Groq
import PyPDF2
import docx
import json
import pandas as pd
import os
from supabase import create_client

# ── Configuration ─────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
APP_URL       = os.environ.get("APP_URL", "http://localhost:8501")

@st.cache_resource
def get_groq():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── CV Text Extraction ────────────────────────────────
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_docx(file):
    document = docx.Document(file)
    return "\n".join([para.text for para in document.paragraphs])

def read_cv(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_docx(uploaded_file)
    return uploaded_file.read().decode("utf-8")

# ── AI Analysis ───────────────────────────────────────
def analyze_with_groq(cv_text, jd_text, candidate_name):
    client = get_groq()
    prompt = f"""You are a senior HR recruiter and talent acquisition specialist.
Carefully analyze the candidate's CV against the provided Job Description.

JOB DESCRIPTION:
{jd_text}

CANDIDATE: {candidate_name}
CV:
{cv_text}

Respond ONLY in this exact JSON format with no additional text:
{{
  "score": <integer 0-100>,
  "verdict": "<SUITABLE|MAYBE|NOT SUITABLE>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "summary": "<A concise 2-sentence professional summary>",
  "interview_questions": ["<question 1>", "<question 2>", "<question 3>"]
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

# ── Page Configuration ────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.big-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; color: #1a1a2e; }
.subtitle  { color: #6c757d; margin-bottom: 1.5rem; font-size: 1rem; }

.score-card { padding: 1.4rem; border-radius: 14px; text-align: center; margin-bottom: 0.6rem; transition: transform 0.2s; }
.score-card:hover { transform: translateY(-2px); }
.suitable { background: #d4edda; border: 2px solid #28a745; }
.maybe    { background: #fff3cd; border: 2px solid #ffc107; }
.nofit    { background: #f8d7da; border: 2px solid #dc3545; }

.login-container {
    max-width: 440px; margin: 3rem auto; padding: 2.5rem;
    border-radius: 20px; border: 1px solid #e8e8e8;
    box-shadow: 0 8px 40px rgba(0,0,0,0.10); background: white; text-align: center;
}
.login-logo     { font-size: 3.5rem; margin-bottom: 0.5rem; }
.login-title    { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; color: #1a1a2e; }
.login-subtitle { color: #888; margin-bottom: 2rem; font-size: 0.95rem; }

.google-btn {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    background: white; border: 1.5px solid #dadce0; border-radius: 10px;
    padding: 12px 20px; font-size: 0.95rem; font-weight: 500; color: #3c4043;
    cursor: pointer; width: 100%; transition: all 0.2s;
    text-decoration: none; margin-bottom: 1rem;
}
.google-btn:hover { background: #f8f9fa; box-shadow: 0 2px 10px rgba(0,0,0,0.12); border-color: #bbb; }

.divider { display: flex; align-items: center; gap: 12px; margin: 1.2rem 0; color: #aaa; font-size: 0.85rem; }
.divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: #e8e8e8; }

.user-badge {
    background: #f0f4ff; border: 1px solid #d0dbff; border-radius: 20px;
    padding: 6px 14px; font-size: 0.85rem; color: #3d5afe;
    font-weight: 500; display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ── Handle OAuth Callback ─────────────────────────────
if st.session_state.user is None:
    try:
        supabase = get_supabase()
        params   = st.query_params
        if "code" in params:
            res = supabase.auth.exchange_code_for_session({"auth_code": params["code"]})
            st.session_state.user = res.user
            st.query_params.clear()
            st.rerun()
    except Exception:
        pass

# ══════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════
if st.session_state.user is None:

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-logo">🧠</div>
            <div class="login-title">AI Resume Screener Pro</div>
            <div class="login-subtitle">Sign in to start screening candidates</div>
        </div>
        """, unsafe_allow_html=True)

        # Google OAuth
        try:
            supabase    = get_supabase()
            google_auth = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {"redirect_to": APP_URL}
            })
            st.markdown(f"""
            <a href="{google_auth.url}" class="google-btn">
                <svg width="18" height="18" viewBox="0 0 48 48">
                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                </svg>
                Continue with Google
            </a>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Google login error: {e}")

        st.markdown('<div class="divider">or continue with email</div>', unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            email    = st.text_input("Email Address", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")

            if st.button("Sign In", use_container_width=True, type="primary"):
                if not email or not password:
                    st.warning("Please enter your email and password.")
                else:
                    try:
                        supabase = get_supabase()
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user = res.user
                        st.rerun()
                    except Exception as e:
                        err = str(e).lower()
                        if "invalid" in err or "credentials" in err:
                            st.error("❌ Invalid email or password. Please try again.")
                        elif "confirm" in err or "verified" in err:
                            st.error("❌ Please verify your email address before signing in.")
                        else:
                            st.error(f"❌ Sign in failed: {e}")

            with st.expander("Forgot your password?"):
                reset_email = st.text_input("Enter your email address", key="reset_email", placeholder="you@example.com")
                if st.button("Send Reset Link", use_container_width=True):
                    if reset_email:
                        try:
                            supabase = get_supabase()
                            supabase.auth.reset_password_email(reset_email)
                            st.success("✅ Password reset link sent! Please check your inbox.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.warning("Please enter your email address.")

        with tab_register:
            reg_email     = st.text_input("Email Address", key="reg_email", placeholder="you@example.com")
            reg_password  = st.text_input("Password", type="password", key="reg_pass", placeholder="Minimum 6 characters")
            reg_password2 = st.text_input("Confirm Password", type="password", key="reg_pass2", placeholder="Re-enter your password")

            if st.button("Create Account", use_container_width=True, type="primary"):
                if not reg_email or not reg_password:
                    st.warning("Please fill in all fields.")
                elif reg_password != reg_password2:
                    st.error("❌ Passwords do not match.")
                elif len(reg_password) < 6:
                    st.error("❌ Password must be at least 6 characters.")
                else:
                    try:
                        supabase = get_supabase()
                        res = supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                        if res.user:
                            st.success("✅ Account created! Please verify your email before signing in.")
                        else:
                            st.warning("Please check your inbox to confirm your email.")
                    except Exception as e:
                        err = str(e).lower()
                        if "already" in err or "registered" in err:
                            st.error("❌ This email is already registered. Please sign in instead.")
                        else:
                            st.error(f"❌ Registration failed: {e}")

# ══════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════
else:
    user     = st.session_state.user
    supabase = get_supabase()

    # Header
    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="big-title">🧠 AI Resume Screener Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Screen multiple candidates instantly using Groq AI</div>', unsafe_allow_html=True)
    with col_user:
        st.markdown(f'<div class="user-badge">👤 {user.email}</div>', unsafe_allow_html=True)
        if st.button("Sign Out"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🚀 New Screening", "📋 History", "💼 Saved JDs"])

    # ══════════════════════════════════════════════════
    # TAB 1 — New Screening
    # ══════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("📋 Job Description")
            try:
                saved      = supabase.table("saved_jds").select("*").eq("user_email", user.email).execute()
                saved_list = saved.data if saved.data else []
            except:
                saved_list = []

            jd_text = ""
            if saved_list:
                jd_options  = ["-- Write a new JD --"] + [j["title"] for j in saved_list]
                selected_jd = st.selectbox("Select a saved Job Description", jd_options)
                if selected_jd != "-- Write a new JD --":
                    jd_text = next(j["description"] for j in saved_list if j["title"] == selected_jd)
                    st.text_area("Job Description", value=jd_text, height=260, key="jd_display")
                else:
                    jd_text = st.text_area("Paste your Job Description here", height=260,
                        placeholder="e.g. We are looking for a Python Developer with experience in Django, REST APIs, and PostgreSQL...")
            else:
                jd_text = st.text_area("Paste your Job Description here", height=260,
                    placeholder="e.g. We are looking for a Python Developer with experience in Django, REST APIs, and PostgreSQL...")

        with col2:
            st.subheader("📁 Upload Resumes")
            uploaded_files = st.file_uploader(
                "Upload multiple resumes (PDF, DOCX, or TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} resume(s) ready for analysis")
                for f in uploaded_files:
                    st.caption(f"• {f.name}")

        st.divider()

        if st.button("🚀 Analyze All Resumes", use_container_width=True, type="primary"):
            if not jd_text.strip():
                st.warning("⚠️ Please provide a Job Description before analyzing.")
            elif not uploaded_files:
                st.warning("⚠️ Please upload at least one resume.")
            else:
                results  = []
                progress = st.progress(0, text="Analyzing resumes with AI...")

                for i, uploaded in enumerate(uploaded_files):
                    progress.progress(
                        (i + 1) / len(uploaded_files),
                        text=f"Analyzing: {uploaded.name} ({i+1}/{len(uploaded_files)})"
                    )
                    cv_text = read_cv(uploaded)
                    name    = uploaded.name.rsplit(".", 1)[0]
                    try:
                        result         = analyze_with_groq(cv_text, jd_text, name)
                        result["name"] = name
                        results.append(result)

                        supabase.table("screenings").insert({
                            "user_id":             user.id,
                            "user_email":          user.email,
                            "job_description":     jd_text[:500],
                            "candidate_name":      name,
                            "score":               result["score"],
                            "verdict":             result["verdict"],
                            "strengths":           result["strengths"],
                            "weaknesses":          result["weaknesses"],
                            "summary":             result["summary"],
                            "interview_questions": result["interview_questions"]
                        }).execute()

                    except Exception as e:
                        st.error(f"❌ Failed to analyze {uploaded.name}: {e}")

                progress.empty()
                results.sort(key=lambda x: x["score"], reverse=True)

                st.subheader("🏆 Candidate Rankings")
                num_cols = min(len(results), 4)
                cols     = st.columns(num_cols)
                for i, res in enumerate(results):
                    with cols[i % num_cols]:
                        verdict = res["verdict"]
                        css     = "suitable" if verdict == "SUITABLE" else ("maybe" if verdict == "MAYBE" else "nofit")
                        medal   = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "🎖️"))
                        st.markdown(f"""
<div class="score-card {css}">
  <div style="font-size:2.2rem;font-weight:700">{res["score"]}%</div>
  <div style="font-size:1.05rem;font-weight:600;margin-top:4px">{medal} {res["name"]}</div>
  <div style="font-size:0.82rem;margin-top:6px;opacity:0.85">{verdict}</div>
</div>""", unsafe_allow_html=True)

                st.divider()
                st.subheader("📊 Score Comparison")
                chart_data = pd.DataFrame({
                    "Candidate": [r["name"] for r in results],
                    "Score":     [r["score"] for r in results]
                })
                st.bar_chart(chart_data.set_index("Candidate")["Score"], color="#4A90D9", height=350)

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("✅ Suitable",     sum(1 for r in results if r["verdict"] == "SUITABLE"))
                col_b.metric("⚠️ Maybe",        sum(1 for r in results if r["verdict"] == "MAYBE"))
                col_c.metric("❌ Not Suitable", sum(1 for r in results if r["verdict"] == "NOT SUITABLE"))

                st.divider()
                st.subheader("📋 Detailed Analysis")
                for i, res in enumerate(results):
                    medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "📄"))
                    with st.expander(f"{medal} #{i+1} — {res['name']}  |  Score: {res['score']}%  |  {res['verdict']}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**✅ Key Strengths**")
                            for s in res["strengths"]:
                                st.markdown(f"- {s}")
                        with c2:
                            st.markdown("**⚠️ Areas for Improvement**")
                            for w in res["weaknesses"]:
                                st.markdown(f"- {w}")
                        st.markdown("**📝 AI Summary**")
                        st.info(res["summary"])
                        st.markdown("**❓ Suggested Interview Questions**")
                        for q in res["interview_questions"]:
                            st.markdown(f"- {q}")

                st.divider()
                report  = "AI RESUME SCREENER PRO — SCREENING REPORT\n"
                report += "=" * 60 + "\n\n"
                report += f"Job Description (excerpt):\n{jd_text[:400]}...\n\n"
                report += "CANDIDATE RANKINGS:\n" + "-" * 40 + "\n"
                for i, res in enumerate(results):
                    report += f"\n#{i+1}  {res['name']}\n"
                    report += f"    Score    : {res['score']}%\n"
                    report += f"    Verdict  : {res['verdict']}\n"
                    report += f"    Summary  : {res['summary']}\n"
                    report += f"    Strengths    : {', '.join(res['strengths'])}\n"
                    report += f"    Weaknesses   : {', '.join(res['weaknesses'])}\n"
                    report += f"    Interview Qs : {', '.join(res['interview_questions'])}\n"

                st.download_button(
                    label="📥 Download Full Report",
                    data=report,
                    file_name="screening_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # ══════════════════════════════════════════════════
    # TAB 2 — History
    # ══════════════════════════════════════════════════
    with tab2:
        st.subheader("📋 Screening History")
        try:
            history = supabase.table("screenings") \
                .select("*") \
                .eq("user_email", user.email) \
                .order("created_at", desc=True) \
                .execute()

            if history.data:
                for record in history.data:
                    verdict = record["verdict"]
                    icon    = "✅" if verdict == "SUITABLE" else ("⚠️" if verdict == "MAYBE" else "❌")
                    with st.expander(f"{icon}  {record['candidate_name']}  —  {record['score']}%  —  {verdict}  |  {record['created_at'][:10]}"):
                        st.markdown(f"**Job Description (excerpt):** {record['job_description'][:200]}...")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**✅ Strengths**")
                            for s in record["strengths"]:
                                st.markdown(f"- {s}")
                        with c2:
                            st.markdown("**⚠️ Weaknesses**")
                            for w in record["weaknesses"]:
                                st.markdown(f"- {w}")
                        st.info(record["summary"])
            else:
                st.info("No screening history yet. Analyze your first resume to get started!")
        except Exception as e:
            st.error(f"Failed to load history: {e}")

    # ══════════════════════════════════════════════════
    # TAB 3 — Saved JDs
    # ══════════════════════════════════════════════════
    with tab3:
        st.subheader("💼 Saved Job Descriptions")

        with st.expander("➕ Save a New Job Description"):
            jd_title = st.text_input("Job Title", placeholder="e.g. Senior Python Developer, Data Analyst...")
            jd_desc  = st.text_area("Job Description", height=200,
                placeholder="Paste the full job description here...")
            if st.button("💾 Save Job Description", type="primary"):
                if jd_title.strip() and jd_desc.strip():
                    try:
                        supabase.table("saved_jds").insert({
                            "user_id":     user.id,
                            "user_email":  user.email,
                            "title":       jd_title,
                            "description": jd_desc
                        }).execute()
                        st.success(f"✅ '{jd_title}' saved successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.warning("Please provide both a title and a description.")

        st.divider()

        try:
            saved = supabase.table("saved_jds") \
                .select("*") \
                .eq("user_email", user.email) \
                .order("created_at", desc=True) \
                .execute()

            if saved.data:
                for jd in saved.data:
                    with st.expander(f"💼  {jd['title']}  —  Saved on {jd['created_at'][:10]}"):
                        st.text_area("Content", value=jd["description"], height=160, key=f"jd_{jd['id']}")
                        if st.button("🗑️ Delete", key=f"del_{jd['id']}"):
                            supabase.table("saved_jds").delete().eq("id", jd["id"]).execute()
                            st.success("Job description deleted.")
                            st.rerun()
            else:
                st.info("No saved job descriptions yet. Add one above to reuse it later.")
        except Exception as e:
            st.error(f"Failed to load saved JDs: {e}")

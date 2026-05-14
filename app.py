import streamlit as st
from groq import Groq
import PyPDF2
import docx
import json
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client

# ── Configuration ─────────────────────────────────────
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY")
SUPABASE_URL       = os.environ.get("SUPABASE_URL")
SUPABASE_KEY       = os.environ.get("SUPABASE_KEY")
APP_URL            = os.environ.get("APP_URL", "http://localhost:8501")
GMAIL_ADDRESS      = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

@st.cache_resource
def get_groq():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Email Agent ───────────────────────────────────────
def send_email(to_email, candidate_name, verdict, score, strengths, summary):
    st.write(f"📧 Email bheja ja raha hai: {to_email}")
    try:
        GMAIL      = os.environ.get("GMAIL_ADDRESS")
        GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD")

        if not GMAIL or not GMAIL_PASS:
            st.error("❌ GMAIL_ADDRESS ya GMAIL_APP_PASSWORD secret nahi mili!")
            return False

        if verdict == "SUITABLE":
            subject = "Congratulations! You've been shortlisted"
            body    = f"""Dear {candidate_name},

We are pleased to inform you that you have been shortlisted!

Match Score: {score}%

Key Strengths:
{chr(10).join(f"• {s}" for s in strengths)}

Summary: {summary}

We will contact you shortly for an interview.

Best regards,
AI Hiring Team"""

        elif verdict == "MAYBE":
            subject = "Application Update"
            body    = f"""Dear {candidate_name},

Thank you for applying. Your profile is under further review.

Match Score: {score}%
Summary: {summary}

We will get back to you soon.

Best regards,
AI Hiring Team"""

        else:
            subject = "Application Status Update"
            body    = f"""Dear {candidate_name},

Thank you for your interest. After careful review,
your profile does not match our current requirements.

We appreciate your time and wish you the best.

Best regards,
AI Hiring Team"""

        msg            = MIMEMultipart()
        msg["From"]    = f"AI Hiring <{GMAIL}>"
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(GMAIL, GMAIL_PASS)
        server.sendmail(GMAIL, to_email, msg.as_string())
        server.quit()

        st.success(f"✅ Email sent to {to_email}")
        return True

    except Exception as e:
        st.error(f"❌ Email error: {e}")
        return False

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

# ── Screener Agent ────────────────────────────────────
def screener_agent(cv_text, jd_text, candidate_name):
    prompt = f"""You are an autonomous AI Hiring Agent. Your job is to:
1. Analyze the candidate CV against the Job Description
2. Make a hiring decision
3. Extract candidate email if present in CV
4. Generate interview questions if suitable

JOB DESCRIPTION:
{jd_text}

CANDIDATE: {candidate_name}
CV:
{cv_text}

Respond ONLY in this exact JSON format with no extra text:
{{
  "score": <integer 0-100>,
  "verdict": "<SUITABLE|MAYBE|NOT SUITABLE>",
  "candidate_email": "<email from CV or null>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "summary": "<2 sentence professional summary>",
  "interview_questions": ["<question 1>", "<question 2>", "<question 3>"],
  "agent_action": "<what action the agent decided to take>",
  "reasoning": "<why agent made this decision>"
}}"""
    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000
    )
    raw   = response.choices[0].message.content.strip()
    raw   = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    return json.loads(raw[start:end])

# ── Decision Agent ────────────────────────────────────
def decision_agent(results):
    if not results:
        return None
    summary = "\n".join([
        f"- {r['name']}: Score {r['score']}%, Verdict: {r['verdict']}"
        for r in results
    ])
    prompt = f"""You are a senior HR Decision Agent. Based on these screening results:

{summary}

Make final hiring recommendations. Respond ONLY in JSON with no extra text:
{{
  "top_candidate": "<name>",
  "recommended_for_interview": ["<name1>"],
  "rejected": ["<name1>"],
  "hiring_summary": "<overall assessment in 2 sentences>",
  "next_steps": ["<step 1>", "<step 2>", "<step 3>"]
}}"""
    try:
        response = get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        raw   = response.choices[0].message.content.strip()
        raw   = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {
            "top_candidate": results[0]["name"] if results else "N/A",
            "recommended_for_interview": [r["name"] for r in results if r["verdict"] == "SUITABLE"],
            "rejected":      [r["name"] for r in results if r["verdict"] == "NOT SUITABLE"],
            "hiring_summary": "Agent completed screening. Please review results above.",
            "next_steps":    ["Review candidate rankings", "Schedule interviews", "Send offer letters"]
        }

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Resume Screener Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════
# DARK & MODERN CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp {
    background-color: #0D0F18;
    color: #E8EAF0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #12141F;
    border-right: 1px solid #1E2235;
}

/* ── Main block ── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #12141F;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1E2235;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8890A8;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 18px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #3ECFCF) !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #3ECFCF);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.92rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(108, 99, 255, 0.45);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background-color: #12141F !important;
    color: #E8EAF0 !important;
    border: 1px solid #1E2235 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2) !important;
}

/* ── File uploader ── */
.stFileUploader > div {
    background-color: #12141F;
    border: 2px dashed #6C63FF;
    border-radius: 14px;
    padding: 12px;
    transition: 0.3s;
}
.stFileUploader > div:hover {
    border-color: #3ECFCF;
    background-color: #15172A;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #12141F !important;
    border: 1px solid #1E2235 !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
    font-weight: 500;
}
.streamlit-expanderContent {
    background-color: #12141F !important;
    border: 1px solid #1E2235 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Alerts ── */
.stSuccess {
    background-color: #0A2A1E !important;
    border-left: 4px solid #00C896 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #6EFFD8 !important;
}
.stError {
    background-color: #2A0A0A !important;
    border-left: 4px solid #FF4B4B !important;
    border-radius: 0 10px 10px 0 !important;
}
.stWarning {
    background-color: #2A1F0A !important;
    border-left: 4px solid #FFC107 !important;
    border-radius: 0 10px 10px 0 !important;
}
.stInfo {
    background-color: #0A1530 !important;
    border-left: 4px solid #6C63FF !important;
    border-radius: 0 10px 10px 0 !important;
    color: #A8B4FF !important;
}

/* ── Toggle ── */
.stToggle label { color: #E8EAF0 !important; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6C63FF, #3ECFCF) !important;
    border-radius: 10px;
}
.stProgress > div {
    background-color: #1E2235 !important;
    border-radius: 10px;
}

/* ── Divider ── */
hr { border-color: #1E2235 !important; }

/* ── Headings ── */
h1, h2, h3, h4 { color: #E8EAF0 !important; }

/* ── Bar chart ── */
.stVegaLiteChart { background: #12141F !important; border-radius: 12px; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #1E2235; }

/* ── Custom Components ── */
.hero-header {
    background: linear-gradient(135deg, #12141F 0%, #1A1D35 100%);
    border: 1px solid #1E2235;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #3ECFCF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
    line-height: 1.2;
}
.hero-sub {
    color: #6A7290;
    font-size: 1rem;
    margin: 0;
}
.user-badge {
    background: #12141F;
    border: 1px solid #1E2235;
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #A8B4FF;
    font-weight: 500;
    display: inline-block;
}
.agent-box {
    background: linear-gradient(135deg, #13172E, #1A1D35);
    border: 1px solid #6C63FF44;
    border-left: 4px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    margin: 0.6rem 0;
    color: #A8B4FF;
    font-size: 0.92rem;
}
.agent-thinking {
    background: #14120A;
    border-left: 3px solid #FFC107;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
    font-size: 0.87rem;
    color: #D4B96A;
}
.score-card {
    padding: 1.4rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 0.6rem;
    transition: transform 0.2s;
}
.score-card:hover { transform: translateY(-3px); }
.suitable {
    background: linear-gradient(135deg, #0A2A1E, #0D3525);
    border: 1px solid #00C89644;
    box-shadow: 0 4px 20px rgba(0, 200, 150, 0.1);
}
.maybe {
    background: linear-gradient(135deg, #2A1F0A, #352808);
    border: 1px solid #FFC10744;
    box-shadow: 0 4px 20px rgba(255, 193, 7, 0.1);
}
.nofit {
    background: linear-gradient(135deg, #2A0A0A, #350D0D);
    border: 1px solid #FF4B4B44;
    box-shadow: 0 4px 20px rgba(255, 75, 75, 0.1);
}
.decision-box {
    background: linear-gradient(135deg, #0A2A1E, #0D3525);
    border: 1px solid #00C89644;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    color: #E8EAF0;
}
.login-container {
    max-width: 440px;
    margin: 3rem auto;
    padding: 2.5rem;
    border-radius: 24px;
    border: 1px solid #1E2235;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    background: #12141F;
    text-align: center;
}
.google-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    background: #1A1D2E;
    border: 1px solid #2A2D40;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 0.95rem;
    font-weight: 500;
    color: #E8EAF0;
    cursor: pointer;
    width: 100%;
    transition: all 0.2s;
    text-decoration: none;
    margin-bottom: 1rem;
}
.google-btn:hover {
    background: #20243A;
    border-color: #6C63FF;
    color: #E8EAF0;
}
.divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.2rem 0;
    color: #3A3F55;
    font-size: 0.85rem;
}
.divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E2235;
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
            <div style="font-size:3.5rem; margin-bottom: 0.5rem;">🤖</div>
            <div style="font-size:1.8rem; font-weight:800;
                background: linear-gradient(135deg, #6C63FF, #3ECFCF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
                AI Resume Agent
            </div>
            <div style="color:#4A5070; margin-bottom:2rem; font-size:0.95rem;">
                Sign in to start screening candidates
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            supabase    = get_supabase()
            google_auth = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options":  {"redirect_to": APP_URL}
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
            email    = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Sign In", use_container_width=True, type="primary"):
                if not email or not password:
                    st.warning("Please enter email and password.")
                else:
                    try:
                        supabase = get_supabase()
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user = res.user
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Sign in failed: {e}")

        with tab_register:
            reg_email    = st.text_input("Email", key="reg_email", placeholder="you@example.com")
            reg_password = st.text_input("Password", type="password", key="reg_pass")
            reg_pass2    = st.text_input("Confirm Password", type="password", key="reg_pass2")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if reg_password != reg_pass2:
                    st.error("❌ Passwords do not match.")
                else:
                    try:
                        supabase = get_supabase()
                        res = supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                        st.success("✅ Account created! Please verify your email.")
                    except Exception as e:
                        st.error(f"❌ Registration failed: {e}")

# ══════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════
else:
    user     = st.session_state.user
    supabase = get_supabase()

    # ── Hero Header ──
    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.markdown("""
        <div class="hero-header">
            <div class="hero-title">🤖 AI Resume Screener</div>
            <p class="hero-sub">Autonomous AI Agent — screens, decides & emails candidates automatically</p>
        </div>
        """, unsafe_allow_html=True)
    with col_user:
        st.markdown(f'<div class="user-badge" style="margin-top:1rem">👤 {user.email}</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Sign Out"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🤖 Agent Screening", "📋 History", "💼 Saved JDs"])

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
                selected_jd = st.selectbox("Select saved JD", jd_options)
                if selected_jd != "-- Write a new JD --":
                    jd_text = next(j["description"] for j in saved_list if j["title"] == selected_jd)
                    st.text_area("Job Description", value=jd_text, height=250, key="jd_display")
                else:
                    jd_text = st.text_area("Paste Job Description", height=250)
            else:
                jd_text = st.text_area("Paste Job Description", height=250,
                    placeholder="e.g. Python Developer needed with Django, REST APIs, 3+ years experience...")

        with col2:
            st.subheader("📁 Upload CVs")
            uploaded_files = st.file_uploader(
                "Upload resumes (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )
            send_emails = st.toggle("📧 Auto-email candidates", value=False,
                help="Agent will automatically email candidates based on verdict")
            if send_emails:
                st.info("📧 Agent will send emails automatically after screening!")
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} CV(s) ready")
                for f in uploaded_files:
                    st.caption(f"• {f.name}")

        st.divider()

        if st.button("🚀 Launch Agent", use_container_width=True, type="primary"):
            if not jd_text.strip():
                st.warning("⚠️ Please provide a Job Description.")
            elif not uploaded_files:
                st.warning("⚠️ Please upload at least one CV.")
            else:
                st.markdown('<div class="agent-box">🤖 <b>Agent Activated</b> — Starting autonomous screening pipeline...</div>', unsafe_allow_html=True)

                results  = []
                progress = st.progress(0, text="Agent initializing...")

                for i, uploaded in enumerate(uploaded_files):
                    progress.progress(
                        (i + 1) / len(uploaded_files),
                        text=f"🤖 Agent analyzing: {uploaded.name}"
                    )
                    st.markdown(f'<div class="agent-thinking">🧠 Agent thinking... Reading CV: <b>{uploaded.name}</b></div>', unsafe_allow_html=True)

                    cv_text = read_cv(uploaded)
                    name    = uploaded.name.rsplit(".", 1)[0]

                    try:
                        result         = screener_agent(cv_text, jd_text, name)
                        result["name"] = name
                        results.append(result)

                        st.markdown(f'<div class="agent-thinking">⚡ Agent decision for <b>{name}</b>: {result["verdict"]} ({result["score"]}%) — {result["agent_action"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="agent-thinking">💭 Reasoning: {result["reasoning"]}</div>', unsafe_allow_html=True)

                        if send_emails and result.get("candidate_email"):
                            send_email(
                                result["candidate_email"],
                                name,
                                result["verdict"],
                                result["score"],
                                result["strengths"],
                                result["summary"]
                            )

                        try:
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
                        except:
                            pass

                    except Exception as e:
                        st.error(f"❌ Agent failed for {uploaded.name}: {e}")

                progress.empty()

                st.markdown('<div class="agent-box">🧠 <b>Decision Agent</b> — Making final hiring recommendations...</div>', unsafe_allow_html=True)
                decision = decision_agent(results)

                if decision:
                    st.markdown(f"""
<div class="decision-box">
<h4 style="margin-top:0; background: linear-gradient(135deg, #6C63FF, #3ECFCF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    🤖 Agent Final Decision
</h4>
<b style="color:#A8B4FF">Top Candidate:</b> <span style="color:#E8EAF0">{decision.get('top_candidate', 'N/A')}</span><br>
<b style="color:#A8B4FF">Recommended for Interview:</b> <span style="color:#6EFFD8">{', '.join(decision.get('recommended_for_interview', []))}</span><br>
<b style="color:#A8B4FF">Rejected:</b> <span style="color:#FF8080">{', '.join(decision.get('rejected', []))}</span><br><br>
<b style="color:#A8B4FF">Hiring Summary:</b><br>
<span style="color:#C8CADB">{decision.get('hiring_summary', '')}</span><br><br>
<b style="color:#A8B4FF">Next Steps:</b><br>
<span style="color:#C8CADB">{"".join(f"• {s}<br>" for s in decision.get('next_steps', []))}</span>
</div>""", unsafe_allow_html=True)

                results.sort(key=lambda x: x["score"], reverse=True)

                st.subheader("🏆 Agent Rankings")
                num_cols = min(len(results), 4)
                cols     = st.columns(num_cols)
                for i, res in enumerate(results):
                    with cols[i % num_cols]:
                        verdict = res["verdict"]
                        css     = "suitable" if verdict == "SUITABLE" else ("maybe" if verdict == "MAYBE" else "nofit")
                        medal   = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "🎖️"))
                        score_color = "#6EFFD8" if verdict == "SUITABLE" else ("#FFD97D" if verdict == "MAYBE" else "#FF8080")
                        st.markdown(f"""
<div class="score-card {css}">
  <div style="font-size:2.2rem; font-weight:800; color:{score_color}">{res["score"]}%</div>
  <div style="font-size:1rem; font-weight:600; color:#E8EAF0; margin: 6px 0">{medal} {res["name"]}</div>
  <div style="font-size:0.82rem; color:{score_color}; opacity:0.9; font-weight:500">{verdict}</div>
</div>""", unsafe_allow_html=True)

                st.divider()
                st.subheader("📊 Score Comparison")
                chart_data = pd.DataFrame({
                    "Candidate": [r["name"] for r in results],
                    "Score":     [r["score"] for r in results]
                })
                st.bar_chart(chart_data.set_index("Candidate")["Score"], height=300)

                st.subheader("📋 Detailed Analysis")
                for i, res in enumerate(results):
                    medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "📄"))
                    with st.expander(f"{medal} {res['name']} | {res['score']}% | {res['verdict']}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**✅ Strengths**")
                            for s in res["strengths"]:
                                st.markdown(f"- {s}")
                        with c2:
                            st.markdown("**⚠️ Weaknesses**")
                            for w in res["weaknesses"]:
                                st.markdown(f"- {w}")
                        st.info(res["summary"])
                        if res.get("candidate_email"):
                            st.markdown(f"**📧 Candidate Email:** {res['candidate_email']}")
                        st.markdown("**❓ Interview Questions**")
                        for q in res["interview_questions"]:
                            st.markdown(f"- {q}")

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
                    with st.expander(f"{icon} {record['candidate_name']} — {record['score']}% — {record['created_at'][:10]}"):
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
                st.info("No history yet!")
        except Exception as e:
            st.error(f"Failed to load history: {e}")

    with tab3:
        st.subheader("💼 Saved Job Descriptions")
        with st.expander("➕ Save New JD"):
            jd_title = st.text_input("Job Title", placeholder="e.g. Senior Python Developer")
            jd_desc  = st.text_area("Job Description", height=200)
            if st.button("💾 Save", type="primary"):
                if jd_title.strip() and jd_desc.strip():
                    try:
                        supabase.table("saved_jds").insert({
                            "user_id":     user.id,
                            "user_email":  user.email,
                            "title":       jd_title,
                            "description": jd_desc
                        }).execute()
                        st.success(f"✅ '{jd_title}' saved!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
                else:
                    st.warning("Fill both fields.")
        st.divider()
        try:
            saved = supabase.table("saved_jds") \
                .select("*") \
                .eq("user_email", user.email) \
                .order("created_at", desc=True) \
                .execute()
            if saved.data:
                for jd in saved.data:
                    with st.expander(f"💼 {jd['title']} — {jd['created_at'][:10]}"):
                        st.text_area("Content", value=jd["description"], height=150, key=f"jd_{jd['id']}")
                        if st.button("🗑️ Delete", key=f"del_{jd['id']}"):
                            supabase.table("saved_jds").delete().eq("id", jd["id"]).execute()
                            st.rerun()
            else:
                st.info("No saved JDs yet!")
        except Exception as e:
            st.error(f"Failed: {e}")

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
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Application Update — {candidate_name}"
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = to_email

        if verdict == "SUITABLE":
            body = f"""
Dear {candidate_name},

We are pleased to inform you that after reviewing your application, 
you have been shortlisted for the next round!

Match Score: {score}%

Key Strengths we noticed:
{chr(10).join(f"• {s}" for s in strengths)}

Summary:
{summary}

We will contact you shortly to schedule an interview.

Best regards,
AI Hiring Team
            """
        elif verdict == "MAYBE":
            body = f"""
Dear {candidate_name},

Thank you for applying. Your profile is under further review.

Match Score: {score}%
Summary: {summary}

We will get back to you soon.

Best regards,
AI Hiring Team
            """
        else:
            body = f"""
Dear {candidate_name},

Thank you for your interest. After careful review, 
we regret to inform you that your profile does not 
match our current requirements.

We appreciate your time and wish you the best.

Best regards,
AI Hiring Team
            """

        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
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
    client = get_groq()
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

Respond ONLY in this exact JSON format:
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
        temperature=0.3,
        max_tokens=1000
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Decision Agent ────────────────────────────────────
def decision_agent(results):
    if not results:
        return None
    client   = get_groq()
    summary  = "\n".join([
        f"- {r['name']}: Score {r['score']}%, Verdict: {r['verdict']}"
        for r in results
    ])
    prompt = f"""You are a senior HR Decision Agent. Based on these screening results:

{summary}

Make final hiring recommendations. Respond in JSON:
{{
  "top_candidate": "<name>",
  "recommended_for_interview": ["<name1>", "<name2>"],
  "rejected": ["<name1>"],
  "hiring_summary": "<overall assessment in 2 sentences>",
  "next_steps": ["<step 1>", "<step 2>", "<step 3>"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Page Configuration ────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.big-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; color: #1a1a2e; }
.subtitle  { color: #6c757d; margin-bottom: 1.5rem; font-size: 1rem; }
.agent-box { background: #f0f4ff; border: 2px solid #4A90D9; border-radius: 12px; padding: 1rem 1.4rem; margin: 0.5rem 0; }
.agent-thinking { background: #fff8e1; border-left: 4px solid #ffc107; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; font-size: 0.9rem; }
.score-card { padding: 1.4rem; border-radius: 14px; text-align: center; margin-bottom: 0.6rem; }
.suitable { background: #d4edda; border: 2px solid #28a745; }
.maybe    { background: #fff3cd; border: 2px solid #ffc107; }
.nofit    { background: #f8d7da; border: 2px solid #dc3545; }
.decision-box { background: #e8f5e9; border: 2px solid #4CAF50; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0; }
.user-badge { background: #f0f4ff; border: 1px solid #d0dbff; border-radius: 20px; padding: 6px 14px; font-size: 0.85rem; color: #3d5afe; font-weight: 500; display: inline-block; }
.login-container { max-width: 440px; margin: 3rem auto; padding: 2.5rem; border-radius: 20px; border: 1px solid #e8e8e8; box-shadow: 0 8px 40px rgba(0,0,0,0.10); background: white; text-align: center; }
.google-btn { display: flex; align-items: center; justify-content: center; gap: 10px; background: white; border: 1.5px solid #dadce0; border-radius: 10px; padding: 12px 20px; font-size: 0.95rem; font-weight: 500; color: #3c4043; cursor: pointer; width: 100%; transition: all 0.2s; text-decoration: none; margin-bottom: 1rem; }
.divider { display: flex; align-items: center; gap: 12px; margin: 1.2rem 0; color: #aaa; font-size: 0.85rem; }
.divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: #e8e8e8; }
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
            <div style="font-size:3.5rem">🤖</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1a1a2e">AI Resume Agent</div>
            <div style="color:#888;margin-bottom:2rem">Sign in to start screening</div>
        </div>
        """, unsafe_allow_html=True)

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

    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="big-title">🤖 AI Resume Screening Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Autonomous AI Agent — screens, decides, and emails candidates automatically</div>', unsafe_allow_html=True)
    with col_user:
        st.markdown(f'<div class="user-badge">👤 {user.email}</div>', unsafe_allow_html=True)
        if st.button("Sign Out"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🤖 Agent Screening", "📋 History", "💼 Saved JDs"])

    # ══════════════════════════════════════════════════
    # TAB 1 — Agent Screening
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

        if st.button("🤖 Launch Agent", use_container_width=True, type="primary"):
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

                        # Email Agent
                        if send_emails and result.get("candidate_email"):
                            email_sent = send_email(
                                result["candidate_email"],
                                name,
                                result["verdict"],
                                result["score"],
                                result["strengths"],
                                result["summary"]
                            )
                            if email_sent:
                                st.markdown(f'<div class="agent-thinking">📧 Email Agent: Email sent to <b>{result["candidate_email"]}</b></div>', unsafe_allow_html=True)

                        # Save to Supabase
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

                # Decision Agent
                st.markdown('<div class="agent-box">🧠 <b>Decision Agent</b> — Making final hiring recommendations...</div>', unsafe_allow_html=True)

                decision = decision_agent(results)

                if decision:
                    st.markdown(f"""
<div class="decision-box">
<h4>🤖 Agent Final Decision</h4>
<b>Top Candidate:</b> {decision.get('top_candidate', 'N/A')}<br>
<b>Recommended for Interview:</b> {', '.join(decision.get('recommended_for_interview', []))}<br>
<b>Rejected:</b> {', '.join(decision.get('rejected', []))}<br><br>
<b>Hiring Summary:</b> {decision.get('hiring_summary', '')}<br><br>
<b>Next Steps:</b><br>
{"".join(f"• {s}<br>" for s in decision.get('next_steps', []))}
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
                        st.markdown(f"""
<div class="score-card {css}">
  <div style="font-size:2rem;font-weight:700">{res["score"]}%</div>
  <div style="font-size:1rem;font-weight:600">{medal} {res["name"]}</div>
  <div style="font-size:0.82rem;opacity:0.85">{verdict}</div>
</div>""", unsafe_allow_html=True)

                st.divider()
                st.subheader("📊 Score Comparison")
                chart_data = pd.DataFrame({
                    "Candidate": [r["name"] for r in results],
                    "Score":     [r["score"] for r in results]
                })
                st.bar_chart(chart_data.set_index("Candidate")["Score"], color="#4A90D9", height=300)

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

    # ══════════════════════════════════════════════════
    # TAB 3 — Saved JDs
    # ══════════════════════════════════════════════════
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

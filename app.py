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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.big-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
.subtitle  { color: #888; margin-bottom: 1.5rem; font-size: 1rem; }

.score-card { padding: 1.2rem; border-radius: 12px; text-align: center; margin-bottom: 0.5rem; }
.suitable  { background: #d4edda; border: 2px solid #28a745; }
.maybe     { background: #fff3cd; border: 2px solid #ffc107; }
.nofit     { background: #f8d7da; border: 2px solid #dc3545; }

.login-container {
    max-width: 420px;
    margin: 3rem auto;
    padding: 2.5rem;
    border-radius: 20px;
    border: 1px solid #e8e8e8;
    box-shadow: 0 8px 32px rgba(0,0,0,0.10);
    background: white;
    text-align: center;
}

.login-logo { font-size: 3.5rem; margin-bottom: 0.5rem; }
.login-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; color: #1a1a1a; }
.login-subtitle { color: #888; margin-bottom: 2rem; font-size: 0.95rem; }

.google-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    background: white;
    border: 1.5px solid #dadce0;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 0.95rem;
    font-weight: 500;
    color: #3c4043;
    cursor: pointer;
    width: 100%;
    transition: all 0.2s;
    text-decoration: none;
    margin-bottom: 1rem;
}
.google-btn:hover {
    background: #f8f9fa;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    border-color: #c0c0c0;
}

.divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.2rem 0;
    color: #aaa;
    font-size: 0.85rem;
}
.divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e8e8e8;
}

.user-badge {
    background: #f0f4ff;
    border: 1px solid #d0dbff;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.85rem;
    color: #3d5afe;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ── Handle OAuth Callback ─────────────────────────────
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
    except Exception as e:
        pass

# ── Login Page ────────────────────────────────────────
if st.session_state.user is None:

    # Get the app URL for Google OAuth redirect
    app_url = os.environ.get("APP_URL", "http://localhost:8501")

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-logo">🧠</div>
            <div class="login-title">AI Resume Screener Pro</div>
            <div class="login-subtitle">Login karo aur CVs analyze karo</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Google Login Button ──
        try:
            supabase = get_supabase()
            google_url = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": app_url
                }
            })
            st.markdown(f"""
            <a href="{google_url.url}" class="google-btn">
                <svg width="18" height="18" viewBox="0 0 48 48">
                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                </svg>
                Google se Login Karo
            </a>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Google login setup error: {e}")

        st.markdown('<div class="divider">ya email se</div>', unsafe_allow_html=True)

        # ── Email/Password Tabs ──
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            email = st.text_input("Email", key="login_email", placeholder="aapki@email.com")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")

            if st.button("Login", use_container_width=True, type="primary", key="login_btn"):
                if not email or not password:
                    st.warning("⚠️ Email aur password daalo!")
                else:
                    try:
                        supabase = get_supabase()
                        res = supabase.auth.sign_in_with_password({
                            "email": email,
                            "password": password
                        })
                        st.session_state.user = res.user
                        st.rerun()
                    except Exception as e:
                        err = str(e).lower()
                        if "invalid" in err or "credentials" in err:
                            st.error("❌ Email ya password galat hai!")
                        elif "confirm" in err or "verified" in err:
                            st.error("❌ Pehle email confirm karo — inbox check karo!")
                        else:
                            st.error(f"❌ Login failed: {e}")

            # Forgot Password
            with st.expander("🔑 Password bhool gaye?"):
                reset_email = st.text_input("Apni email daalo", key="reset_email", placeholder="aapki@email.com")
                if st.button("Reset Link Bhejo", use_container_width=True):
                    if reset_email:
                        try:
                            supabase = get_supabase()
                            supabase.auth.reset_password_email(reset_email)
                            st.success("✅ Password reset link bhej diya — email check karo!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.warning("Email daalo!")

        with tab_register:
            reg_email = st.text_input("Email", key="reg_email", placeholder="aapki@email.com")
            reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Min 6 characters")
            reg_password2 = st.text_input("Password confirm karo", type="password", key="reg_pass2", placeholder="Dobara likho")

            if st.button("Register", use_container_width=True, type="primary", key="reg_btn"):
                if not reg_email or not reg_password:
                    st.warning("⚠️ Email aur password daalo!")
                elif reg_password != reg_password2:
                    st.error("❌ Passwords match nahi kar rahe!")
                elif len(reg_password) < 6:
                    st.error("❌ Password kam se kam 6 characters ka hona chahiye!")
                else:
                    try:
                        supabase = get_supabase()
                        res = supabase.auth.sign_up({
                            "email": reg_email,
                            "password": reg_password
                        })
                        if res.user:
                            st.success("✅ Account ban gaya! Ab login karo.")
                        else:
                            st.warning("⚠️ Email confirm karo — inbox check karo!")
                    except Exception as e:
                        err = str(e).lower()
                        if "already" in err or "registered" in err:
                            st.error("❌ Yeh email pehle se registered hai — login karo!")
                        else:
                            st.error(f"❌ Register failed: {e}")

# ── Main App ──────────────────────────────────────────
else:
    user = st.session_state.user
    supabase = get_supabase()

    # Header
    col_title, col_user = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="big-title">🧠 AI Resume Screener Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Groq AI se multiple CVs analyze karo</div>', unsafe_allow_html=True)
    with col_user:
        st.markdown(f'<div class="user-badge">👤 {user.email}</div>', unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

    # ── Tabs ──────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🚀 New Screening", "📋 History", "💼 Saved JDs"])

    # ══════════════════════════════════════════════════
    # TAB 1 — New Screening
    # ══════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("📋 Job Description")

            # Saved JDs dropdown
            try:
                saved = supabase.table("saved_jds").select("*").eq("user_email", user.email).execute()
                saved_list = saved.data if saved.data else []
            except:
                saved_list = []

            jd_text = ""
            if saved_list:
                jd_options = ["-- Naya likho --"] + [j["title"] for j in saved_list]
                selected_jd = st.selectbox("Saved JD select karo", jd_options)
                if selected_jd != "-- Naya likho --":
                    jd_text = next(j["description"] for j in saved_list if j["title"] == selected_jd)
                    st.text_area("JD", value=jd_text, height=250, key="jd_display")
                else:
                    jd_text = st.text_area("JD paste karo", height=250,
                        placeholder="e.g. Python developer needed with Django, REST APIs...")
            else:
                jd_text = st.text_area("JD paste karo", height=250,
                    placeholder="e.g. Python developer needed with Django, REST APIs...")

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

                        # Save to Supabase
                        supabase.table("screenings").insert({
                            "user_id": user.id,
                            "user_email": user.email,
                            "job_description": jd_text[:500],
                            "candidate_name": name,
                            "score": result["score"],
                            "verdict": result["verdict"],
                            "strengths": result["strengths"],
                            "weaknesses": result["weaknesses"],
                            "summary": result["summary"],
                            "interview_questions": result["interview_questions"]
                        }).execute()

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

    # ══════════════════════════════════════════════════
    # TAB 2 — History
    # ══════════════════════════════════════════════════
    with tab2:
        st.subheader("📋 Screening History")
        try:
            history = supabase.table("screenings").select("*").eq("user_email", user.email).order("created_at", desc=True).execute()
            if history.data:
                for record in history.data:
                    verdict = record["verdict"]
                    with st.expander(f"📄 {record['candidate_name']} — {record['score']}% — {verdict} | {record['created_at'][:10]}"):
                        st.markdown(f"**JD:** {record['job_description'][:200]}...")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**✅ Strengths**")
                            for s in record["strengths"]:
                                st.markdown(f"- {s}")
                        with c2:
                            st.markdown("**❌ Weaknesses**")
                            for w in record["weaknesses"]:
                                st.markdown(f"- {w}")
                        st.info(record["summary"])
            else:
                st.info("Abhi koi screening nahi — pehli CV analyze karo!")
        except Exception as e:
            st.error(f"Error: {e}")

    # ══════════════════════════════════════════════════
    # TAB 3 — Saved JDs
    # ══════════════════════════════════════════════════
    with tab3:
        st.subheader("💼 Saved Job Descriptions")

        with st.expander("➕ Naya JD Save Karo"):
            jd_title = st.text_input("JD Title", placeholder="e.g. Python Developer, Data Analyst...")
            jd_desc = st.text_area("Job Description", height=200,
                placeholder="Poori job description paste karo...")
            if st.button("💾 Save JD", type="primary"):
                if jd_title.strip() and jd_desc.strip():
                    try:
                        supabase.table("saved_jds").insert({
                            "user_id": user.id,
                            "user_email": user.email,
                            "title": jd_title,
                            "description": jd_desc
                        }).execute()
                        st.success(f"✅ '{jd_title}' save ho gaya!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Title aur Description dono daalo!")

        st.divider()

        try:
            saved = supabase.table("saved_jds").select("*").eq("user_email", user.email).order("created_at", desc=True).execute()
            if saved.data:
                for jd in saved.data:
                    with st.expander(f"💼 {jd['title']} — {jd['created_at'][:10]}"):
                        st.text_area("JD Content", value=jd["description"], height=150, key=f"jd_{jd['id']}")
                        if st.button(f"🗑️ Delete", key=f"del_{jd['id']}"):
                            supabase.table("saved_jds").delete().eq("id", jd["id"]).execute()
                            st.success("Deleted!")
                            st.rerun()
            else:
                st.info("Koi JD save nahi — upar se add karo!")
        except Exception as e:
            st.error(f"Error: {e}")

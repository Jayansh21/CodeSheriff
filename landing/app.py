"""
CodeSheriff Landing Page
Deployed on HuggingFace Spaces (Streamlit)

This page serves three purposes:
1. Explain what CodeSheriff does to developers who find it
2. Let visitors try CodeSheriff live without installing anything
3. Direct interested users to install the GitHub App
"""

import os

import httpx
import streamlit as st

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="CodeSheriff — AI Code Reviewer",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -- Backend URL — set via Space secret after Render deployment -------------
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://your-render-app.onrender.com",
)

# -- Sample diff for the live demo -----------------------------------------
SAMPLE_DIFF = """--- a/app/user_service.py
+++ b/app/user_service.py
@@ -1,12 +1,12 @@
 import sqlite3

 def get_user(user_id):
-    query = "SELECT * FROM users WHERE id = " + user_id
-    conn = sqlite3.connect("users.db")
-    result = conn.execute(query)
-    return result.fetchone().name
+    query = "SELECT * FROM users WHERE id = ?"
+    conn = sqlite3.connect("users.db")
+    result = conn.execute(query, (user_id,))
+    row = result.fetchone()
+    return row.name if row else None

 def process_items(items):
     total = 0
-    for i in range(len(items) + 1):
+    for i in range(len(items)):
         total += items[i]
     return total"""


# ═══════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='font-size: 3rem;'>🔍 CodeSheriff</h1>
    <p style='font-size: 1.3rem; color: #888;'>
        AI-powered code review, directly inside your GitHub pull requests.
    </p>
    <p style='font-size: 1rem; color: #aaa;'>
        Catches bugs the moment you push — before any human reviews your code.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# METRICS ROW
# ═══════════════════════════════════════════════════════════════════════════

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bug Types Detected", "4")
col2.metric("Avg Review Time", "~20s")
col3.metric("Setup Time", "2 min")
col4.metric("Cost", "$0")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════

st.header("How It Works")

steps_col1, steps_col2 = st.columns(2)

with steps_col1:
    st.markdown(
        """
    **1. Install the GitHub App**
    One-click install on any repository you own.

    **2. Open a Pull Request**
    Work exactly as you normally would. Nothing changes.

    **3. CodeSheriff Reviews It**
    Within seconds, a structured review appears as a PR comment.

    **4. Fix Before Humans See It**
    Address issues before your team, professor, or client reviews your code.
    """
    )

with steps_col2:
    st.markdown(
        """
    **What CodeSheriff catches:**

    🔴 **Security Vulnerabilities**
    SQL injection, unsafe eval(), shell injection

    🟠 **Null Reference Risks**
    Accessing attributes on values that could be None

    🟡 **Type Mismatches**
    String + int errors, wrong comparison types

    🟢 **Logic Flaws**
    Off-by-one errors, incorrect conditions, division by zero
    """
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# WHO IS IT FOR
# ═══════════════════════════════════════════════════════════════════════════

st.header("Who Is It For")

persona_col1, persona_col2, persona_col3 = st.columns(3)

with persona_col1:
    st.markdown(
        """
    **🎓 Students**
    Get instant feedback on every commit without waiting for a
    professor or peer reviewer. Learn from specific, actionable
    corrections immediately.
    """
    )

with persona_col2:
    st.markdown(
        """
    **🧑‍💻 Solo Developers**
    No team means no code reviews. CodeSheriff is your
    always-available reviewer that never gets tired.
    """
    )

with persona_col3:
    st.markdown(
        """
    **🚀 Small Teams**
    Let CodeSheriff handle the first-pass review so your senior
    engineers focus only on what matters.
    """
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════════

st.header("Try It Live")
st.caption("Paste any Python code diff below and CodeSheriff will review it instantly.")

diff_input = st.text_area(
    label="Paste your code diff here:",
    value=SAMPLE_DIFF,
    height=250,
    help="Paste a unified diff (the kind GitHub shows in pull requests)",
)

analyze_col, clear_col = st.columns([3, 1])

with analyze_col:
    analyze_clicked = st.button(
        "🔍 Analyze This Diff",
        type="primary",
        use_container_width=True,
    )

with clear_col:
    if st.button("Clear", use_container_width=True):
        diff_input = ""

if analyze_clicked:
    if not diff_input.strip():
        st.warning("Please paste a code diff before analyzing.")
    else:
        with st.spinner("CodeSheriff is reviewing your code..."):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/review",
                    json={"diff": diff_input},
                    timeout=60.0,  # generous timeout for cold starts
                )
                response.raise_for_status()
                result = response.json()

                issues_found = result.get("issues_found", 0)

                if issues_found == 0:
                    st.success("✅ No significant issues detected in this diff.")
                else:
                    st.error(f"🔍 CodeSheriff found {issues_found} issue(s)")

                st.markdown("### Review Output")
                st.markdown(result.get("review", "No review generated."))

            except httpx.TimeoutException:
                st.warning(
                    "⏳ The backend is waking up from sleep (free tier cold start). "
                    "Please wait 30 seconds and try again."
                )
            except httpx.HTTPStatusError as e:
                st.error(f"Backend returned an error: {e.response.status_code}")
            except Exception as e:
                st.error(f"Could not reach the CodeSheriff backend: {e}")
                st.caption(f"Backend URL: {BACKEND_URL}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# INSTALL CTA
# ═══════════════════════════════════════════════════════════════════════════

st.header("Install CodeSheriff on Your Repos")

install_col1, install_col2 = st.columns(2)

with install_col1:
    st.markdown(
        """
    **Setup takes under 2 minutes:**
    1. Click Install below
    2. Select which repositories to enable
    3. Open any pull request
    4. CodeSheriff appears automatically
    """
    )

with install_col2:
    st.link_button(
        "Install on GitHub →",
        "https://github.com/apps/codesheriff",
        use_container_width=True,
        type="primary",
    )
    st.link_button(
        "View Source Code →",
        "https://github.com/your-username/CodeSheriff",
        use_container_width=True,
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;'>
    Built with PyTorch · HuggingFace Transformers · LangGraph · Groq · FastAPI<br>
    Model: fine-tuned CodeBERT · Deployed on HuggingFace Spaces + Render<br><br>
    <em>CodeSheriff is a portfolio project. Free to use, open source.</em>
</div>
""",
    unsafe_allow_html=True,
)

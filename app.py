import streamlit as st
import json
import pickle
import os
from dotenv import load_dotenv
from BM_25 import (
    load_students,
    load_jsonl_file,
    preprocess_jobs,
    build_bm25_model,
    match_students_to_jobs,
)
from chatbot_together import analyze_matches
from supabase import create_client, Client
from datetime import datetime

# --- Supabase setup ---
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ────────────── Setup ────────────── #
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
BASE_DIR = os.path.dirname(__file__)

# ────────────── Streamlit App ────────────── #
st.set_page_config(page_title="Profeshare Job Matcher", layout="wide")
st.title("🔍 Profeshare Job Matcher")

# --- Intern Name Dropdown (store immediately) ---
intern_names = [
    "Pranav Singh", "Devi Lakkoji", "Hemant", "Harsh", "Preethi",
    "Rohithauppala", "Samarpan Karra", "Sandhya Medapati",
    "Sibtain", "SMA", "Yashpal", "Yeluri Nakshatra"
]
selected_intern = st.selectbox("🧑‍💼 Select Your Name :", intern_names)
st.session_state["intern_name"] = selected_intern  # ← store early

# --- Inputs ---
st.markdown("### 🧑‍🎓 Student Profile Input")

json_input = st.text_area(
    "📄 Paste student profile JSON here:",
    placeholder='[{"first_name": "John", "last_name": "Doe", "skills": ["python", "react"], "job_preferences": {"job_roles": ["backend"]}}]',
    height=250
)

interest_input = st.text_input(
    "💡 Enter interests (separated by '+')",
    placeholder="e.g. frontend+developer+remote"
)

# ────────────── Process Input & Generate Matches ────────────── #
if json_input and interest_input and "matches" not in st.session_state:
    # Parse JSON
    try:
        student_data = json.loads(json_input)
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON: {e}")
        st.stop()

    if not isinstance(student_data, list):
        student_data = [student_data]

    # Update interests
    interest_list = [tok.strip() for tok in interest_input.split("+") if tok.strip()]
    for student in student_data:
        student.setdefault("job_preferences", {})["interests"] = interest_list

    # Save to local file
    students_path = os.path.join(BASE_DIR, "students.json")
    with open(students_path, "w") as f:
        json.dump(student_data, f, indent=2)

    st.success("✅ Student data loaded and interests set!")

    # Load & index job data
    part_files = ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]
    jobs = []
    with st.spinner("🚀 Loading and Indexing Jobs..."):
        for fname in part_files:
            path = os.path.join(BASE_DIR, fname)
            if not os.path.exists(path):
                st.error(f"❌ Missing file in repo: {fname}")
                st.stop()
            with open(path, "r") as f:
                raw_lines = f.read().splitlines()
            try:
                records = [json.loads(line) for line in raw_lines if line.strip()]
            except Exception as e:
                st.error(f"❌ Error loading {fname}: {e}")
                st.stop()
            jobs.extend(records)

        job_texts, job_index = preprocess_jobs(jobs)
        bm25 = build_bm25_model(job_texts)

    st.success("✅ Loaded and indexed jobs")

    # Match students to jobs
    matches = match_students_to_jobs(
        student_data, jobs, bm25, job_index, top_n=10
    )
    pickle_path = os.path.join(BASE_DIR, "student_job_matches.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(matches, f)

    # LLM reasoning on matches
    try:
        with st.spinner("🧠 LLM is analyzing job matches..."):
            final_resp = analyze_matches(pickle_path, student_data)
        st.markdown("## 🤖 LLM Career Analysis")
        st.write(final_resp)
    except Exception as e:
        st.error(f"❌ LLM reasoning failed: {e}")
        final_resp = {}

    # ─── Save to session_state ─── #
    st.session_state["student_json"] = json_input
    st.session_state["matches"] = matches
    st.session_state["final_resp"] = final_resp

# ────────────── Show & Push to Supabase ────────────── #
if "matches" in st.session_state and "final_resp" in st.session_state:
    st.markdown("### ✅ Data is ready to be uploaded")
    st.write(st.session_state["final_resp"])

    if st.button("📤 Push to Supabase"):
        with st.spinner("🚀 Uploading data to Supabase..."):
            try:
                data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "intern_name": st.session_state["intern_name"],
                    "student_profile": st.session_state["student_json"],
                    "bm25_matches": st.session_state["matches"],
                    "llm_analysis": st.session_state["final_resp"],
                }
                response = supabase.table("job_matches").insert(data).execute()
                if response.data:
                    st.success("✅ Data inserted successfully into Supabase!")
                    # Clear state for next run
                    for key in ["matches", "final_resp", "student_json", "intern_name"]:
                        st.session_state.pop(key, None)
                else:
                    st.error(f"❌ Insert failed: {response}")
            except Exception as e:
                st.error(f"❌ Supabase error: {e}")

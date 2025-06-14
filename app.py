import streamlit as st
import json
import pickle
import os
from dotenv import load_dotenv
from BM_25 import load_students, load_jsonl_file, preprocess_jobs, build_bm25_model, match_students_to_jobs
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

if json_input and interest_input:
    try:
        student_data = json.loads(json_input)
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON: {e}")
        st.stop()

    if not isinstance(student_data, list):
        student_data = [student_data]

    # Update job_preferences.interests
    interest_list = [tok.strip() for tok in interest_input.split("+") if tok.strip()]
    for student in student_data:
        student.setdefault("job_preferences", {})["interests"] = interest_list

    # Save to file for further processing
    students_path = os.path.join(BASE_DIR, "students.json")
    with open(students_path, "w") as f:
        json.dump(student_data, f, indent=2)

    st.success("✅ Student data loaded and interests set!")

    # ───── Load & Preprocess Job Data ───── #
    part_files = ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]
    jobs = []
    for fname in part_files:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            st.error(f"❌ Missing file in repo: {fname}")
            st.stop()

        try:
            raw_lines = open(path, "r").read().splitlines()
            # skip blank lines
            records = [json.loads(line) for line in raw_lines if line.strip()]
        except Exception as e:
            st.error(f"❌ Error loading {fname}: {e}")
            st.stop()

        jobs.extend(records)

    job_texts, job_index = preprocess_jobs(jobs)
    bm25 = build_bm25_model(job_texts)
    st.success(f"✅ Loaded and indexed {len(jobs)} jobs")

    # ───── Match Students to Jobs ───── #
    matches = match_students_to_jobs(
        student_data, jobs, bm25, job_index, top_n=10
    )
    pickle_path = os.path.join(BASE_DIR, "student_job_matches.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(matches, f)
    st.success("🎯 Top job matches generated using BM25!")
    st.write(matches)

    # ───── LLM Reasoning on Matches ───── #
    try:
        final_resp = analyze_matches(pickle_path, student_data)
        st.markdown("## 🤖 LLM Career Analysis")
        st.write(final_resp)
    except Exception as e:
        st.error(f"❌ LLM reasoning failed: {e}")
    # --- Prepare data ---
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "student_profile": json_input,
        "bm25_matches": matches,
        "llm_analysis": final_resp
    }
# --- Insert into Supabase ---
    response = supabase.table("job_matches").insert(data).execute()

    if response.data:
        st.success("✅ Data inserted successfully!")
    else:
        st.error(f"❌ Insert failed: {response}")

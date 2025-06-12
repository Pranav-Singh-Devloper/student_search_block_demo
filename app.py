import streamlit as st
import json
import pickle
import os
from dotenv import load_dotenv
from BM_25 import load_students, load_jsonl_file, preprocess_jobs, build_bm25_model, match_students_to_jobs
from chatbot_together import analyze_matches

# ────────────── Setup ────────────── #
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
BASE_DIR = os.path.dirname(__file__)

# ────────────── Streamlit App ────────────── #
st.set_page_config(page_title="Profeshare Job Matcher", layout="wide")
st.title("🔍 Profeshare Job Matcher")

# --- Inputs ---
uploaded_file = st.file_uploader("📁 Upload student profile JSON file", type=["json"])
interest_input = st.text_input(
    "💡 Enter interests (separated by '+')", 
    placeholder="e.g. frontend+developer+>10LPA+hybrid"
)

if uploaded_file and interest_input:
    # ───── Parse & Validate Uploaded JSON ───── #
    raw = uploaded_file.read()
    if not raw:
        st.error("❌ Uploaded file was empty.")
        st.stop()

    try:
        student_data = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"❌ Could not parse uploaded JSON: {e}")
        st.stop()

    if not isinstance(student_data, list):
        student_data = [student_data]

    # ───── Update Interests in the Payload ───── #
    interest_list = [tok.strip() for tok in interest_input.split("+") if tok.strip()]
    for student in student_data:
        student.setdefault("job_preferences", {})["interests"] = interest_list

    # ───── Save Temp Student File ───── #
    students_path = os.path.join(BASE_DIR, "students.json")
    with open(students_path, "w") as f:
        json.dump(student_data, f, indent=2)
    st.success("✅ Interests updated and student profile processed!")

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

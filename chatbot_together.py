import os
import pickle
import json
from openai import OpenAI
import streamlit as st

def analyze_matches(pickle_file_path: str, student_data: list) -> str:
    """
    Analyze job matches for a given student using Together AI and DeepSeek model.

    Args:
        pickle_file_path (str): Path to .pkl file containing job matches.
        student_data (list): A list with one student dictionary.

    Returns:
        str: A human-readable, structured match analysis.
    """

    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    if not TOGETHER_API_KEY:
        return "❌ TOGETHER_API_KEY is missing from Streamlit secrets."

    # Initialize Together-compatible OpenAI client
    client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=TOGETHER_API_KEY
    )

    # Load job matches from pickle
    try:
        with open(pickle_file_path, 'rb') as f:
            matches = pickle.load(f)
    except Exception as e:
        return f"❌ Failed to load job matches: {e}"

    # Extract student info
    student = student_data[0]
    student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()

    # Retrieve matched jobs
    top_jobs = matches.get(student_name, [])
    if not top_jobs:
        return f"❌ No jobs matched for student {student_name}."

    # Prepare prompts
    system_prompt = """
You are an expert career advisor and the world’s most accurate job matcher, capable of analyzing JSON data with precision and delivering deep, insightful evaluations.

Below is a JSON array of job postings (companies). When I give you a student profile (also in JSON), you must:

🔍 1. Analyze each job posting minutely for:
   - Required and preferred skills
   - Work type (internship/full-time)
   - Start date, job title, location preferences
   - Domain fit (e.g., software, AI, management, etc.)
   - Any additional qualification criteria
   - Explain your Analysis in detail 

🎯 2. Assign a “Match Score” (0–100%) to each job based on how well the student's profile aligns.

📊 3. Sort all jobs in ascending order of Match Score (least to most relevant).

💡 4. For each job, provide a detailed but reader-friendly breakdown:

🧾 **Job X (Start Date: YYYY-MM-DD)** – “{{Job Title}} at {{Company Name}}”
- 🔢 Match Score: XX%
- ✅ Why it's a good fit (EXPLAIN IN DETAIL):
  • ...
  • ...
- ⚠️ Potential difficulties / mismatches:
  • ...
  • ...

🎨 5. Format the output beautifully using bullet points, emojis, and bold headers. Only return a clean, structured, and human-readable evaluation. Sort it in most relevant to least relevant.
""" + "\n\nHere are the company profiles:\n" + json.dumps(top_jobs, indent=2)

    user_prompt = "🎓 Student Profile JSON:\n" + json.dumps(student, indent=2)

    # Send to LLM
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ LLM processing failed: {e}"

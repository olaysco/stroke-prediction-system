import json
import os
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Securely load OpenAI API key
openai_key = st.secrets.get("OPENAI_API_KEY", "")
if not openai_key or openai_key == "your_openai_api_key_here":
    st.error("OpenAI API key missing. Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` before using the assistant.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key

# Model download configuration
MODEL_URL = (
    "https://github.com/olaysco/stroke-prediction-system/releases/download/"
    "v1.0.0-beta/rf_stroke_combined_model.joblib"
)
MODEL_PATH = Path("rf_stroke_combined_model.joblib")


def _ensure_model_file() -> Path:
    try:
        with requests.get(MODEL_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with MODEL_PATH.open("wb") as model_file:
                for chunk in response.iter_content(8192):
                    if chunk:
                        model_file.write(chunk)
    except requests.RequestException as exc:
        st.error(
            "Unable to download the stroke model. Please check your internet "
            f"connection or download it manually from {MODEL_URL}.\nError: {exc}"
        )
        st.stop()

    return MODEL_PATH


# Load XGBoost stroke model
try:
    with _ensure_model_file().open("rb") as f:
        stroke_model = joblib.load(f)
except FileNotFoundError:
    st.error("Model file 'rf_stroke_combined_model.joblib' not found. Please check the path.")
    st.stop()

NUMERIC_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "diabetes",
]
CAT_COLUMNS = ["gender", "smoking_status", "ever_married"]
ALL_COLUMNS = NUMERIC_COLUMNS + CAT_COLUMNS
ALIASES = {
    "glucose_level": "avg_glucose_level",
    "evermarried": "ever_married",
}


def _features_to_dataframe(raw_features: dict) -> pd.DataFrame:
    normalized = {}
    for key, value in raw_features.items():
        normalized[ALIASES.get(key, key)] = value

    row = {column: normalized.get(column) for column in ALL_COLUMNS}
    missing = [column for column, value in row.items() if value is None]
    if missing:
        raise ValueError(
            "Missing features: " + ", ".join(missing) + ". Provide these values before prediction."
        )
    return pd.DataFrame([row], columns=ALL_COLUMNS)


@tool
def stroke_prediction_tool(features_str: str) -> str:
    """Predicts stroke risk from JSON features (age, hypertension, glucose_level, heart_disease, bmi). Returns a JSON payload with probability for downstream messaging."""
    try:
        features = json.loads(features_str)
        feature_frame = _features_to_dataframe(features)
        probability = stroke_model.predict_proba(feature_frame)[0][1]
        return json.dumps({"probability": float(probability)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7)

system_prompt = """
You are the Stroke Risk Prediction Assistant for this Streamlit app. Your role is to guide non-technical users through a brief, conversational stroke risk assessment.

Your scope: Focus only on stroke-related information. If a user asks something unrelated, politely remind them that your role is limited to stroke risk discussions.

---

**Workflow**

1. Begin naturally — greet the user and briefly explain that you’ll ask a few questions to estimate their stroke risk.  
   Example: “Hi there! I can help estimate your stroke risk based on some quick health details.”

2. Collect the required clinical features one by one (or together if the user prefers).  
   Always confirm unclear values and maintain a reassuring, conversational tone.

   Required keys for the JSON payload:
   - age (integer)
   - gender ("male" / "female")
   - ever_married ("yes" / "no")
   - hypertension (1 = yes, 0 = no)
   - heart_disease (1 = yes, 0 = no)
   - diabetes (1 = yes, 0 = no)
   - avg_glucose_level (numeric)
   - bmi (numeric)
   - smoking_status ("smokes", "formerly smoked", "never smoked", or "Unknown")

3. Only call the Stroke_Prediction tool after **all values are collected**.  
   Example payload: {"age": 67, "gender": "male", "ever_married": "yes", "hypertension": 1, "heart_disease": 0, "diabetes": 1, "avg_glucose_level": 155, "bmi": 32.1, "smoking_status": "formerly smoked"}

4. After receiving the model’s output, summarize it in plain language:
   - **Risk Level:** “Based on your data, your estimated stroke risk is High/Low.”
   - **Key Factors:** If risk is high, mention 1–2 contributing factors (e.g., high glucose, hypertension).
   - **Care Suggestions:** Offer simple next steps, like seeing a doctor, monitoring blood pressure, or maintaining a balanced diet.
   - **Disclaimer:** Always end with “This is not medical advice; please consult a doctor.”

5. If a model error occurs or data is missing, respond kindly:  
   - Apologize briefly.  
   - Explain what’s needed.  
   - Guide the user to correct the missing or invalid detail.

---

**Tone & Style**
- Empathetic, clear, and human.  
- Use short, friendly sentences.  
- Never sound like an instruction manual.  
- Stay factual and clinically aware without sounding robotic.

Example opening:
“Hello! I can help you check your stroke risk. Could you share your age and whether you have high blood pressure or diabetes?”
"""

try:
    agent = create_agent(
        model=llm,
        tools=[stroke_prediction_tool],
        system_prompt=system_prompt,
    )
except Exception as exc:
    st.error(f"Error initializing agent: {exc}")
    st.stop()


st.title("🫀Stroke Risk Prediction Assistant")
st.caption("Built by Olayiwola Odunsi")
st.write("Enter your symptoms or history (e.g., 'I'm 67, male, hypertensive, glucose 155, BMI 32, diabetic').")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []

for message in st.session_state.messages:
    avatar = "🩺" if message["role"] == "assistant" else "🙋"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if user_input := st.chat_input("Your symptoms:"):
    user_msg = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_msg)
    st.session_state.lc_messages.append(HumanMessage(content=user_input))

    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    with st.spinner("Processing..."):
        try:
            result = agent.invoke({"messages": st.session_state.lc_messages})
            lc_messages = result.get("messages", [])
            if not lc_messages:
                raise ValueError("Agent response did not include messages.")
            st.session_state.lc_messages = lc_messages
            assistant_message = lc_messages[-1]
            if isinstance(assistant_message, AIMessage):
                response_text = assistant_message.content
            else:
                response_text = getattr(assistant_message, "content", "")
        except Exception as exc:
            response_text = f"Error processing input: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant", avatar="🩺"):
        if "High risk" in response_text:
            st.markdown(f"<p style='color:red'>{response_text}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:green'>{response_text}</p>", unsafe_allow_html=True)

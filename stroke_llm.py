import json
import os

import joblib
import pandas as pd
import streamlit as st
import altair as alt
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

# Load XGBoost stroke model
try:
    with open("rf_stroke_combined_model.joblib", "rb") as f:
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
REQUIRED_COLUMNS = set(ALL_COLUMNS)
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
        print("Features received for prediction:", features_str)  # Debug statement
        feature_frame = _features_to_dataframe(features)
        probability = stroke_model.predict_proba(feature_frame)[0][1]
        return json.dumps({"probability": float(probability)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

system_prompt = """
You are the Stroke Risk Prediction Assistant for this Streamlit app. You must focus exclusively on stroke-related assessments—ignore requests that are unrelated to stroke risk and reply with a brief reminder of your scope.

Workflow:
1. Collect clinical features as a JSON payload using these keys: age, hypertension, avg_glucose_level, heart_disease, bmi, gender, smoking_status, ever_married, diabetes. Example payload: {"age": 67, "hypertension": 1, "heart_disease": 0, "avg_glucose_level": 155, "bmi": 32.1, "gender": "male", "smoking_status": "formerly smoked", "ever_married": "yes", "diabetes": 1}.
   - Use canonical categorical values: gender → "male"/"female"; smoking_status → "smokes", "formerly smoked", "never smoked", or "Unknown"; ever_married → "yes"/"no"; diabetes → 1 or 0.
2. Do not call the Stroke_Prediction tool until every key above has a value. If any are missing, ask focused follow-up questions.
3. When the tool returns, craft a response in the following parts as neccessary:
   - Risk Level: High if probability > 0.5, Low otherwise.
   - Key Factors: Include Key Factors if Risk Level is High. Highlight 1-2 factors from the input that influence the risk.
   - Care Suggestions: Offer practical next steps (monitor metrics, consult clinicians, lifestyle guidance) (tone and suggesstion should be based on risk level).
   - Disclaimer: Always end with "This is not medical advice; consult a doctor."
4. If the tool returns an error, apologize, explain the issue, and request corrected or missing information.

Tone: Empathetic, concise, and clinically aware. Keep answers relevant to stroke prevention and management.
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

# ------------------ IMPORTS ------------------
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage
from utils.web_search import perform_web_search
from models.llm import get_groq_model
from utils.rag_utils import load_and_split_pdf, embed_chunks, retrieve_relevant_chunks


# ------------------ UTILS ------------------
def detect_thyroid_type(tsh, t3, t4):
    if tsh > 4.0 and (t3 < 2.3 or t4 < 0.8):
        return "Hypothyroidism"
    elif tsh < 0.4 and (t3 > 4.2 or t4 > 1.8):
        return "Hyperthyroidism"
    elif 0.4 <= tsh <= 4.0 and 2.3 <= t3 <= 4.2 and 0.8 <= t4 <= 1.8:
        return "Normal"
    else:
        return "Borderline / Consult Physician"


# ------------------ PAGE FUNCTIONS ------------------

def patient_profile_page():
    st.title("👤 Patient Profile")
    st.markdown("Enter the patient’s basic and thyroid health information.")

    with st.form("profile_form"):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, step=1)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])

        st.markdown("### Thyroid Lab Values")
        tsh = st.number_input("TSH (mIU/L)", step=0.1, format="%.2f")
        t3 = st.number_input("Free T3 (pg/mL)", step=0.1, format="%.2f")
        t4 = st.number_input("Free T4 (ng/dL)", step=0.1, format="%.2f")

        st.markdown("### Other Information (Optional)")
        weight = st.number_input("Weight (kg)", step=0.1)
        height = st.number_input("Height (cm)", step=0.1)
        symptoms = st.text_area("Symptoms", placeholder="Fatigue, hair loss, weight gain...")
        medication = st.text_area("Ongoing Medications")

        submitted = st.form_submit_button("Save Profile")

        if submitted:
            thyroid_type = detect_thyroid_type(tsh, t3, t4)
            st.session_state.patient_profile = {
                "name": name,
                "age": age,
                "gender": gender,
                "tsh": tsh,
                "t3": t3,
                "t4": t4,
                "thyroid_type": thyroid_type,
                "weight": weight,
                "height": height,
                "symptoms": symptoms,
                "medication": medication
            }
            st.success(f"✅ Profile Saved! Thyroid Type: **{thyroid_type}**")


def chat_page():
    chat_model = get_groq_model()

    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Concise"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = {}

    st.title("ThyBot")
    st.markdown("Ask anything about thyroid health.")

    with st.sidebar:
        st.session_state.response_mode = st.radio(
            "Response Style",
            ["Concise", "Detailed"],
            index=0 if st.session_state.response_mode == "Concise" else 1
        )

    uploaded_file = st.file_uploader("Upload a PDF (lab report)", type=["pdf"])
    if uploaded_file:
        st.session_state["chunks"] = load_and_split_pdf(uploaded_file)
        st.session_state["faiss_index"] = embed_chunks(st.session_state["chunks"])
        st.success("Document embedded for retrieval!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                profile = st.session_state.patient_profile
                thyroid_type = profile.get("thyroid_type", "Not set")

                system_prompt = (
                    f"You are a helpful medical assistant. The user has {thyroid_type}. Keep the answer brief and to the point."
                    if st.session_state.response_mode == "Concise"
                    else f"You are a helpful medical assistant. The user has {thyroid_type}. Provide a detailed explanation."
                )

                if "faiss_index" in st.session_state:
                    docs = retrieve_relevant_chunks(prompt, st.session_state["faiss_index"])
                    context = "\n\n".join([doc.page_content for doc in docs])
                    full_prompt = f"Context: {context}\n\nUser: {prompt}\n\nAnswer {('briefly' if st.session_state.response_mode == 'Concise' else 'in detail')}"
                    response = chat_model.invoke(full_prompt)
                    reply = response.content if hasattr(response, "content") else response
                else:
                    response = chat_model.invoke([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ])
                    reply = response.content if hasattr(response, "content") else response

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})


def meal_analysis_page():
    st.title("Meal Analysis")
    st.markdown("Add food items to analyze whether they are thyroid-friendly.")

    df = pd.read_csv("data/Indian_Food_Nutrition_Processed.csv")
    chat_model = get_groq_model()

    if "meal_items" not in st.session_state:
        st.session_state.meal_items = []

    col1, col2 = st.columns([3, 1])
    with col1:
        new_item = st.text_input("Enter food item")
    with col2:
        if st.button("Add") and new_item:
            st.session_state.meal_items.append(new_item)

    for idx, item in enumerate(st.session_state.meal_items):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"- {item}")
        with col2:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.meal_items.pop(idx)
                st.rerun()

    if st.button("Analyze Meal"):
        profile = st.session_state.get("patient_profile", {})
        thyroid_type = profile.get("thyroid_type", "Not set")

        if thyroid_type == "Not set":
            st.error("Please fill in your thyroid profile first from the Patient Profile page.")
            return

        for item in st.session_state.meal_items:
            match = df[df['Dish Name'].str.lower() == item.lower()]
            if not match.empty:
                row = match.iloc[0]
                nutrients = f"Calories: {row['Calories (kcal)']:.0f} kcal | Protein: {row['Protein (g)']}g | Sugar: {row['Free Sugar (g)']}g"
                impact = row['Thyroid_Impact']
                prompt = (
                    f"The user has {thyroid_type}. They are eating '{item}', which has the following nutritional values: {nutrients}. "
                    f"The dataset marks its thyroid impact as '{impact}'. Is this good or bad for the user and why? Also suggest what else could be added or avoided."
                )
            else:
                prompt = (
                    f"The user has {thyroid_type}. They are eating '{item}', but it is not found in the dataset. "
                    f"Based on general nutritional knowledge, is this good or bad for thyroid health? Give a brief reason and suggest improvements."
                )

            response = chat_model.invoke(prompt)
            reply = response.content if hasattr(response, "content") else response

            with st.container():
                st.markdown(f"#### 🍽️ **{item.title()}**")
                if not match.empty:
                    st.info(f"**Nutrients**: {nutrients}\n\n**Impact**: {impact}")
                st.success(reply)


# ------------------ MAIN ------------------

def main():
    st.set_page_config(page_title="ThyBot", page_icon="assets/logo.png", layout="centered")

    with st.sidebar:
        st.image("assets/logo.png",width=180)
        st.markdown("Select a feature from below:")
        page = st.radio("Navigation", ["Chat", "Patient Profile", "Meal Analysis"])

    if page == "Chat":
        chat_page()
    elif page == "Patient Profile":
        patient_profile_page()
    elif page == "Meal Analysis":
        meal_analysis_page()


# ------------------ LAUNCH ------------------

if __name__ == "__main__":
    main()

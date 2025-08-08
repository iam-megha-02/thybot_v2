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
    st.title("Patient Profile")

    # --- STATE MANAGEMENT ---
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = {} 
    
    if "editing_profile" not in st.session_state:
        st.session_state.editing_profile = not st.session_state.patient_profile.get("name")


    # ------------------ 1. THE EDITING VIEW (FORM) ------------------
    if st.session_state.editing_profile:
        st.markdown("Enter or update the patient’s health information below.")
        
        profile_data = st.session_state.patient_profile
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name", value=profile_data.get("name", ""))
                age = st.number_input("Age", min_value=0, max_value=120, step=1, value=profile_data.get("age", 0))
                gender_options = ["Female", "Male", "Other"]
                gender_index = gender_options.index(profile_data.get("gender", "Female")) if profile_data.get("gender") in gender_options else 0
                gender = st.selectbox("Gender", gender_options, index=gender_index)

            with col2:
                weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1, format="%.1f", value=profile_data.get("weight", 0.0))
                height = st.number_input("Height (cm)", min_value=0.0, step=0.1, format="%.1f", value=profile_data.get("height", 0.0))
                bmi = (weight / ((height / 100) ** 2)) if height > 0 else 0

            st.markdown("---")
            st.markdown("<h5>Thyroid Lab Values</h5>", unsafe_allow_html=True)
            tsh = st.number_input("TSH (mIU/L)", step=0.1, format="%.2f", value=profile_data.get("tsh", 0.0))
            t3 = st.number_input("Free T3 (pg/mL)", step=0.1, format="%.2f", value=profile_data.get("t3", 0.0))
            t4 = st.number_input("Free T4 (ng/dL)", step=0.1, format="%.2f", value=profile_data.get("t4", 0.0))

            submitted = st.form_submit_button("Save Profile")

            if submitted:
                thyroid_type = detect_thyroid_type(tsh, t3, t4)
                st.session_state.patient_profile = {
                    "name": name, "age": age, "gender": gender,
                    "tsh": tsh, "t3": t3, "t4": t4,
                    "thyroid_type": thyroid_type,
                    "weight": weight, "height": height, "bmi": bmi,
                }
                st.session_state.editing_profile = False
                st.success(f"✅ Profile for **{name}** saved!")
                st.rerun()

    # ------------------ 2. THE DISPLAY VIEW (CARD) ------------------
    else:
        st.markdown("Here is your current profile. The AI Assistant will use this information for tailored responses.")
        profile = st.session_state.patient_profile
        
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"#### {profile.get('name', 'No Name')}")
                st.markdown(f"**Age:** {profile.get('age', 'N/A')} | **Gender:** {profile.get('gender', 'N/A')} | **BMI:** {profile.get('bmi', 0.0):.2f}")
            
            with col2:
                status = profile.get('thyroid_type', 'N/A')
                st.markdown(
                    f"""
                    <div style="text-align: right;">
                        <p style="font-size: 0.9rem; margin-bottom: -5px;">Thyroid Status</p>
                        <p style="font-size: 1.25rem; font-weight: 600;">{status}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            c1, c2, c3 = st.columns(3)
            metric_style = "text-align: center;"
            with c1:
                st.markdown(f"<div style='{metric_style}'><p style='font-size: 0.9rem; margin-bottom: -5px;'>TSH (mIU/L)</p><p style='font-size: 2rem; font-weight: 600;'>{profile.get('tsh', 0.0):.2f}</p></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='{metric_style}'><p style='font-size: 0.9rem; margin-bottom: -5px;'>Free T3 (pg/mL)</p><p style='font-size: 2rem; font-weight: 600;'>{profile.get('t3', 0.0):.2f}</p></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div style='{metric_style}'><p style='font-size: 0.9rem; margin-bottom: -5px;'>Free T4 (ng/dL)</p><p style='font-size: 2rem; font-weight: 600;'>{profile.get('t4', 0.0):.2f}</p></div>", unsafe_allow_html=True)

        if st.button(" Edit Profile"):
            st.session_state.editing_profile = True
            st.rerun()


def chat_page():
    chat_model = get_groq_model()

    # --- Session State Initialization ---
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Concise"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = {}

    st.title("ThyBot")
    st.markdown("Your AI assistant for thyroid health. Ask me anything!")

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.markdown("### Chat Settings")
        st.session_state.response_mode = st.radio(
            "Response Style",
            ["Concise", "Detailed"],
            index=0 if st.session_state.response_mode == "Concise" else 1,
            horizontal=True
        )
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                profile = st.session_state.patient_profile
                thyroid_type = profile.get("thyroid_type", "Not specified")
                
                system_message_content = (
                    f"You are ThyBot, an expert AI medical assistant specializing in thyroid health. "
                    f"The user's thyroid status is '{thyroid_type}'. "
                    f"Your response style should be {st.session_state.response_mode}."
                )

                context = ""
               
                if "faiss_index" in st.session_state:
                    docs = retrieve_relevant_chunks(prompt, st.session_state["faiss_index"])
                    if docs:
                        context = "\n\n".join([doc.page_content for doc in docs])
                        system_message_content += (
                            "\n\nUse the following document context to answer the user's question. "
                            "If the context doesn't contain the answer, say so and answer based on your general knowledge.\n\n"
                            f"CONTEXT:\n---\n{context}"
                        )
                
                messages_for_llm = [
                    {"role": "system", "content": system_message_content}
                ] + st.session_state.messages

                response = chat_model.invoke(messages_for_llm)
                reply = response.content if hasattr(response, "content") else str(response)

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})


def meal_analysis_page():
    st.title("Meal Analysis")
    st.markdown("Select food items from the list to analyze their impact on thyroid health.")

    chat_model = get_groq_model()
    
    @st.cache_data
    def load_food_data():
        df = pd.read_csv("data/Indian_Food_Nutrition_Processed.csv")
        return df

    df = load_food_data()
    food_list = [""] + sorted(df['Dish Name'].tolist())

    if "meal_items" not in st.session_state:
        st.session_state.meal_items = []
        
    profile = st.session_state.get("patient_profile", {})
    thyroid_type = profile.get("thyroid_type", "Not set")

    if thyroid_type == "Not set":
        st.error("Please create a patient profile first on the 'Patient Profile' page to get tailored meal analysis.")
        st.stop()
    else:
        st.info(f"Analyzing meals for a patient with: **{thyroid_type}**")

    new_item = st.selectbox("Select a food item to add to your meal:", options=food_list)
    if st.button("Add Item") and new_item and new_item not in st.session_state.meal_items:
        st.session_state.meal_items.append(new_item)
        st.rerun()

    st.markdown("---")
    st.markdown("#### Your Current Meal")

    if not st.session_state.meal_items:
        st.write("No items added yet.")
    else:
        for i, item in enumerate(st.session_state.meal_items):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f"- **{item}**")
            with col2:
                if st.button("✖️", key=f"remove_{i}", help="Remove item"):
                    st.session_state.meal_items.pop(i)
                    st.rerun()

    if st.session_state.meal_items and st.button("Analyze Meal"):
        with st.spinner("Analyzing your meal..."):
            for item in st.session_state.meal_items:
                match = df[df['Dish Name'] == item]
                row = match.iloc[0]
                nutrients = f"Calories: {row['Calories (kcal)']:.0f} kcal | Protein: {row['Protein (g)']}g | Sugar: {row['Free Sugar (g)']}g"
                impact = row['Thyroid_Impact']
                prompt = (
                    f"A patient with '{thyroid_type}' is eating '{item}'. Its known thyroid impact is '{impact}' "
                    f"and its nutrients are: {nutrients}. Briefly explain if this food is generally beneficial, neutral, "
                    f"or should be consumed with caution for their condition and why. "
                    f"Provide one simple suggestion for a healthy pairing or alternative."
                )
                
                response = chat_model.invoke(prompt)
                reply = response.content if hasattr(response, "content") else str(response)

                with st.expander(f"Analysis for: **{item}**", expanded=True):
                    st.info(f"**Thyroid Impact:** {impact} | **Nutrients:** {nutrients}")
                    st.markdown(reply)
                    
def document_analysis_page():
    st.title("📄 Document Analysis (RAG)")
    st.markdown("Upload a PDF document (like a lab report) to provide extra context for the chatbot.")
    st.info("After uploading, you can go to the **Chat** page to ask questions about the document's contents.", icon="💡")

    uploaded_file = st.file_uploader("Upload a PDF report", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing document... This may take a moment."):
            st.session_state["chunks"] = load_and_split_pdf(uploaded_file)
            st.session_state["faiss_index"] = embed_chunks(st.session_state["chunks"])
        st.success("✅ Document processed successfully! You can now ask about it in the Chat page.")

    if "faiss_index" in st.session_state:
        st.markdown("---")
        st.markdown("A document is currently loaded in memory. You can ask questions about it on the Chat page.")
        if st.button("Remove Document"):
            del st.session_state["chunks"]
            del st.session_state["faiss_index"]
            st.rerun()


# ------------------ MAIN ------------------
def main():
    st.set_page_config(page_title="ThyBot", page_icon="assets/logo.png", layout="centered")

    with st.sidebar:
        st.image("assets/logo.png", width=150) # Reduced logo size slightly to save space
        st.markdown("Select a feature from below:")
        # --- CHANGED: Added Document Analysis to navigation ---
        page = st.radio("Navigation", ["Chat", "Patient Profile", "Meal Analysis", "Document Analysis"])

    # --- CHANGED: Added routing for the new page ---
    if page == "Chat":
        chat_page()
    elif page == "Patient Profile":
        patient_profile_page()
    elif page == "Meal Analysis":
        meal_analysis_page()
    elif page == "Document Analysis":
        document_analysis_page()


# ------------------ LAUNCH ------------------
if __name__ == "__main__":
    main()

# ------------------ IMPORTS ------------------
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage
# REMOVED: Unused imports
# from utils.web_search import perform_web_search
from models.llm import get_groq_model
from utils.rag_utils import embed_chunks, retrieve_relevant_chunks
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

def load_and_split_document(uploaded_file):
    """Loads and splits a document based on its file type."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(tmp_file_path)
    elif file_extension == ".txt":
        loader = TextLoader(tmp_file_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        os.remove(tmp_file_path)
        return None

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    os.remove(tmp_file_path)
    return chunks


# ------------------ PAGE FUNCTIONS ------------------

def patient_profile_page():
    st.title("Patient Profile")
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = {}
    if "editing_profile" not in st.session_state:
        st.session_state.editing_profile = not st.session_state.patient_profile.get("name")
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
    else:
        st.markdown("Here is the current patient profile. This information provides context for other features.")
        profile = st.session_state.patient_profile
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"#### {profile.get('name', 'No Name')}")
                st.markdown(f"**Age:** {profile.get('age', 'N/A')} | **Gender:** {profile.get('gender', 'N/A')} | **BMI:** {profile.get('bmi', 0.0):.2f}")
            with col2:
                status = profile.get('thyroid_type', 'N/A')
                st.markdown(f"""<div style="text-align: right;"><p style="font-size: 0.9rem; margin-bottom: -5px;">Thyroid Status</p><p style="font-size: 1.25rem; font-weight: 600;">{status}</p></div>""", unsafe_allow_html=True)
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            metric_style = "text-align: center;"
            with c1:
                st.markdown(f"<div style='{metric_style}'><p style='font-size: 0.9rem; margin-bottom: -5px;'>TSH (mIU/L)</p><p style='font-size: 2rem; font-weight: 600;'>{profile.get('tsh', 0.0):.2f}</p></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='{metric_style}'><p style='font-size: 0.9rem; margin-bottom: -5px;'>Free T3 (pg/mL)</p><p style='font-size: 2rem; font-weight: 600;'>{profile.get('t3', 0.0):.2f}</p></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div style='{metric_style}'><p style='font-size: 0.9rem; margin-bottom: -5px;'>Free T4 (ng/dL)</p><p style='font-size: 2rem; font-weight: 600;'>{profile.get('t4', 0.0):.2f}</p></div>", unsafe_allow_html=True)
        if st.button("Edit Profile"):
            st.session_state.editing_profile = True
            st.rerun()

# --- NEW: General Chat Page ---
def general_chat_page():
    st.title("General Chat")
    st.markdown("Ask questions about thyroid health. This chat uses your patient profile for context.")
    chat_model = get_groq_model()

    # Initialize session state for general chat
    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []

    with st.sidebar:
        st.markdown("### Chat Settings")
        st.session_state.response_mode = st.radio("Response Style", ["Concise", "Detailed"], index=st.session_state.get("response_mode_index", 0), horizontal=True)
        st.session_state.response_mode_index = ["Concise", "Detailed"].index(st.session_state.response_mode)
        if st.button("Clear Chat History"):
            st.session_state.general_messages = []
            st.rerun()

    # Display chat history
    for message in st.session_state.general_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.general_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                profile = st.session_state.get("patient_profile", {})
                thyroid_type = profile.get("thyroid_type", "Not specified")
                
                system_message_content = (f"You are ThyBot, an expert AI medical assistant specializing in thyroid health. The user's thyroid status is '{thyroid_type}'. Your response style should be {st.session_state.response_mode}.")
                
                messages_for_llm = [{"role": "system", "content": system_message_content}] + st.session_state.general_messages
                response = chat_model.invoke(messages_for_llm)
                reply = response.content if hasattr(response, "content") else str(response)

                st.markdown(reply)
                st.session_state.general_messages.append({"role": "assistant", "content": reply})

def document_chat_page():
    st.title("Document Chat")
    st.markdown("Upload a document (`PDF`, `DOCX`, `TXT`) to ask specific questions about its contents.")
    chat_model = get_groq_model()

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    
    if uploaded_file and "doc_faiss_index" not in st.session_state:
        with st.spinner("Processing document..."):
            chunks = load_and_split_document(uploaded_file)
            if chunks:
                st.session_state.doc_faiss_index = embed_chunks(chunks)
                st.session_state.doc_messages = [{"role": "assistant", "content": f"I've finished reading **{uploaded_file.name}**. What would you like to know?"}]
                st.session_state.uploaded_file_name = uploaded_file.name

    if "doc_faiss_index" in st.session_state:
        st.info(f"Currently chatting with **{st.session_state.uploaded_file_name}**.")
        for message in st.session_state.doc_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.doc_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    profile = st.session_state.get("patient_profile", {})
                    thyroid_type = profile.get("thyroid_type", "Not specified")
                    docs = retrieve_relevant_chunks(prompt, st.session_state["doc_faiss_index"])
                    context = "\n\n".join([doc.page_content for doc in docs])
                    system_message_content = (f"You are ThyBot, an expert AI assistant. Answer questions based ONLY on the provided document context. The user's thyroid status is '{thyroid_type}'.\n\nCONTEXT:\n---\n{context}")
                    messages_for_llm = [{"role": "system", "content": system_message_content}, {"role": "user", "content": prompt}]
                    response = chat_model.invoke(messages_for_llm)
                    reply = response.content if hasattr(response, "content") else str(response)
                    st.markdown(reply)
                    st.session_state.doc_messages.append({"role": "assistant", "content": reply})
        if st.button("End Document Chat Session"):
            del st.session_state["doc_faiss_index"]
            del st.session_state["doc_messages"]
            del st.session_state["uploaded_file_name"]
            st.rerun()
    else:
        st.info("Upload a file to start the document chat.")


def meal_analysis_page():
    st.title("Meal Analysis")
    st.markdown("Select food items from a list to analyze their impact on thyroid health.")
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
        st.error("Please create a patient profile first to get tailored meal analysis.")
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
                prompt = (f"A patient with '{thyroid_type}' is eating '{item}'. Its known thyroid impact is '{impact}' and its nutrients are: {nutrients}. Briefly explain if this food is generally beneficial, neutral, or should be consumed with caution for their condition and why. Provide one simple suggestion for a healthy pairing or alternative.")
                response = chat_model.invoke(prompt)
                reply = response.content if hasattr(response, "content") else str(response)
                with st.expander(f"Analysis for: **{item}**", expanded=True):
                    st.info(f"**Thyroid Impact:** {impact} | **Nutrients:** {nutrients}")
                    st.markdown(reply)

# ------------------ MAIN ------------------
def main():
    st.set_page_config(page_title="ThyBot", page_icon="assets/logo.png", layout="centered")

    with st.sidebar:
        st.image("assets/logo.png", width=150)
        st.markdown("Select a feature from below:")
        
        page = st.radio("Navigation", ["Patient Profile", "General Chat", "Document Chat", "Meal Analysis"])

    # --- Routing for both chat pages ---
    if page == "Patient Profile":
        patient_profile_page()
    elif page == "General Chat":
        general_chat_page()
    elif page == "Document Chat":
        document_chat_page()
    elif page == "Meal Analysis":
        meal_analysis_page()

# ------------------ LAUNCH ------------------
if __name__ == "__main__":
    main()

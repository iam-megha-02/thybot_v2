# ------------------ IMPORTS ------------------
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from models.llm import get_groq_model
from utils.rag_utils import embed_chunks, retrieve_relevant_chunks
from utils.web_search import perform_web_search, get_completion
import tempfile
import os

# Imports for Document Handling
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

def home_page():
    st.title("Welcome to ThyBot!")
    st.markdown("### Your AI-Powered Companion for Navigating Thyroid Health")
    st.markdown("---")
    st.info("To get started, create your profile on the **Patient Profile** page. This will allow ThyBot to provide personalized insights and analysis.")
    st.markdown("""
    **What can ThyBot do?**
    - **General Chat:** Ask any question about thyroid health.
    - **Document Chat:** Upload your lab reports for a detailed analysis and ask specific questions about them.
    - **Meal Analysis:** Get personalized feedback on whether your meals are thyroid-friendly.
    """)

def patient_profile_page():
    st.title("👤 Patient Profile")
    st.markdown("This is the most important step. Enter your information here so ThyBot can provide personalized responses.")
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = {} 
    if "editing_profile" not in st.session_state:
        st.session_state.editing_profile = not st.session_state.patient_profile.get("name")
    if st.session_state.editing_profile:
        st.markdown("Enter or update your health information below.")
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

--------------------------------------

LOW_CONFIDENCE_FLAGS = [
    "i don't know", "i am not sure", "not sure", "cannot find", "no information",
    "i don’t have enough information", "insufficient", "can't answer", "cannot answer"
]

def needs_web_search(reply: str) -> bool:
    if not reply:
        return True
    text = reply.strip().lower()
    if text == "[[need_web]]":
        return True
    if len(text) < 40 and any(w in text for w in ["maybe", "unsure", "unclear", "unknown"]):
        return True
    if any(flag in text for flag in LOW_CONFIDENCE_FLAGS):
        return True
    return False

def summarize_search_for_thyroid(query: str, results: list) -> str:
    if not results:
        return "I couldn’t find reliable sources right now. Try rephrasing or a different angle."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        snippet = r.get("snippet", "No description available.")
        link = r.get("link", "#")
        lines.append(f"{i}. {title} — {snippet} ({link})")
    prompt = (
        "You are ThyBot, a thyroid-focused assistant. Using ONLY the snippets below, write a concise, reliable answer. "
        "Highlight consensus, note contradictions if any, and give 2–3 practical tips. "
        "Do not invent facts beyond these snippets.\n\n" + "\n".join(lines)
    )
    return get_completion(prompt) or "No summary generated."


def general_chat_page():
    st.title("General Chat")
    st.markdown("Ask anything about thyroid health.")

    chat_model = get_groq_model()

    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []

    if st.sidebar.button("Clear Chat History"):
        st.session_state.general_messages = []
        st.rerun()

    # Render previous messages
    for message in st.session_state.general_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.general_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                profile = st.session_state.get("patient_profile", {})
                thyroid_type = profile.get("thyroid_type", "Not specified")
                response_style = st.session_state.get("response_mode", "Detailed")

                # System message tells LLM to signal [[NEED_WEB]] if unsure
                system_message = (
                    "You are ThyBot, an expert AI assistant for thyroid health. "
                    f"The user's thyroid status is '{thyroid_type}'. "
                    f"Your response style should be {response_style}. "
                    "If you are NOT reasonably certain based on your internal knowledge, reply EXACTLY with [[NEED_WEB]] and nothing else."
                )

                messages_for_llm = [{"role": "system", "content": system_message}]
                # Keep recent history short to avoid long context
                for msg in st.session_state.general_messages[-8:]:
                    messages_for_llm.append({"role": msg["role"], "content": msg["content"]})

            
                try:
                    response = chat_model.invoke(messages_for_llm)
                    reply = response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    reply = f"⚠️ Error generating response: {e}"

            
                if needs_web_search(reply):
                    try:
                        results = perform_web_search(prompt, max_results=6)
                        summary = summarize_search_for_thyroid(prompt, results)
                        st.markdown(summary)
                        with st.expander("Sources"):
                            for r in results:
                                st.markdown(f"- [{r.get('title','Source')}]({r.get('link','#')})\n\n> {r.get('snippet','')}")
                        st.session_state.general_messages.append({"role": "assistant", "content": summary})
                    except Exception as e:
                        fallback_msg = f"⚠️ Web search failed: {e}"
                        st.markdown(fallback_msg)
                        st.session_state.general_messages.append({"role": "assistant", "content": fallback_msg})
                else:
                    st.markdown(reply)
                    st.session_state.general_messages.append({"role": "assistant", "content": reply})

def document_chat_page():
    st.title("Document Chat")
    st.markdown("Upload a document (`PDF`, `DOCX`, `TXT`) to ask specific questions about its contents. This chat will focus *only* on the document.")
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
                    response_style = st.session_state.get("response_mode", "Detailed")
                    
                    docs = retrieve_relevant_chunks(prompt, st.session_state["doc_faiss_index"])
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    system_message_content = (f"You are ThyBot, an expert AI assistant. Answer questions based ONLY on the provided document context. "
                                              f"The user's thyroid status is '{thyroid_type}'. Your response style should be {response_style}.\n\n"
                                              f"CONTEXT:\n---\n{context}")
                    
                    messages_for_llm = [{"role": "system", "content": system_message_content}, {"role": "user", "content": prompt}]
                    response = chat_model.invoke(messages_for_llm)
                    reply = response.content if hasattr(response, "content") else str(response)
                    st.markdown(reply)
                    st.session_state.doc_messages.append({"role": "assistant", "content": reply})
        if st.button("End Document Chat Session"):
            if "doc_faiss_index" in st.session_state: del st.session_state["doc_faiss_index"]
            if "doc_messages" in st.session_state: del st.session_state["doc_messages"]
            if "uploaded_file_name" in st.session_state: del st.session_state["uploaded_file_name"]
            st.rerun()
    else:
        st.info("Upload a file to start the document chat.")

def meal_analysis_page():
    st.title("Meal Analysis")
    st.markdown("Select food items from a list to analyze the impact on your thyroid health. This analysis is personalized using your patient profile.")
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
    if st.session_state.meal_items and st.button("🍽️ Analyze Meal"):
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
        st.markdown("### Settings")
        st.session_state.response_mode = st.radio(
            "Response Style", ["Concise", "Detailed"],
            index=st.session_state.get("response_mode_index", 1), # Default to Detailed
            help="Choose how you want the AI to respond in the chat sessions."
        )
        st.session_state.response_mode_index = ["Concise", "Detailed"].index(st.session_state.response_mode)
        
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio(
            "Navigation",
            ["Home", "Patient Profile", "General Chat", "Document Chat", "Meal Analysis"],
            label_visibility="collapsed"
        )

    if page == "Home":
        home_page()
    elif page == "Patient Profile":
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

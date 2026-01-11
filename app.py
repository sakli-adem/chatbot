import streamlit as st
from PyPDF2 import PdfReader
import os
import time
import json
import re
from google import genai

# --- Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 🔑 API Key ---
FIXED_API_KEY = "AIzaSyDjZ0igflBSazzUC3bEHf3mjLQ1L_E6dkI" 
INDEX_FOLDER = "faiss_index_ae"

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المبادر الذاتي - Assistant", page_icon="🇹🇳", layout="centered")

# --- CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stChatMessage {
        direction: rtl !important;
        text-align: right !important;
        border-radius: 15px;
        padding: 15px;
        font-size: 16px;
    }
    div[data-testid="stChatMessageContent"] {
        direction: rtl !important;
        text-align: right !important;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background-color: #f0f2f6;
        color: #1f77b4;
        border: 1px solid #d6d6d6;
        font-family: 'Cairo', sans-serif;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #e2e6ea;
        border-color: #1f77b4;
    }
    h1, h2, h3 { 
        color: #1f77b4; 
        text-align: center; 
        font-family: 'Cairo', sans-serif;
    }
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- الدوال (Functions) ---

def get_all_files_text(file_list):
    text = ""
    for file_path in file_list:
        try:
            if file_path.endswith('.pdf'):
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text += f.read() + "\n"
        except FileNotFoundError:
            st.warning(f"⚠️ الملف {file_path} مفقود.")
        except Exception as e:
            st.error(f"خطأ في قراءة الملف {file_path}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store_with_batches(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = None
    batch_size = 5
    total_chunks = len(text_chunks)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i in range(0, total_chunks, batch_size):
            batch = text_chunks[i:i+batch_size]
            progress = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"جاري تدريب الموديل: {int(progress*100)}% ...")

            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)
            time.sleep(1)
            
        vector_store.save_local(INDEX_FOLDER)
        status_text.success("✅ تم تحديث الموديل بنجاح!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        return True
    except Exception as e:
        st.error(f"خطأ تقني: {e}")
        return False

def get_gemini_response_with_suggestions(context_text, user_question, api_key):
    client = genai.Client(api_key=api_key)
    
    # --- Prompt ذكي يطلب الجواب + 3 اقتراحات ---
    prompt = f"""
    أنت المساعد الذكي الرسمي لمنصة "المبادر الذاتي" في تونس.
    
    السياق (المعلومات):
    {context_text}
    
    سؤال المواطن:
    {user_question}
    
    🔴 تعليمات صارمة (Red Lines):
    1. **التحية**: إذا كانت مجرد تحية (سلام، صباح الخير)، جاوب بترحيب فقط ولا تقترح أسئلة معقدة.
    2. **اللغة**: العربية الفصحى فقط.
    3. **التنسيق**: أريد الإجابة في شكل JSON يحتوي على حقلين:
       - "answer": نص الإجابة (منسق بخطوات إذا لزم الأمر).
       - "suggestions": قائمة فيها بالضبط 3 أسئلة قصيرة لها علاقة مباشرة بموضوع السؤال الحالي (لتسهيل الحوار).
    4. **المحتوى**: لا تذكر الرموز التقنية (IHM, Zone).
    
    مثال للنتيجة المطلوبة (JSON):
    {{
      "answer": "نص الإجابة هنا...",
      "suggestions": ["سؤال مقترح 1", "سؤال مقترح 2", "سؤال مقترح 3"]
    }}
    
    جاوب الآن بصيغة JSON فقط:
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-flash-latest', # نستعملو Flash باش يكون سريع في توليد JSON
            contents=prompt,
            config={'response_mime_type': 'application/json'} # نجبدو JSON صافي
        )
        return json.loads(response.text)
    except Exception as e:
        # في صورة ما صار خطأ، نرجعو جواب عادي واقتراحات عامة
        return {
            "answer": "عذراً، حدث خطأ مؤقت. الرجاء المحاولة مرة أخرى.",
            "suggestions": ["ما هي شروط الانخراط؟", "كيف أدفع؟", "اتصل بالدعم"]
        }

def process_query(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    try:
        new_db = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=6)
        context = "\n".join([doc.page_content for doc in docs])
        return get_gemini_response_with_suggestions(context, user_question, api_key)
    except Exception:
        return {
            "answer": "⚠️ النظام غير جاهز. الرجاء تحديث البيانات.",
            "suggestions": []
        }

# --- الواجهة الرئيسية (Main) ---

def main():
    st.title("🇹🇳 المساعد الذكي للمبادر الذاتي")
    st.markdown("<p style='text-align: center; color: gray;'>أنا هنا لمساعدتك في كل ما يخص نظام المبادر الذاتي</p>", unsafe_allow_html=True)

    # 1. تهيئة الـ Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "مرحباً بك! 👋\nأنا المساعد الآلي لمنصة المبادر الذاتي.\n\nتفضل، كيف يمكنني مساعدتك اليوم؟"}
        ]
    
    # 2. تهيئة الاقتراحات (Suggestions)
    if "current_suggestions" not in st.session_state:
        st.session_state.current_suggestions = ["ما هي شروط الانخراط؟", "كيف أدفع المساهمات؟", "الوثائق المطلوبة؟"]

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://www.autoentrepreneur.tn/assets/images/logo-ae.png", width=150)
        st.header("الإعدادات")
        
        if os.path.exists(f"{INDEX_FOLDER}/index.faiss"):
            st.success("البيانات متصلة 🟢")
        else:
            st.error("البيانات غير موجودة 🔴")
            
        if st.button("🔄 إطلاق التدريب (Entrainement)"):
            with st.spinner("جاري التحديث..."):
                files_to_process = [
                    "TDRS AE  PHASE 1_07-2024.pdf", 
                    "projet cahier des charges phase II Autoentrepreneur.pdf",
                    "rapport-auto-entrepreneur.pdf",
                    "more_data.txt"
                ]
                existing_files = [f for f in files_to_process if os.path.exists(f)]
                if existing_files:
                    raw_text = get_all_files_text(existing_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        create_vector_store_with_batches(text_chunks, FIXED_API_KEY)
                        st.rerun()

    # 3. عرض المحادثة
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. عرض الأزرار الديناميكية (Dynamic Buttons)
    # تظهر فقط إذا كان آخر ميساج من عند الـ Assistant
    if st.session_state.messages[-1]["role"] == "assistant":
        suggestions = st.session_state.current_suggestions
        if suggestions:
            st.markdown("###### أسئلة مقترحة:")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion, key=f"sugg_{len(st.session_state.messages)}_{i}"):
                    handle_user_input(suggestion)

    # 5. خانة الكتابة
    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        handle_user_input(prompt)

def handle_user_input(prompt):
    # عرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # معالجة الجواب
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("جاري المعالجة..."):
            # نتحصلو على الـ JSON (الجواب + الاقتراحات)
            result_json = process_query(prompt, FIXED_API_KEY)
            
            full_response = result_json.get("answer", "عذراً، لا توجد إجابة.")
            new_suggestions = result_json.get("suggestions", [])
            
            message_placeholder.markdown(full_response)
    
    # تحديث الحالة (State)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # تحديث الاقتراحات للأزرار القادمة
    if new_suggestions:
        st.session_state.current_suggestions = new_suggestions
    else:
        st.session_state.current_suggestions = [] # تفريغ إذا مفماش اقتراحات
        
    st.rerun()

if __name__ == "__main__":
    main()
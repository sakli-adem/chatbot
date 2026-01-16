import streamlit as st
import os
import json
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 🔑 API Key ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ لم يتم العثور على ملف الأسرار (secrets.toml).")
    st.stop()

INDEX_FOLDER = "faiss_index_ae"

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المبادر الذاتي - Assistant", page_icon="🇹🇳", layout="centered")

# --- CSS Styling (RTL + Hide Sidebar) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

    /* 1. إخفاء القائمة الجانبية تماماً */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* 2. تطبيق الخط والعربية على كامل التطبيق */
    html, body, .stApp {
        font-family: 'Cairo', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }

    /* 3. قلب اتجاه رسائل الشات */
    .stChatMessage {
        flex-direction: row-reverse !important;
        text-align: right !important;
        direction: rtl !important;
        gap: 10px;
    }
    
    /* 4. تصليح المحتوى داخل الرسالة */
    div[data-testid="stChatMessageContent"] {
        text-align: right !important;
        direction: rtl !important;
        margin-right: 10px !important;
        margin-left: 0px !important;
    }

    /* 5. تصليح مكان الـ Avatar */
    .stChatMessage .stChatMessageAvatar {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /* 6. تصليح القوائم والنقاط */
    ul, ol {
        direction: rtl !important;
        text-align: right !important;
        margin-right: 20px !important;
    }
    
    /* 7. تصليح خانة الكتابة */
    .stChatInputContainer textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    /* 8. الأزرار */
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
    
    /* 9. العناوين والنصوص */
    p, h1, h2, h3, h4, h5, h6, span, div {
        text-align: right;
    }
    
    /* إخفاء زر Deploy */
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)


# --- Fonctions Chat (Inference Only) ---

def get_gemini_response_with_suggestions(context_text, user_question, api_key):
    client = genai.Client(api_key=api_key)
    
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
            model='gemini-flash-latest', 
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "answer": "عذراً، حدث خطأ مؤقت في الاتصال. الرجاء المحاولة مرة أخرى.",
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
            "answer": "⚠️ النظام غير جاهز (قاعدة البيانات مفقودة).",
            "suggestions": []
        }

# --- Main UI ---

def main():
    # Logo Centré en haut
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://www.autoentrepreneur.tn/assets/images/logo-ae.png", use_container_width=True)

    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>المساعد الذكي للمبادر الذاتي</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>مرحباً بك، أنا هنا لمساعدتك في كل ما يخص نظام المبادر الذاتي</p>", unsafe_allow_html=True)

    # 1. Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "مرحباً بك! 👋\nأنا المساعد الآلي لمنصة المبادر الذاتي.\n\nتفضل، كيف يمكنني مساعدتك اليوم؟"}
        ]
    
    if "current_suggestions" not in st.session_state:
        st.session_state.current_suggestions = ["ما هي شروط الانخراط؟", "كيف أدفع المساهمات؟", "الوثائق المطلوبة؟"]

    # 2. Affichage Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Boutons Dynamiques
    if st.session_state.messages[-1]["role"] == "assistant":
        suggestions = st.session_state.current_suggestions
        if suggestions:
            st.markdown("###### أسئلة مقترحة:")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion, key=f"sugg_{len(st.session_state.messages)}_{i}"):
                    handle_user_input(suggestion)

    # 4. Input Area
    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        handle_user_input(prompt)

def handle_user_input(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("جاري المعالجة..."):
            result_json = process_query(prompt, api_key)
            
            full_response = result_json.get("answer", "عذراً، لا توجد إجابة.")
            new_suggestions = result_json.get("suggestions", [])
            
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    if new_suggestions:
        st.session_state.current_suggestions = new_suggestions
    else:
        st.session_state.current_suggestions = []
        
    st.rerun()

if __name__ == "__main__":
    main()
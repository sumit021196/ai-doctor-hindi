import streamlit as st
from transformers import pipeline

# Load the model with caching
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

pipe = load_model()

# Streamlit page setup
st.set_page_config(page_title="AI Doctor - हिंदी", layout="centered")
st.title("🩺 AI Doctor (हिंदी में)")
st.markdown("**कृपया अपने लक्षण नीचे हिंदी में लिखें (जैसे - बुखार, खांसी, गले में दर्द)।**")

# Input box for symptoms
symptoms = st.text_area("लक्षण दर्ज करें:", height=150)

# Diagnose button
if st.button("🧠 निदान करें"):
    if not symptoms.strip():
        st.warning("कृपया कुछ लक्षण दर्ज करें।")
    else:
        prompt = f"""
        मरीज ने बताया: {symptoms}
        आप एक अनुभवी डॉक्टर हैं। कृपया संभावित बीमारी, ज़रूरी टेस्ट और दवाइयाँ हिंदी में सरल भाषा में बताएं।
        """
        try:
            result = pipe(prompt, max_new_tokens=200)[0]['generated_text']
            st.success("✅ निदान और परामर्श:")
            st.write(result)
        except Exception as e:
            st.error("मॉडल से उत्तर प्राप्त करने में त्रुटि हुई। कृपया पुनः प्रयास करें।")
            st.exception(e)

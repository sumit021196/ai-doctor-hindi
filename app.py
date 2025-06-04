import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer only once
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.set_page_config(page_title="AI Doctor - Hindi", layout="centered")
st.title("🩺 AI Doctor (हिंदी में)")

st.markdown("कृपया अपने लक्षण नीचे लिखें (जैसे - बुखार, खांसी, गले में दर्द)।")

# Input Box
symptoms = st.text_area("लक्षण दर्ज करें:", height=150)

if st.button("🧠 निदान करें (Diagnose)"):
    if symptoms.strip() == "":
        st.warning("कृपया कुछ लक्षण दर्ज करें।")
    else:
        # Create Prompt
        prompt = f"""
        मरीज ने बताया: {symptoms}
        आप एक अनुभवी डॉक्टर हैं। कृपया संभावित बीमारी, ज़रूरी टेस्ट और दवाइयाँ हिंदी में सरल भाषा में बताएं।
        """

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=200)

        # Decode and show result
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success("✅ निदान और परामर्श:")
        st.write(result)

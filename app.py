import streamlit as st
from model import train_model, translate

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="English → French Translator",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return train_model()

with st.spinner("Building Transformer Model..."):
    transformer, source_vec, target_vec = load_model()

# ----------------------------
# Header
# ----------------------------
st.title("🌍 English → French Translator")

st.caption(
    "Custom Transformer Encoder-Decoder Neural Network built with TensorFlow"
)

st.divider()

# ----------------------------
# Metrics
# ----------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Dataset", "10 Pairs")

with m2:
    st.metric("Embedding", "128")

with m3:
    st.metric("Heads", "4")

with m4:
    st.metric("Dense Dim", "256")

st.divider()

# ----------------------------
# Main Layout
# ----------------------------
left, right = st.columns([3, 1])

with left:

    sentence = st.text_area(
        "Enter English Sentence",
        placeholder="Example: i like coffee",
        height=120
    )

    translate_btn = st.button(
        "🚀 Translate",
        type="primary",
        use_container_width=True
    )

    if translate_btn:

        if sentence.strip() == "":
            st.warning("Please enter a sentence.")

        else:

            with st.spinner("Translating..."):

                output = translate(
                    sentence,
                    transformer,
                    source_vec,
                    target_vec
                )

            st.subheader("🇫🇷 French Translation")

            st.success(output)

            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append(
                (sentence, output)
            )

with right:

    st.subheader("📌 Examples")

    st.code("i like coffee")

    st.code("thank you")

    st.code("good morning")

    st.code("where are you going")

    st.code("i am a student")

# ----------------------------
# History
# ----------------------------
st.divider()

st.subheader("📜 Translation History")

if "history" in st.session_state and len(st.session_state.history) > 0:

    for eng, fr in reversed(st.session_state.history):

        with st.container():

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"🇬🇧 {eng}")

            with col2:
                st.success(f"🇫🇷 {fr}")

else:
    st.caption("No translations yet.")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    st.title("⚙️ Model Information")

    st.metric(
        "Training Samples",
        "10"
    )

    st.metric(
        "Vocabulary Size",
        "1000"
    )

    st.metric(
        "Attention Heads",
        "4"
    )

    st.metric(
        "Embedding Size",
        "128"
    )

    st.divider()

    st.warning(
        """
        This is a demonstration model trained on only 10 sentence pairs.

        Translation quality is limited because real-world Transformers are trained on thousands or millions of examples.
        """
    )

    st.divider()

    st.markdown("### 📚 Dataset")

    st.markdown("""
    - i like coffee
    - thank you
    - welcome
    - good morning
    - how are you
    """)
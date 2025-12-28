import os
import pickle
import time

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

# --- 1. APP CONFIG ---
st.set_page_config(
    page_title="Yaqeen-AI | MTL Content Safety",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; }
    
    /* Text Inputs */
    .stTextArea textarea {
        background-color: #262730;
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid #41444e;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.2);
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }

    /* Custom Cards */
    .metric-card {
        background-color: #1f2229;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .safe-card { border-left: 5px solid #00c853 !important; }
    
    /* Typography */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    .sub-text { color: #a0a0a0; font-size: 14px; }
</style>
""",
    unsafe_allow_html=True,
)


# --- 2. DATA MAPPING ---
LABEL_MAP = {
    "GH": {
        "title": "Gender-Based Hate",
        "desc": "Content promoting hostility, discrimination, or violence based on gender or sexual orientation.",
        "icon": "⚥",
    },
    "OH": {
        "title": "Origin-based Hate",
        "desc": "Content inciting hatred against individuals based on nationality, ethnicity, or race.",
        "icon": "🌍",
    },
    "IH": {
        "title": "Ideology-based Hate",
        "desc": "Content promoting violence or prejudice against religious or ideological beliefs or groups.",
        "icon": "🕌",
    },
    "Normal": {
        "title": "Safe Content",
        "desc": "No hate speech detected.",
        "icon": "✅",
    },
}


# --- 3. MODEL ARCHITECTURE (Required for loading) ---
class MTL_Marbert(nn.Module):
    def __init__(self, model_name, n_multi_classes):
        super(MTL_Marbert, self).__init__()

        # 1. Shared Encoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)

        # 2. Binary Head (Is Hate?)
        self.binary_head = nn.Linear(self.bert.config.hidden_size, 1)

        # 3. Multi-class Head (Type of Hate)
        self.multi_head = nn.Linear(self.bert.config.hidden_size, n_multi_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Using the pooler output equivalent (CLS token state)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)

        binary_logits = self.binary_head(output)
        multi_logits = self.multi_head(output)

        return binary_logits, multi_logits


# --- 4. LOAD RESOURCES ---
@st.cache_resource
def load_mtl_resources(model_dir):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # 2. Load Label Encoder
        le_path = os.path.join(model_dir, "label_encoder.pkl")
        with open(le_path, "rb") as f:
            le = pickle.load(f)

        num_classes = len(le.classes_)

        # 3. Initialize Model Architecture
        # We use the base name here because the weights file contains the fine-tuned weights
        model = MTL_Marbert("UBC-NLP/MARBERTv2", n_multi_classes=num_classes)

        # 4. Load Weights
        weights_path = os.path.join(model_dir, "model_weights.pth")
        model.load_state_dict(torch.load(weights_path, map_location=device))

        model.to(device)
        model.eval()

        return tokenizer, model, le, device

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


# --- MODEL PATH ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/multitask_marbert")

# Load Models
with st.spinner("🚀 Initializing Multi-Task AI Engine..."):
    tokenizer, model, le, device = load_mtl_resources(MODEL_PATH)


# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/12532/12532687.png", width=60)
    st.title("Yaqeen-AI")
    st.markdown("---")
    st.markdown("### ⚙️ System Status")

    if model:
        st.success("🟢 MTL Model Online")
        st.caption(f"Device: {str(device).upper()}")
    else:
        st.error("🔴 Model Offline")
        st.warning(f"Check path: `{MODEL_PATH}`")

    st.markdown("### ℹ️ About")
    st.info(
        "This system uses a **Multi-Task Learning (MTL)** architecture based on MARBERTv2. "
        "It simultaneously predicts safety and hate categories in a single pass."
    )
    st.markdown("---")
    st.caption("v2.1 | MTL Build")


# --- 6. MAIN INTERFACE ---
st.markdown("## 🛡️ Hateful Language Detector (MTL)")
st.markdown(
    "<p class='sub-text'>Paste Arabic text below to scan for hate speech using Multi-Task Learning.</p>",
    unsafe_allow_html=True,
)

# Input Section
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area(
            "Input Stream", height=120, placeholder="أدخل النص هنا للتحليل..."
        )
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Spacer
        scan_btn = st.button("🔍 SCAN CONTENT")

# Logic
if scan_btn and user_input:
    if not model:
        st.error("Please configure the model path in the code.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Tokenizing input...")
        progress_bar.progress(20)

        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)

        status_text.text("Running Multi-Task Inference...")
        progress_bar.progress(60)

        # --- INFERENCE ---
        with torch.no_grad():
            # The model returns two outputs: binary_logits (1 value) and multi_logits (4 values)
            binary_logits, multi_logits = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )

            # Process Binary Head (Sigmoid because output dim is 1)
            bin_prob = torch.sigmoid(binary_logits).item()
            is_hate = bin_prob > 0.5

            # Process Multi-class Head (Softmax)
            multi_probs = F.softmax(multi_logits, dim=1)
            class_id = torch.argmax(multi_probs, dim=1).item()
            raw_label = le.inverse_transform([class_id])[0]
            multi_conf = multi_probs[0][class_id].item()

        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()

        # --- PREPARE RESULTS ---
        final_result = {}

        if not is_hate:
            # SAFE (Binary head says Safe)
            # We use the complement of the hate probability as confidence
            confidence = 1.0 - bin_prob
            final_result = {
                "type": "Safe",
                "label": "Normal",
                "conf": confidence,
                "color": "green",
            }
        else:
            # HATE (Binary head says Hate)
            final_result = {
                "type": "Unsafe",
                "label": raw_label,  # Use the result from the multi-head
                "conf": multi_conf,
                "color": "red",
            }

        # --- 7. RESULTS DISPLAY ---
        st.divider()

        lbl_info = LABEL_MAP.get(final_result["label"], LABEL_MAP["Normal"])

        r_col1, r_col2 = st.columns([1, 2])

        with r_col1:
            # Big Metric Card
            if final_result["type"] == "Safe":
                st.markdown(
                    f"""
                <div class="metric-card safe-card">
                    <h3 style="color:#00c853; margin:0;">{lbl_info["icon"]} No Hate Detected</h3>
                    <p style="font-size:40px; font-weight:bold; margin:0;">{final_result["conf"] * 100:.1f}%</p>
                    <p class="sub-text">Safety Score</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="metric-card" style="border-left-color: #ff4b4b;">
                    <h3 style="color:#ff4b4b; margin:0;">{lbl_info["icon"]} HATE</h3>
                    <p style="font-size:40px; font-weight:bold; margin:0;">{final_result["conf"] * 100:.1f}%</p>
                    <p class="sub-text">Confidence Level</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        with r_col2:
            # Detailed Info
            st.subheader("Analysis Report")
            if final_result["type"] == "Safe":
                st.success(f"**Classification:** {lbl_info['title']}")
                st.markdown(f"_{lbl_info['desc']}_")
            else:
                st.error(f"**Classification:** {lbl_info['title']}")
                st.markdown(f"_{lbl_info['desc']}_")

                st.write("Model Certainty:")
                st.progress(final_result["conf"])

                with st.expander("Show Technical Details"):
                    st.json(
                        {
                            "Model Architecture": "Multi-Task MARBERTv2",
                            "Binary Head Output": f"{bin_prob:.4f} (Threshold: 0.5)",
                            "Multi-Head Prediction": final_result["label"],
                            "Multi-Head Confidence": f"{multi_conf:.4f}",
                        }
                    )

elif scan_btn and not user_input:
    st.toast("⚠️ Please enter text before scanning.", icon="⚠️")

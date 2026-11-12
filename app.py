import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import gradio as gr

# Import model architecture and multi-digit computer vision engine
from train_model import MNISTNet, segment_and_predict_digits, get_device, train_and_save_model

# 1. Initialize Compute Device & Load PyTorch Model
device = get_device()
print(f"[*] Initializing DigitMind App on device: {device}")

model_path = "mnist_cnn.pth"
model = MNISTNet().to(device)

if os.path.exists(model_path):
    print(f"[+] Loading trained model checkpoint from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("[*] Training PyTorch CNN model...")
    model = train_and_save_model(model_save_path=model_path, epochs=2)

model.eval()


# 2. Prediction Function for Gradio Interface
def predict_handwritten_digits(input_sketch):
    if input_sketch is None:
        return "READY", "Please write one or more digits on the canvas.", None

    # Parse input from Gradio Sketchpad (dictionary or numpy array)
    if isinstance(input_sketch, dict):
        img_data = input_sketch.get("composite", input_sketch.get("image", None))
    else:
        img_data = input_sketch

    if img_data is None:
        return "NO INPUT", "Canvas is empty.", None

    # Run multi-digit segmentation and PyTorch CNN inference
    seq_str, details, annotated_img = segment_and_predict_digits(img_data, model, device)

    if not seq_str:
        return "NO DIGITS FOUND", "No clear digits detected. Try drawing thicker strokes.", img_data

    # Format text breakdown for UI display
    breakdown_lines = [f"🏆 FULL DETECTED SEQUENCE: {seq_str}\n"]
    for d in details:
        breakdown_lines.append(
            f"  • Digit #{d['digit_index']}: PREDICTED '{d['predicted_digit']}' "
            f"(Confidence: {d['confidence']}%)"
        )
        top3_str = ", ".join([f"{item['digit']}: {item['probability']*100:.1f}%" for item in d['top3']])
        breakdown_lines.append(f"    Top-3 Probabilities: [{top3_str}]")

    breakdown_text = "\n".join(breakdown_lines)
    
    # Convert BGR annotated image to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    return seq_str, breakdown_text, annotated_rgb


# 3. Attention-Seeking Cyber Dark Theme UI Design
custom_css = """
body, .gradio-container {
    background-color: #0f172a !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
    color: #f8fafc !important;
}
.gradio-container {
    max-width: 1000px !important;
    margin: 20px auto !important;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6) !important;
    padding: 30px !important;
}
#title-header h1 {
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    margin-bottom: 5px !important;
}
#title-header p {
    color: #94a3b8 !important;
    text-align: center !important;
    font-size: 1rem !important;
    margin-bottom: 25px !important;
}
.seq-box textarea {
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: #38bdf8 !important;
    text-align: center !important;
    background: #020617 !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 16px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="cyan")) as app:
    gr.Markdown(
        """
        # ⚡ DIGITMIND: MULTI-DIGIT RECOGNIZER
        Write single digits or multi-digit numbers (e.g. `427`, `901`) on the canvas to trigger real-time neural detection!
        """,
        elem_id="title-header"
    )

    with gr.Row():
        with gr.Column(scale=5):
            canvas = gr.Sketchpad(
                label="🖊️ Cybernetic Drawing Canvas (Write multiple digits side-by-side)",
                brush=gr.Brush(colors=["#FFFFFF"], default_size=16),
                height=260,
            )
            with gr.Row():
                predict_btn = gr.Button("✨ Predict Multi-Digits", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear Canvas", variant="secondary", size="lg")

        with gr.Column(scale=5):
            output_seq = gr.Textbox(
                label="🎯 Detected Number Sequence",
                placeholder="Result sequence appears here...",
                elem_classes=["seq-box"]
            )
            output_img = gr.Image(
                label="🔍 OpenCV Bounding Boxes Overlay",
                type="numpy"
            )
            output_details = gr.Textbox(
                label="📊 Per-Digit Confidence Breakdown",
                lines=7
            )

    predict_btn.click(
        fn=predict_handwritten_digits,
        inputs=canvas,
        outputs=[output_seq, output_details, output_img]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, None),
        inputs=None,
        outputs=[canvas, output_seq, output_details]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching DigitMind Web App...")
    print("👉 Open in your browser: http://127.0.0.1:7860")
    print("="*60 + "\n")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)

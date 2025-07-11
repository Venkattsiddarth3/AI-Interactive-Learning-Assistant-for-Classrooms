# Text and Image Generation Code.

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO
import torch
import datetime

app = Flask(__name__)
CORS(app)

# Text Generation/ Chat Assistant Model

try:
    print("🚀 Trying to load TinyLlama...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    chat_template = True
    print("✅ Loaded TinyLlama.")
except Exception as e:
    print(f"⚠️ TinyLlama failed: {e}")
    print("🔄 Falling back to distilgpt2.")
    model_id = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    chat_template = False
    print("✅ Loaded distilgpt2.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


#  Load Stable Diffusion Model for Image Generation
device_sd = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading Stable Diffusion on {device_sd}...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16 if device_sd == "cuda" else torch.float32
).to(device_sd)
print("✅ Stable Diffusion ready.")


#  Chat history storage

log_file = "chat_log.txt"
history_list = []

# LLM text generation endpoint

@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    user_prompt = data.get("query", "").strip()
    if not user_prompt:
        return jsonify({"error": "Empty query."}), 400

    # Prepare input for model
    if chat_template:
        formatted_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    else:
        formatted_prompt = user_prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    start = datetime.datetime.now()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=300,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = (datetime.datetime.now() - start).total_seconds()
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = decoded_output.split("<|assistant|>")[-1].strip() if chat_template else decoded_output.strip()

    # Save to history and file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_list.append({
        "user": user_prompt,
        "assistant": response,
        "timestamp": timestamp,
        "response_time_sec": round(elapsed, 2)
    })
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\nYou: {user_prompt}\nAI: {response}\n\n")

    return jsonify({"response": response})

@app.route("/api/generate_image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    with torch.no_grad():
        image = pipe(prompt).images[0]

    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

#  Get chat history

@app.route("/api/chat_history")
def chat_history():
    return jsonify(history_list[-50:])


#  Download chat log file

@app.route("/download_log")
def download_log():
    return send_file(log_file, as_attachment=True)


@app.route("/analyze_emotion", methods=["POST"])
def analyze_emotion():
    return jsonify({"emotion": "Happy"})

#  Run the server 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

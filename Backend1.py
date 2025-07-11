from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # this fixes most browser fetch block issues

# Load TinyLlama
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("🚀 Loading TinyLlama...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
device = torch.device("cpu")
model.to(device)
model.eval()
print("✅ TinyLlama ready.")

log_file = "chat_log.txt"
history_list = []

@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    user_prompt = data.get("query", "").strip()
    if not user_prompt:
        return jsonify({"error": "Empty query."}), 400

    formatted_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    start = datetime.datetime.now()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=600,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    response_time = (datetime.datetime.now() - start).total_seconds()

    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = decoded_output.split("<|assistant|>")[-1].strip()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_list.append({
        "user": user_prompt,
        "assistant": response,
        "timestamp": timestamp,
        "response_time_sec": round(response_time, 2)
    })

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\nYou: {user_prompt}\nAI: {response}\n\n")

    print(f"\n🧑 You: {user_prompt}")
    print(f"🤖 AI ({round(response_time,2)}s): {response}\n")

    return jsonify({"response": response})

@app.route("/api/chat_history")
def chat_history():
    return jsonify(history_list[-50:])

@app.route("/download_log")
def download_log():
    return send_file(log_file, as_attachment=True)

@app.route("/api/generate_image", methods=["POST"])
def generate_image():
    img = Image.new('RGB', (512, 512), color=(173, 216, 230))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route("/analyze_emotion", methods=["POST"])
def analyze_emotion():
    return jsonify({"emotion": "Happy"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
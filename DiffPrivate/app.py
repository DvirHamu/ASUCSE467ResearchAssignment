import os, uuid, torch
from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from omegaconf import OmegaConf
from src.attCtr import AttentionControlEdit
import src.diffprivate_pert as diffprivate_pert
from src.utils import seed_torch

app = Flask(__name__)
UPLOAD_FOLDER = "data/uploads"
OUTPUT_FOLDER = "data/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model once at startup
print("Loading model...")
seed_torch(42)
cfg = OmegaConf.load("configs/config.yaml")
model = StableDiffusionPipeline.from_pretrained(cfg.paths.pretrained_diffusion_path).to("cuda:0")
if cfg.model.lora_path:
    from safetensors.torch import load_file
    lora_state = load_file(os.path.join(cfg.model.lora_path, "pytorch_lora_weights.safetensors"))
    lora_state = {k.replace("unet.", ""): v for k, v in lora_state.items()}
    model.unet.load_state_dict(lora_state, strict=False)
model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
print("Model ready.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/protect", methods=["POST"])
def protect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return jsonify({"error": "Only JPG/PNG supported"}), 400

    job_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.png")
    output_path = os.path.join(OUTPUT_FOLDER, job_id)

    Image.open(file).convert("RGB").save(input_path)

    controller = AttentionControlEdit(
        cfg.diffusion.diffusion_steps,
        cfg.diffusion.self_replace_steps,
        cfg.diffusion.res,
    )

    diffprivate_pert.protect(
        model=model,
        controller=controller,
        args=cfg,
        image_path=input_path,
        save_path=output_path,
    )

    result_path = output_path + "_adv_image.png"
    return send_file(result_path, mimetype="image/png", as_attachment=True, download_name="protected.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

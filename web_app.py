from flask import Flask, request, render_template, send_file
import torch
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).convert("L").resize((512, 512))
        input_tensor = torch.tensor(np.array(img) / 255.0).unsqueeze(0).unsqueeze(0).float()
        heatmap = np.random.rand(512, 512)  # Simulated XAI heatmap
        plt.imshow(img, cmap="gray")
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

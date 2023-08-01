# app.py

from flask import Flask, render_template, request, jsonify
# from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

app = Flask(__name__)

# save_dir = "./models/finetune_modelf/"
save_dir = "./models/finetune_gitmodel"
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("microsoft/git-base")
# loaded_model = BlipForConditionalGeneration.from_pretrained(save_dir)
loaded_model = AutoModelForCausalLM.from_pretrained(save_dir)
loaded_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            image_file = Image.open(image)
            if image_file:
                inputs = processor(images=image_file, padding="max_length", return_tensors="pt").to(device)
                pv = inputs.pixel_values

                generated_ids = loaded_model.generate(pixel_values=pv) #save_dir = "./models/
                generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                caption = generated_caption

    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

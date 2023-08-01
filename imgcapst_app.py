from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import streamlit as st

save_dir_f = "./finetune_modelf"
save_dir_g = "./finetune_gitmodel"
processor_f = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
processor_g = AutoProcessor.from_pretrained("microsoft/git-base")
loaded_model_f = BlipForConditionalGeneration.from_pretrained(save_dir_f)
loaded_model_g = AutoModelForCausalLM.from_pretrained(save_dir_g)
loaded_model_f.eval()
loaded_model_g.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model_f.to(device)
loaded_model_g.to(device)

def image_captioning(image_file,model):
    # image_file = Image.open(img)

    if model==0:
        inputs = processor_f(images=image_file, padding="max_length", return_tensors="pt").to(device)
        pv = inputs.pixel_values
        generated_ids = loaded_model_f.generate(pixel_values=pv) #save_dir = "./models/
        generated_caption = processor_f.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = generated_caption
    else:
        inputs = processor_g(images=image_file, padding="max_length", return_tensors="pt").to(device)
        pv = inputs.pixel_values
        generated_ids = loaded_model_g.generate(pixel_values=pv) #save_dir = "./models/
        generated_caption = processor_g.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = generated_caption

    return caption

def main():
    st.title("X-ray image captioning model")

    # Display the file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Check if an image was uploaded
    if uploaded_file is not None:
        # Open the image using PIL
        pil_image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    # List of available models
    model_options = ["Microsoft Git Model", "Salesforce Blip Model"]
    
    # Display the selectbox to choose a model
    selected_model = st.selectbox("Select a model:", model_options)

    if st.button("Generate Caption"):
        # Display the selected model
        if selected_model == "Microsoft Git Model":
            st.write("You selected Microsoft Git Model.")
            # Call the function for Model 1
            caption = image_captioning(pil_image,1)
            st.text("Caption: " + caption)
        elif selected_model == "Salesforce Blip Model":
            st.write("You selected Salesforce Blip Model.")
            # Call the function for Model 2
            caption = image_captioning(pil_image,0)
            st.text("Caption: " + caption)

if __name__ == "__main__":
    main()

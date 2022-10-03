import torch
from diffusers import StableDiffusionPipeline
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import streamlit as st

mname = "facebook/wmt19-de-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

prompt = st.text_input("Gib mir deinen Input, Freund: ")

finished = False

if st.button("Submit"):
    print(prompt)
    finished = True

if finished:    
    print("Dein Input war auf deutsch: " + prompt)
    st.title(prompt)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(input_ids)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    i = 0
    while i < len(decoded)+33:
        print("-", end="")
        i = i + 1
        
    print("\nThis is your input translated: " + decoded) 

    i = 0
    while i < len(decoded)+33:
        print("-", end="")
        i = i + 1


    st.title(decoded)


    pipe = StableDiffusionPipeline.from_pretrained(
    	"CompVis/stable-diffusion-v1-4",
    	revision="fp16", 
    	torch_dtype=torch.float16, 
    	use_auth_token=True
    ).to("cuda")


    with torch.cuda.amp.autocast(True):
        image = pipe(decoded)["sample"][0]  

    #print("\nSaved your file as: " + "f" + decoded + ".png")    
    #image.save("f" + decoded + ".png")
    st.image(image)

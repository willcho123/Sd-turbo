import torch
from diffusers import AutoPipelineForText2Image

print("Cargando modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
    # FloatXX 32 o 16
    # Me indica la cantidad de decimales de precision
    # con el que el modelo creara la imagen
    # si mi pc me genera un imagen negra, cambio a 16
    
)

modelo = modelo.to("cpu")
# cuda - nvidia
# mps - Mac com procesadores MX
# xpu - Grafica Intel IRIS

prompt = input("Escribe el prompt de la imagen: ")

negative_prompt = "blurry, low quality, low resolution, distorted, deformed,bad anatomy, bad proportions, extra fingers, missing fingers,extra limbs, malformed hands, duplicate, cropped,text, watermark, logo, artifacts, noisy"


print("Generando la imagen...")

imagen = modelo(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10, # Numero de veces que creara y corregira la imagen
    guidance_scale=2.0, # Que tan fiel sera la prompt
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")
print("Imagen guardada !")

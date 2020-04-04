import matplotlib.pyplot as plt
from keras.models import Sequential,Model,model_from_json
import numpy as np
import GAN

def load_model():
    with open("generator.json", "r") as json_file:
        md_json = json_file.read()
    t = model_from_json(md_json)
    t.load_weights("generator.h5")
    return t

def generate_image(model):
    gan = GAN.GAN(0,0,True,0)
    gan.G = model
    gan.show_for_epoch(style=2,epoch='_numbers')

if __name__ == "__main__":
    model = load_model()
    generate_image(model)

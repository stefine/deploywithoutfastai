import gradio as gr
from PIL import Image
from transform import pad, crop, gpu_crop
from model import apply_weights, get_model, copy_weight
import torch
from torchvision.transforms import ToTensor, Normalize

vocab = ['Abyssinian', 'Bengal', 'Birman',
         'Bombay', 'British_Shorthair', 'Egyptian_Mau',
         'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
         'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
         'basset_hound', 'beagle', 'boxer', 'chihuahua',
         'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
         'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
         'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian',
         'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
         'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

# load the model
model = get_model()
state = torch.load('models/exported_model.pth', map_location="cpu")
apply_weights(model, state, copy_weight)
# model.cuda()

to_tensor = ToTensor()
norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def classfiy_image(inp):
    inp = Image.fromarray(inp)
    size = (460, 460)
    transform_inp = pad(crop(inp, size), size)
    transform_inp = gpu_crop(to_tensor(transform_inp).unsqueeze(0), (224, 224))
    transform_inp = norm(transform_inp)
    model.eval()
    with torch.no_grad():
        pred = model(transform_inp)
    pred = torch.argmax(pred, dim=-1)
    return vocab[pred]


iface = gr.Interface(
    fn=classfiy_image,
    inputs=gr.inputs.Image(shape=(224, 224)),
    outputs="text",
    title="No Fastai Classifier",
    description="An example of not using fastai in Gradio"
).launch()

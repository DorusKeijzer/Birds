import torch
from torchvision import transforms

class saliency_map_generator():
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


    def __call__(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        image.requires_grad_()


        forward_pass = model(image)
        model.zero_grad()
        predicted_class = forward_pass.argmax().item()
        forward_pass[0, predicted_class].backward()
        gradients = image.grad.data
        saliency, _ = torch.max(gradients.abs(), dim=1)
        saliency = saliency.squeeze()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        saliency = saliency.cpu().numpy()

        return saliency

if __name__ == "__main__":
    import sys
    import os
    from PIL import Image
    from matplotlib import pyplot as plt
    import numpy as np

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.torch_model import Flower_model

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model_path = sys.argv[1]
    model = Flower_model(525).to(device)

    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    for img in os.listdir("./data/train/SHOEBILL/"):
        print(img)
        image = Image.open(f"./data/train/SHOEBILL/{img}")
        
        smg = saliency_map_generator(model, device)
        saliency = smg(image)



        input_image = np.array(image.resize((224, 224)))  # Ensure the sizes match

        input_image = input_image / 255.0  # Normalize the image to [0,1]
        plt.imshow(input_image)
        plt.imshow(saliency, cmap='afmhot', alpha=0.5)
        plt.axis('off')
        plt.show()


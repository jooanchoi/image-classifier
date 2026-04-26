import argparse
import torch
import json
from PIL import Image
import numpy as np
from torchvision import models


def get_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to show')
    parser.add_argument('--category_names', type=str, help='Path to JSON mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage, weights_only=False)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Torch tensor
    '''
    img = Image.open(image_path)

    img.thumbnail((256, 10000) if img.width > img.height else (10000, 256))

    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    img = img.crop((left, top, left + 224, top + 224))

    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    return torch.from_numpy(np_image.transpose((2, 0, 1))).type(torch.FloatTensor)


def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()

    img = process_image(image_path).unsqueeze_(0).to(device)

    with torch.no_grad():
        output = model.forward(img)

    probs = torch.exp(output)
    top_p, top_indices = probs.topk(topk)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]

    return top_p[0].tolist(), top_classes


def main():
    args = get_args()
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model = load_checkpoint(args.checkpoint)

    probs, classes = predict(args.image_path, model, args.top_k, device)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        labels = [cat_to_name[c] for c in classes]
    else:
        labels = classes

    print(f"\nResults for image: {args.image_path}")
    for i in range(len(labels)):
        print(f"Rank {i + 1}: {labels[i]:<20} Probability: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
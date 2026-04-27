import argparse
import torch
from classifier import load_classifier, classify_image, get_class_names
from rag import load_rag, chat
import json

WEIGHTS = "../models/resnet_plants.pth"
SL_PATH = "../data/class_names.json"
KB_PATH = "../data/wiki_kb.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rag = load_rag(KB_PATH, device)

    species = None
    if args.image:
        with open(SL_PATH) as f:
            class_names = json.load(f)

        model = load_classifier(WEIGHTS, len(class_names), device)
        species = classify_image(args.image, model, class_names, device)
        print(f"predicted: {species}")

    chat(species, rag, device)

if __name__ == "__main__":
    main()
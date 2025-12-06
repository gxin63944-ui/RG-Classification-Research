import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
from model import get_resnet101

CONFIG = {
    'num_classes': 2,
    'img_size': 256,

    'class_names': ['Intact', 'Relict'], 

    'default_model_path': '/media/xsar/F/resnet/data/saved_models/resnet101_fold1_best.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model = get_resnet101(num_classes=CONFIG['num_classes'], pretrained=False)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=CONFIG['device'])
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Weight file not found at {model_path}")

    model.to(CONFIG['device'])
    model.eval() 
    return model

def process_image(image_path):
    """读取并预处理单张图片"""
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

def predict_single(model, image_path):
    tensor = process_image(image_path).to(CONFIG['device'])
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    class_name = CONFIG['class_names'][predicted_idx.item()]
    score = confidence.item()
    
    return class_name, score

def main():
    parser = argparse.ArgumentParser(description='Predict Rock Glacier Status')
    parser.add_argument('--input', type=str, required=True, help='Path to an image file or a directory of images')
    parser.add_argument('--model', type=str, default=CONFIG['default_model_path'], help='Path to .pth model weights')
    args = parser.parse_args()
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error: {e}")
        return

    if os.path.isfile(args.input):

        print(f"\nProcessing single image: {args.input}")
        cls, score = predict_single(model, args.input)
        print("-" * 30)
        print(f"Result: {cls}")
        print(f"Confidence: {score:.4f}")
        print("-" * 30)
        
    elif os.path.isdir(args.input):
        print(f"\nProcessing directory: {args.input}")
        print(f"{'Filename':<30} | {'Prediction':<10} | {'Confidence'}")
        print("-" * 60)
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_extensions)]
        
        for fname in sorted(files):
            fpath = os.path.join(args.input, fname)
            try:
                cls, score = predict_single(model, fpath)
                print(f"{fname:<30} | {cls:<10} | {score:.4f}")
            except Exception as e:
                print(f"{fname:<30} | Error: {e}")
                
    else:
        print("Invalid input path.")

if __name__ == '__main__':
    main()
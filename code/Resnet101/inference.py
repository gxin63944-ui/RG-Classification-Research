import torch
import os
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import numpy as np
from model import get_resnet101


CONFIG = {
    'num_classes': 2,
    'img_size': 256,
    'model_path': '/media/xsar/F/resnet/data/saved_models/resnet101_fold1_best.pth', # 训练好的权重路径
    'test_dir': '/media/xsar/F/resnet/data',  
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def majority_voting(predictions):

    counts = np.bincount(predictions)
    return np.argmax(counts) # 返回出现次数最多的类别

def run_inference():
    print(f"Running Inference on {CONFIG['device']}...")

    model = get_resnet101(num_classes=CONFIG['num_classes'], pretrained=False)
    

    if os.path.exists(CONFIG['model_path']):
        state_dict = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Weight file not found at {CONFIG['model_path']}. Using random weights.")
    
    model.to(CONFIG['device'])
    model.eval()


    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    polygon_groups = defaultdict(list)
    true_labels = {} # 记录真实标签用于对比

    class_names = ['intact', 'relict']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(CONFIG['test_dir'], class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.tif', '.tiff')):
                continue
            

            polygon_id = os.path.splitext(img_name)[0].rsplit('_', 1)[0] 
            
            img_path = os.path.join(class_dir, img_name)
            

            polygon_groups[polygon_id].append(img_path)
            true_labels[polygon_id] = class_idx

    print(f"Found {len(polygon_groups)} unique polygons.")


    correct_polygons = 0
    total_polygons = 0

    with torch.no_grad():
        for poly_id, img_paths in polygon_groups.items():
            patch_predictions = []
            

            for img_path in img_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    input_tensor = transform(img).unsqueeze(0).to(CONFIG['device'])
                    
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    patch_predictions.append(predicted.item())
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            if not patch_predictions:
                continue


            final_pred = majority_voting(patch_predictions)
            ground_truth = true_labels[poly_id]

            if final_pred == ground_truth:
                correct_polygons += 1
            
            total_polygons += 1
            

            # print(f"ID: {poly_id} | Patches: {patch_predictions} -> Vote: {final_pred} | GT: {ground_truth}")


    if total_polygons > 0:
        poly_acc = correct_polygons / total_polygons
        print(f"\n{'='*30}")
        print(f"Polygon-Level Evaluation (Majority Voting)")
        print(f"{'='*30}")
        print(f"Total Polygons: {total_polygons}")
        print(f"Accuracy: {poly_acc:.4f}")
    else:
        print("No images found to process.")

if __name__ == '__main__':
    run_inference()
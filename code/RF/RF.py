import os
import numpy as np
import cv2
import pandas as pd
import joblib
import time
from glob import glob
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

CONFIG = {
    'seed': 14,
    'data_dir': '/media/xsar/F/resnet/data', 
    'model_save_path': '/media/xsar/F/resnet/data/saved_models/rf_best_model.joblib',
    'n_folds': 5,
    'glcm_dist': [1], 
    'glcm_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'param_grid': {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }
}

def extract_features(image_path):

    # 读取图片 (OpenCV 读取默认为 BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error reading {image_path}")
        return None
    
    # 转为 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 生成掩膜 (Mask): 假设背景是全黑 (0,0,0)
    # 创建灰度图用于生成掩膜
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 0  # 有效区域为 True
    
    # 如果整张图都是黑的（异常数据），返回零特征
    if np.sum(mask) == 0:
        return np.zeros(9)

    r_mean = np.mean(img_rgb[:, :, 0][mask])
    g_mean = np.mean(img_rgb[:, :, 1][mask])
    b_mean = np.mean(img_rgb[:, :, 2][mask])
    brightness = (r_mean + g_mean + b_mean) / 3.0
    area = np.sum(mask)
    g_matrix = graycomatrix(gray, distances=CONFIG['glcm_dist'], 
                            angles=CONFIG['glcm_angles'], levels=256, 
                            symmetric=True, normed=True)
    contrast = np.mean(graycoprops(g_matrix, 'contrast'))
    homogeneity = np.mean(graycoprops(g_matrix, 'homogeneity'))
    asm = np.mean(graycoprops(g_matrix, 'ASM'))
    g_norm = g_matrix / (np.sum(g_matrix) + 1e-10) 
    entropy = -np.sum(g_norm * np.log(g_norm + 1e-10))

    features = np.array([
        r_mean, g_mean, b_mean, brightness,  # Spectral
        area,                                # Geometric
        contrast, entropy, homogeneity, asm  # Texture
    ])
    
    return features

def load_dataset(data_dir):
    print("Loading dataset and extracting features...")
    X = []
    y = []
    file_paths = []
    
    classes = ['intact', 'relict'] 
    
    start_time = time.time()
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue
            
        images = glob(os.path.join(class_dir, "*.*"))
        print(f"Processing {class_name}: {len(images)} images found.")
        
        for img_path in images:
            feats = extract_features(img_path)
            if feats is not None:
                X.append(feats)
                y.append(label) 
                file_paths.append(img_path)
                
    print(f"Feature extraction complete in {time.time() - start_time:.2f}s")
    return np.array(X), np.array(y)


def main():

    X, y = load_dataset(CONFIG['data_dir'])
    print(f"Total samples: {len(X)}")
    
    if len(X) == 0:
        print("No data loaded. Check data_dir path.")
        return

  
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    

    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    fold_metrics = {'acc': [], 'f1': [], 'precision': [], 'recall': []}
    
    print(f"\nStarting {CONFIG['n_folds']}-Fold Cross-Validation with Grid Search...")
    

    best_overall_acc = 0
    best_model = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n--- Fold {fold+1} ---")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        

        rf = RandomForestClassifier(random_state=CONFIG['seed'])
        grid_search = GridSearchCV(estimator=rf, param_grid=CONFIG['param_grid'], 
                                   cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_clf = grid_search.best_estimator_
        # print(f"Best params for fold {fold+1}: {grid_search.best_params_}")
        

        y_pred = best_clf.predict(X_val)
        
        # 计算指标
        acc = accuracy_score(y_val, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
        
        fold_metrics['acc'].append(acc)
        fold_metrics['f1'].append(f1)
        fold_metrics['precision'].append(p)
        fold_metrics['recall'].append(r)
        
        print(f"Fold {fold+1} Acc: {acc:.4f} | F1: {f1:.4f}")
        

        print(classification_report(y_val, y_pred, target_names=['Intact', 'Relict']))
        

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_model = best_clf


    print(f"\n{'='*30}")
    print("Final 5-Fold CV Results (RF)")
    print(f"{'='*30}")
    print(f"Average Accuracy:  {np.mean(fold_metrics['acc']):.4f} (±{np.std(fold_metrics['acc']):.4f})")
    print(f"Average Macro F1:  {np.mean(fold_metrics['f1']):.4f}")
    print(f"Average Precision: {np.mean(fold_metrics['precision']):.4f}")
    print(f"Average Recall:    {np.mean(fold_metrics['recall']):.4f}")

    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
    joblib.dump(best_model, CONFIG['model_save_path'])
    print(f"\nBest model saved to {CONFIG['model_save_path']}")
    print(f"Best model parameters: {best_model.get_params()}")

if __name__ == '__main__':
    main()
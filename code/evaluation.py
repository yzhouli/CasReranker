import numpy as np
import pickle, os, json
from tqdm import tqdm

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_metrics(pred_dict, test_db, K_list=[1,2,5,10,20,50]):
    results = {}
    for K in K_list:
        results[f'H@{K}'] = []
        results[f'M@{K}'] = []
        results[f'N@{K}'] = []
    
    for index, pred_list in pred_dict.items():
        idx = int(index)
        test_item = test_db[idx]
        next_user = test_item['next_user']
        
        if next_user in pred_list:
            rank = pred_list.index(next_user) + 1
        else:
            rank = float('inf')
        
        for K in K_list:
            if rank <= K:
                results[f'H@{K}'].append(1)
                results[f'M@{K}'].append(1.0 / rank)
                results[f'N@{K}'].append(1.0 / np.log2(rank + 1))
            else:
                results[f'H@{K}'].append(0)
                results[f'M@{K}'].append(0)
                results[f'N@{K}'].append(0)
    
    final_metrics = {metric: np.mean(values) for metric, values in results.items()}
    return final_metrics

if __name__ == "__main__":
    dataset_path = '../Casbench'
    test_li = load_pkl(path=f'{dataset_path}/test_hardneg.pkl')
    pred_dict = load_json(path='../results/predictions.json')
    metrics = compute_metrics(pred_dict, test_li)
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')

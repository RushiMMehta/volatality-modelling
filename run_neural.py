import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import hashlib
import json
import matplotlib.pyplot as plt
import pickle
import glob
import copy
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.manual_seed(42)
np.random.seed(42)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

ASSETS = ['nifty50', 'banknifty', 'adanient', 'tatasteel', 'dlf', 'hindunilvr', 'nestleind', 'sunpharma']
FEATURES = ['ewma_rmse_60', 'garch_rmse_60', 'egarch_rmse_60', 'gjr_garch_rmse_60', 
            'realized_vol_5', 'realized_vol_20', 'realized_vol_60', 'vol_of_vol_30', 'return_sign_lag1']
FORECASTS = ['ewma_forecast_variance', 'garch_forecast_variance', 'egarch_forecast_variance', 'gjr_garch_forecast_variance']
MODELS_LIST = ['EWMA', 'GARCH', 'EGARCH', 'GJR_GARCH', 'STATIC_INVERSE_RMSE', 'NEURAL_GATED']

TRAIN_START, TRAIN_END = '2022-01-03', '2023-12-29'
VAL_START, VAL_END = '2024-01-01', '2024-12-31'
TEST_START, TEST_END = '2025-01-01', '2025-12-30'

EPSILON = 1e-12

def numpy_qlike(y, f):
    y = np.maximum(y, EPSILON)
    f = np.maximum(f, EPSILON)
    ratio = y / f
    return np.mean(ratio - np.log(ratio) - 1.0)

def qlike_loss(y, f):
    y = torch.clamp(y, min=EPSILON)
    f = torch.clamp(f, min=EPSILON)
    ratio = y / f
    return torch.mean(ratio - torch.log(ratio) - 1.0)

class NeuralGate(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=12, output_dim=4):
        super(NeuralGate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

def get_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def get_all_protected_hashes():
    hashes = {}
    patterns = [
        'data/*.csv',
        'data/static_model_outputs/*.csv',
        'data/oos_forecasts/*.csv',
        'data/static_baseline/*.csv',
        'data/neural_gating/gating_features/*.csv'
    ]
    for p in patterns:
        for f in glob.glob(p):
            hashes[f] = get_hash(f)
    return hashes

def build_dataset(asset):
    feat_df = pd.read_csv(f'data/neural_gating/gating_features/{asset}_gating_features.csv')
    fc_df = pd.read_csv(f'data/oos_forecasts/{asset}_oos_forecasts.csv')
    df = pd.merge(feat_df, fc_df, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df

def get_split(df, start, end):
    mask = (df['date'] >= start) & (df['date'] <= end)
    return df[mask].copy()

def train_asset(asset):
    os.makedirs(f'data/neural_gating/model/{asset}', exist_ok=True)
    df = build_dataset(asset)
    
    train_df = get_split(df, TRAIN_START, TRAIN_END)
    val_df = get_split(df, VAL_START, VAL_END)
    test_df = get_split(df, TEST_START, TEST_END)
    
    scaler = {}
    zero_variance_fallbacks = []
    for feat in FEATURES:
        if feat == 'return_sign_lag1':
            continue
        std = train_df[feat].std()
        if std < 1e-8 or pd.isna(std):
            std = 1.0
            zero_variance_fallbacks.append(feat)
        mean = train_df[feat].mean()
        scaler[feat] = {'mean': mean, 'std': std, 'min': train_df[feat].min(), 'max': train_df[feat].max()}
        
    with open(f'data/neural_gating/model/{asset}/feature_scaler.pkl', 'wb') as f_out:
        pickle.dump({'scaler': scaler, 'fallbacks': zero_variance_fallbacks}, f_out)
        
    def apply_scaling(data):
        d = data.copy()
        for feat in FEATURES:
            if feat == 'return_sign_lag1': continue
            d[feat] = (d[feat] - scaler[feat]['mean']) / scaler[feat]['std']
        return d
        
    train_scaled = apply_scaling(train_df)
    val_scaled = apply_scaling(val_df)
    
    def get_tensors(d):
        X = torch.tensor(d[FEATURES].values, dtype=torch.float32)
        target = torch.tensor(d['realized_var_proxy'].values, dtype=torch.float32)
        forecasts = torch.tensor(d[FORECASTS].values, dtype=torch.float32)
        return X, target, forecasts
        
    X_tr, y_tr, f_tr = get_tensors(train_scaled)
    X_va, y_va, f_va = get_tensors(val_scaled)
    
    model = NeuralGate()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    dataset = TensorDataset(X_tr, y_tr, f_tr)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float('inf')
    best_epoch = -1
    best_weights = None
    patience = 15
    patience_counter = 0
    history = []
    
    for epoch in range(200):
        model.train()
        epoch_loss = 0.0
        for b_X, b_y, b_f in loader:
            optimizer.zero_grad()
            w = model(b_X)
            ens_var = torch.sum(w * b_f, dim=1)
            loss = qlike_loss(b_y, ens_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * b_X.size(0)
        
        train_loss = epoch_loss / len(loader.dataset)
        
        model.eval()
        with torch.no_grad():
            w_va = model(X_va)
            ens_var_va = torch.sum(w_va * f_va, dim=1)
            val_loss = qlike_loss(y_va, ens_var_va).item()
            
            if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
                print(f'Instability at epoch {epoch}')
                break
                
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f'data/neural_gating/model/{asset}/neural_gate_model.pth')
    
    with open(f'data/neural_gating/model/{asset}/model_config.json', 'w') as f_out:
        json.dump({'input_dim': 9, 'hidden_dim': 12, 'output_dim': 4}, f_out)
        
    pd.DataFrame(history).to_csv(f'data/neural_gating/model/{asset}/training_history.csv', index=False)
    
    model.eval()
    with torch.no_grad():
        w_tr = model(X_tr)
        ens_var_tr = torch.sum(w_tr * f_tr, dim=1)
        best_tr_qlike = qlike_loss(y_tr, ens_var_tr).item()
        tr_rmse = torch.sqrt(torch.mean((y_tr - ens_var_tr)**2)).item()
        
        w_va = model(X_va)
        ens_var_va = torch.sum(w_va * f_va, dim=1)
        best_va_qlike = qlike_loss(y_va, ens_var_va).item()
        va_rmse = torch.sqrt(torch.mean((y_va - ens_var_va)**2)).item()
    
    summary = f"Best Epoch: {best_epoch}\nBest Val QLIKE: {best_va_qlike}\nTrain QLIKE at best epoch: {best_tr_qlike}\nTrain RMSE at best epoch: {tr_rmse}\nVal RMSE at best epoch: {va_rmse}\nZero-variance fallbacks: {zero_variance_fallbacks}\n"
    with open(f'data/neural_gating/model/{asset}/training_summary.txt', 'w') as f_out:
        f_out.write(summary)
        
    return scaler, best_epoch, best_tr_qlike, best_va_qlike, zero_variance_fallbacks

def generate_predictions(asset):
    model = NeuralGate()
    model.load_state_dict(torch.load(f'data/neural_gating/model/{asset}/neural_gate_model.pth'))
    model.eval()
    
    with open(f'data/neural_gating/model/{asset}/feature_scaler.pkl', 'rb') as fin:
        scaler_data = pickle.load(fin)
    scaler = scaler_data['scaler']
    
    df = build_dataset(asset)
    pred_df = df.copy()
    
    for feat in FEATURES:
        if feat == 'return_sign_lag1': continue
        pred_df[feat] = (pred_df[feat] - scaler[feat]['mean']) / scaler[feat]['std']
        
    X = torch.tensor(pred_df[FEATURES].values, dtype=torch.float32)
    f_forecasts = torch.tensor(pred_df[FORECASTS].values, dtype=torch.float32)
    
    with torch.no_grad():
        w = model(X)
        ens_var = torch.sum(w * f_forecasts, dim=1)
    
    out_df = df[['date', 'log_return', 'realized_var_proxy', 'parkinson_var_proxy'] + FORECASTS].copy()
    out_df.insert(0, 'asset', asset)
    out_df['ewma_neural_weight'] = w[:, 0].numpy()
    out_df['garch_neural_weight'] = w[:, 1].numpy()
    out_df['egarch_neural_weight'] = w[:, 2].numpy()
    out_df['gjr_garch_neural_weight'] = w[:, 3].numpy()
    out_df['neural_ensemble_variance'] = ens_var.numpy()
    out_df['neural_ensemble_volatility'] = np.sqrt(ens_var.numpy())
    
    return out_df

def run_all():
    initial_hashes = get_all_protected_hashes()
    with open('data/neural_gating/initial_hashes.json', 'w') as f:
        json.dump(initial_hashes, f)
        
    with open('data/neural_gating/model/feature_order.txt', 'w') as f:
        f.write('\n'.join(FEATURES))
    with open('data/neural_gating/model/random_seed.txt', 'w') as f:
        f.write('42')
        
    all_preds = []
    asset_summaries = []
    all_fallbacks = {}
    
    for asset in ASSETS:
        scaler, best_ep, tr_qlike, va_qlike, fallbacks = train_asset(asset)
        all_fallbacks[asset] = fallbacks
        preds = generate_predictions(asset)
        all_preds.append(preds)
        
        test_preds = preds[(preds['date'] >= TEST_START) & (preds['date'] <= TEST_END)]
        y_test = test_preds['realized_var_proxy'].values
        f_test = test_preds['neural_ensemble_variance'].values
        
        te_mae = np.mean(np.abs(y_test - f_test))
        te_rmse = np.sqrt(np.mean((y_test - f_test)**2))
        te_qlike = numpy_qlike(y_test, f_test)
        
        asset_summaries.append({
            'Asset': asset,
            'Train rows': len(preds[(preds['date'] >= TRAIN_START) & (preds['date'] <= TRAIN_END)]),
            'Validation rows': len(preds[(preds['date'] >= VAL_START) & (preds['date'] <= VAL_END)]),
            'Test rows': len(test_preds),
            'Best epoch': best_ep,
            'Train QLIKE': tr_qlike,
            'Validation QLIKE': va_qlike,
            'Test MAE': te_mae,
            'Test RMSE': te_rmse,
            'Test QLIKE': te_qlike
        })
        
        plt.figure(figsize=(10,6))
        dates = pd.to_datetime(test_preds['date'])
        plt.plot(dates, test_preds['ewma_neural_weight'], label='EWMA')
        plt.plot(dates, test_preds['garch_neural_weight'], label='GARCH')
        plt.plot(dates, test_preds['egarch_neural_weight'], label='EGARCH')
        plt.plot(dates, test_preds['gjr_garch_neural_weight'], label='GJR-GARCH')
        plt.legend()
        plt.title(f'{asset} Neural Weights (Test Period)')
        plt.savefig(f'data/neural_gating/plots/{asset}_neural_weights.png')
        plt.close()
        
    final_preds_df = pd.concat(all_preds, ignore_index=True)
    final_preds_df.to_csv('data/neural_gating/neural_gate_predictions.csv', index=False)
    
    with open('data/neural_gating/all_fallbacks.json', 'w') as f:
        json.dump(all_fallbacks, f)
        
    pd.DataFrame(asset_summaries).to_csv('data/neural_gating/asset_summaries.csv', index=False)
    
if __name__ == '__main__':
    run_all()

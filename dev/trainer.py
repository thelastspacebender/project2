import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import copy
import dashboard

def build_net_from_layers(input_dim, layers, dropout_p):
    modules = []
    current_dim = input_dim
    for hidden_dim in layers:
        modules.append(nn.Linear(current_dim, hidden_dim, bias=True))
        modules.append(nn.ReLU())
        if dropout_p > 0:
            modules.append(nn.Dropout(dropout_p))
        current_dim = hidden_dim
    modules.append(nn.Linear(current_dim, 1, bias=True))
    return nn.Sequential(*modules)

class ReceptorNet(nn.Module):
    def __init__(self, input_dim, layers, n_receptors, dropout_p):
        super().__init__()
        self.receptors = nn.ModuleList([
            build_net_from_layers(input_dim, layers, dropout_p) 
            for _ in range(n_receptors)
        ])
    def forward(self, x):
        signals = torch.cat([r(x) for r in self.receptors], dim=1)
        return signals.sum(dim=1, keepdim=True), signals

def l1_reg(model):
    l1 = torch.tensor(0.0)
    for name, p in model.named_parameters():
        if 'bias' not in name:
            l1 = l1 + p.abs().sum()
    return l1

def run_simulation(config):
    print("Loading data from", config['data_path'])
    df = pd.read_csv(config['data_path'])
    target_col = 'Preferred_Imax' if 'Preferred_Imax' in df.columns else 'Imax'
    y = df[target_col].values.astype(np.float32)

    drop_cols = ['SMILES','Imax','Preferred_Imax','name','cid','CID','Canonical_Name',
                 'intensity_class','class','label','target','cluster','Chemical_Group',
                 'Has_Sulfur','Has_Nitrogen_Ring','Is_Amino_Acid']
    X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
    bad = (X.isnull().mean(axis=1) + (X==0).mean(axis=1)) > 0.95
    X, y = X[~bad], y[~bad]
    X = X.fillna(0).loc[:, X.nunique() > 1]
    
    print("Selecting features with Random Forest...")
    rf = RandomForestRegressor(n_estimators=config['rf_estimators'], random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(config['rf_top_features']).index
    X = X[top_features]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    kf = ShuffleSplit(n_splits=config['n_folds'], test_size=config['val_size'], random_state=42)
    
    fig, axes = dashboard.init_live_plot(config['n_folds'])
    fold_r2_history = []
    oof_preds = np.zeros(len(X))
    oof_counts = np.zeros(len(X))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X).to(device)
    y_t = torch.tensor(y).to(device)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        model = ReceptorNet(X.shape[1], config['layers'], config['n_receptors'], config['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        mse_fn = nn.MSELoss()
        
        history = {'train_loss':[], 'val_loss':[], 'train_r2':[], 'val_r2':[], 'receptor_signals':[]}
        best_val_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        patience_cnt = 0
        
        for epoch in range(1, config['n_epochs'] + 1):
            X_tr, y_tr = X_t[train_idx], y_t[train_idx].unsqueeze(1)
            X_v, y_v = X_t[val_idx], y_t[val_idx].unsqueeze(1)
            
            model.train()
            optimizer.zero_grad()
            y_pred_tr, _ = model(X_tr)
            loss = mse_fn(y_pred_tr, y_tr) + config['l1_lambda'] * l1_reg(model)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                y_pred_v, sigs_v = model(X_v)
                val_loss = mse_fn(y_pred_v, y_v).item()
            
            mse_tr = mse_fn(y_pred_tr, y_tr).item()
            history['train_loss'].append(mse_tr)
            history['val_loss'].append(val_loss)
            history['train_r2'].append(r2_score(y_tr.cpu().numpy(), y_pred_tr.detach().cpu().numpy()))
            history['val_r2'].append(r2_score(y_v.cpu().numpy(), y_pred_v.cpu().numpy()))
            history['receptor_signals'].append(sigs_v.mean(dim=0).cpu().numpy())
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                
            if epoch % 25 == 0 or epoch == 1 or patience_cnt >= config['patience']:
                dashboard.update_live_plot(fig, axes, history, model, X_t, y_t, val_idx, epoch, best_val_loss, patience_cnt, fold, config['n_folds'], fold_r2_history, config['n_epochs'], config['patience'], config['n_receptors'])
                if patience_cnt >= config['patience']:
                    break
        
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            y_pred_f, _ = model(X_t)
        
        y_pred_np = y_pred_f.cpu().numpy().flatten()
        val_r2 = r2_score(y[val_idx], y_pred_np[val_idx])
        fold_r2_history.append(val_r2)
        oof_preds[val_idx] += y_pred_np[val_idx]
        oof_counts[val_idx] += 1
        dashboard.update_live_plot(fig, axes, history, model, X_t, y_t, val_idx, epoch, best_val_loss, patience_cnt, fold, config['n_folds'], fold_r2_history, config['n_epochs'], config['patience'], config['n_receptors'])

    valid_oof = oof_counts > 0
    oof_final = oof_preds[valid_oof] / oof_counts[valid_oof]
    oof_r2 = r2_score(y[valid_oof], oof_final)
    
    print(f"Simulation Finished. True OOF Val R2: {oof_r2:.4f}")
    dashboard.finalize_plot(fig, axes, y[valid_oof], oof_final, oof_r2)

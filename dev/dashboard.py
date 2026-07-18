import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error
import os

BG   = '#0d1117'
CARD = '#161b22'
BLUE = '#58a6ff'
RED  = '#f78166'
GRN  = '#3fb950'
YEL  = '#d29922'
PURP = '#8957e5'
GREY = '#6e7681'
TEXT = '#c9d1d9'
SUBTEXT = '#8b949e'

def init_live_plot(n_folds):
    plt.ion()  
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(f'ReceptorNet - {n_folds}-Fold Ensemble Training', color='white', fontsize=16, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)
    axes = {}
    axes['loss']      = fig.add_subplot(gs[0, 0:2])
    axes['r2']        = fig.add_subplot(gs[0, 2:4])
    axes['stats']     = fig.add_subplot(gs[0:2, 4])
    axes['scatter']   = fig.add_subplot(gs[1, 0:2])
    axes['residuals'] = fig.add_subplot(gs[1, 2:4])
    axes['heatmap']   = fig.add_subplot(gs[2, 0:3])
    axes['ensemble']  = fig.add_subplot(gs[2, 3:5])
    
    for ax in axes.values():
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.tick_params(colors=SUBTEXT, labelsize=8)
    
    fig.canvas.manager.set_window_title('Ensemble Dashboard')
    plt.pause(0.1)
    return fig, axes


def clr(ax, title):
    ax.cla()
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, pad=4)


def update_live_plot(fig, axes, history, model, X_t, y_t, val_idx, epoch, best_val_loss, patience_cnt, fold, n_folds, fold_r2_history, n_epochs, patience, n_receptors):
    import torch
    model.eval()
    with torch.no_grad():
        y_pred_all, sigs_all = model(X_t)
        y_pred_np  = y_pred_all.cpu().numpy().flatten()
        sigs_np    = sigs_all.cpu().numpy()
    
    y_true = y_t.cpu().numpy()
    epochs_arr = np.arange(1, len(history['train_loss']) + 1)
    
    y_true_val = y_true[val_idx]
    y_pred_val = y_pred_np[val_idx]
    r2_v_curr  = r2_score(y_true_val, y_pred_val)

    final_sigs = sigs_np.mean(axis=0)
    n_active   = int((final_sigs > 0.01).sum())
    n_dead     = n_receptors - n_active
    r2_f       = r2_score(y_true, y_pred_np)
    rmse_f     = np.sqrt(mean_squared_error(y_true, y_pred_np))
    res        = y_true - y_pred_np

    # --- Loss ---
    clr(axes['loss'], f'[Fold {fold}/{n_folds}] Train / Val Loss')
    axes['loss'].plot(epochs_arr, history['train_loss'], color=BLUE, lw=1.2, label='Train')
    axes['loss'].plot(epochs_arr, history['val_loss'],   color=RED,  lw=1.2, label='Val')
    axes['loss'].legend(facecolor=CARD, labelcolor=TEXT, fontsize=7)

    # --- R2 ---
    clr(axes['r2'], f'[Fold {fold}/{n_folds}] Train / Val R^2')
    axes['r2'].plot(epochs_arr, history['train_r2'], color=GRN, lw=1.2, label='Train')
    axes['r2'].plot(epochs_arr, history['val_r2'],   color=YEL, lw=1.2, label='Val')
    axes['r2'].axhline(0, color=GREY, lw=0.8, ls='--')
    axes['r2'].legend(facecolor=CARD, labelcolor=TEXT, fontsize=7)

    # --- Scatter ---
    clr(axes['scatter'], f'All Data Scatter | Curr Val R2={r2_v_curr:.3f}')
    axes['scatter'].scatter(y_true, y_pred_np, alpha=0.3, s=12, color=BLUE, edgecolors='none', label='Train')
    axes['scatter'].scatter(y_true_val, y_pred_val, alpha=0.9, s=20, color=RED, edgecolors='white', linewidths=0.5, label='Val')
    lo = min(y_true.min(), y_pred_np.min()) - 3
    hi = max(y_true.max(), y_pred_np.max()) + 3
    axes['scatter'].plot([lo, hi], [lo, hi], 'w--', lw=1, alpha=0.5)

    # --- Stats ---
    clr(axes['stats'], f'Fold {fold} Statistics')
    best_val_r2 = max(history['val_r2']) if history['val_r2'] else 0
    txt = (
        f"FOLD:           {fold}/{n_folds}\n"
        f"Epoch:          {epoch}/{n_epochs}\n"
        f"Best Val Loss:  {best_val_loss:.3f}\n"
        f"Train R2:       {history['train_r2'][-1]:.3f}\n"
        f"Val R2 (live):  {history['val_r2'][-1]:.3f}\n"
        f"Best Val R2:    {best_val_r2:.3f}\n"
        f"\n"
        f"Active Receptors: {n_active}/{n_receptors}\n"
        f"Silenced (L1):    {n_dead}\n"
        f"Patience: {patience_cnt}/{patience}\n"
    )
    axes['stats'].text(0.05, 0.95, txt, transform=axes['stats'].transAxes, color=TEXT, fontsize=9, va='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=CARD, edgecolor='#30363d', alpha=0.9))
    axes['stats'].axis('off')

    # --- Heatmap ---
    clr(axes['heatmap'], 'Receptor Activity Heatmap')
    if len(history['receptor_signals']) > 0:
        rec_matrix = np.array(history['receptor_signals']).T
        axes['heatmap'].imshow(rec_matrix, aspect='auto', cmap='hot', extent=[1, len(epochs_arr), n_receptors + 0.5, 0.5], interpolation='nearest')

    # --- Residuals ---
    clr(axes['residuals'], f'Residuals\nMean={res.mean():.2f}  Std={res.std():.2f}')
    axes['residuals'].hist(res, bins=30, color=PURP, alpha=0.85, edgecolor='none')
    axes['residuals'].axvline(0, color=RED, lw=1.5, ls='--')
    
    # --- Ensemble Bar Chart ---
    clr(axes['ensemble'], 'Completed Folds Final Val R2')
    if len(fold_r2_history) > 0:
        indices = np.arange(1, len(fold_r2_history) + 1)
        axes['ensemble'].bar(indices, fold_r2_history, color=BLUE, alpha=0.8)
        axes['ensemble'].axhline(0.75, color=GRN, lw=1.5, ls='--', label='Target 0.75')
        axes['ensemble'].set_ylim(0, max(1.0, max(fold_r2_history) + 0.1))
        axes['ensemble'].set_xticks(indices)

    fig.canvas.draw()
    fig.canvas.flush_events()


def finalize_plot(fig, axes, y_true, oof_preds, oof_r2):
    clr(axes['scatter'], f'FINAL ENSEMBLE OOF\nTrue R2 = {oof_r2:.3f}')
    axes['scatter'].scatter(y_true, oof_preds, alpha=0.7, color=GRN, edgecolors='white', linewidths=0.5)
    lo = min(y_true.min(), oof_preds.min()) - 3
    hi = max(y_true.max(), oof_preds.max()) + 3
    axes['scatter'].plot([lo, hi], [lo, hi], 'w--', lw=1, alpha=0.5)
    
    fig.canvas.draw()
    os.makedirs('results', exist_ok=True)
    fig.savefig('results/ensemble_final.png', dpi=110, bbox_inches='tight', facecolor=BG)
    plt.ioff()
    plt.show(block=True)

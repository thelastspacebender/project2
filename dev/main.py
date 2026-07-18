import tkinter as tk
from tkinter import ttk, filedialog
import os
import threading
from network_builder import NetworkVisualizer
import trainer

BG = '#0d1117'
FG = '#c9d1d9'
CARD = '#161b22'
BLUE = '#58a6ff'

class ReceptorSimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Receptor Simulator - Configurator")
        self.geometry("900x600")
        self.configure(bg=BG)
        
        # UI Styles
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure('TButton', background='#21262d', foreground=FG, font=("Segoe UI", 10, "bold"), borderwidth=1, bordercolor='#30363d')
        style.map('TButton', background=[('active', '#30363d')])
        style.configure('TEntry', fieldbackground=CARD, foreground=FG, insertcolor=FG, borderwidth=1)
        style.configure('Header.TLabel', font=("Segoe UI", 12, "bold"), foreground=BLUE)

        # Main Layout
        self.left_panel = ttk.Frame(self, padding=20)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_panel = ttk.Frame(self)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.build_left_panel()
        self.build_right_panel()

    def build_left_panel(self):
        # Data Section
        ttk.Label(self.left_panel, text="DATA SETTINGS", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.data_path_var = tk.StringVar(value=r"..\data\waka_dragon_merged.csv")
        data_frame = ttk.Frame(self.left_panel)
        data_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Entry(data_frame, textvariable=self.data_path_var, width=30).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(data_frame, text="Browse", command=self.browse_data, width=8).pack(side=tk.LEFT)

        # RF Settings
        ttk.Label(self.left_panel, text="RANDOM FOREST (FEATURE SELECTION)", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.rf_est = self.add_param("Trees (n_estimators):", "100")
        self.rf_top = self.add_param("Top Features:", "100")
        ttk.Label(self.left_panel, text="").pack(pady=2)

        # Network Settings
        ttk.Label(self.left_panel, text="NETWORK HYPERPARAMETERS", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.n_receptors = self.add_param("N Receptors (Ensemble):", "14")
        self.dropout = self.add_param("Dropout:", "0.2")
        self.l1_lambda = self.add_param("L1 Regularization:", "0.0001")
        self.weight_decay = self.add_param("Weight Decay (L2):", "0.0001")
        self.lr = self.add_param("Learning Rate:", "0.001")
        ttk.Label(self.left_panel, text="").pack(pady=2)

        # Training Settings
        ttk.Label(self.left_panel, text="TRAINING PIPELINE", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.n_folds = self.add_param("N Folds (ShuffleSplit):", "30")
        self.val_size = self.add_param("Validation Size:", "50")
        self.n_epochs = self.add_param("Max Epochs:", "3000")
        self.patience = self.add_param("Early Stopping Patience:", "300")
        
        # Start Button
        btn_frame = tk.Frame(self.left_panel, bg=BG)
        btn_frame.pack(fill=tk.X, pady=20)
        start_btn = tk.Button(btn_frame, text="🚀 LAUNCH SIMULATION", bg='#238636', fg='white', font=("Segoe UI", 12, "bold"), bd=0, padx=10, pady=10, cursor="hand2", command=self.start_simulation)
        start_btn.pack(fill=tk.X)

    def build_right_panel(self):
        title = tk.Label(self.right_panel, text="NEURAL NETWORK ARCHITECTURE BUILDER", bg=BG, fg=FG, font=("Segoe UI", 14, "bold"))
        title.pack(pady=20)
        
        self.visualizer = NetworkVisualizer(self.right_panel)
        self.visualizer.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def add_param(self, label_text, default_val):
        f = ttk.Frame(self.left_panel)
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label_text, width=22).pack(side=tk.LEFT)
        var = tk.StringVar(value=default_val)
        ttk.Entry(f, textvariable=var, width=10).pack(side=tk.RIGHT)
        return var

    def browse_data(self):
        filename = filedialog.askopenfilename(initialdir="..\\data", title="Select Dataset", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if filename:
            self.data_path_var.set(filename)

    def start_simulation(self):
        # Build Config Dictionary
        config = {
            'data_path': self.data_path_var.get(),
            'rf_estimators': int(self.rf_est.get()),
            'rf_top_features': int(self.rf_top.get()),
            'n_receptors': int(self.n_receptors.get()),
            'dropout': float(self.dropout.get()),
            'l1_lambda': float(self.l1_lambda.get()),
            'weight_decay': float(self.weight_decay.get()),
            'lr': float(self.lr.get()),
            'n_folds': int(self.n_folds.get()),
            'val_size': int(self.val_size.get()),
            'n_epochs': int(self.n_epochs.get()),
            'patience': int(self.patience.get()),
            'layers': self.visualizer.get_layers()
        }
        
        self.withdraw() # Hide configuration window
        
        # Run simulation in main thread because Matplotlib requires it
        try:
            trainer.run_simulation(config)
        except Exception as e:
            import traceback
            from tkinter import messagebox
            error_trace = traceback.format_exc()
            print(f"Error during simulation: {e}")
            messagebox.showerror("Simulation Crash", f"An error occurred:\n{e}\n\nTraceback:\n{error_trace}")
        finally:
            self.deiconify() # Show window again when done

if __name__ == "__main__":
    app = ReceptorSimulatorApp()
    app.mainloop()

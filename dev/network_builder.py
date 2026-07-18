import tkinter as tk
import numpy as np

class NetworkVisualizer(tk.Frame):
    def __init__(self, master, bg_color='#0d1117', fg_color='#c9d1d9', update_callback=None, **kwargs):
        super().__init__(master, bg=bg_color, **kwargs)
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.update_callback = update_callback
        
        self.canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.layers = [32, 32, 16, 8] 
        self.bind("<Configure>", self.draw_network)
        
    def add_layer(self, index):
        self.layers.insert(index, 16)
        self.draw_network()
        if self.update_callback: self.update_callback()
        
    def remove_layer(self, index):
        if len(self.layers) > 1:
            self.layers.pop(index)
            self.draw_network()
            if self.update_callback: self.update_callback()
            
    def modify_neurons(self, index, delta):
        if self.layers[index] + delta > 0:
            self.layers[index] += delta
            self.draw_network()
            if self.update_callback: self.update_callback()

    def get_layers(self):
        return self.layers
        
    def draw_network(self, event=None):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 10 or height <= 10:
            return
            
        all_layers = ["Input\n(Features)"] + [str(l) for l in self.layers] + ["Output\n(1)"]
        layer_x = np.linspace(60, width - 60, len(all_layers))
        node_coords = []
        
        for i, (l_name, x) in enumerate(zip(all_layers, layer_x)):
            self.canvas.create_text(x, 30, text=str(l_name), fill=self.fg_color, font=("Segoe UI", 10, "bold"), justify="center")
            
            n_val = int(l_name) if l_name.isdigit() else (5 if i==0 else 1)
            n_draw = min(n_val, 6)
                
            layer_coords = []
            y_start = 80
            y_end = height - 60
            if n_draw == 1:
                y_space = [(y_start + y_end) / 2]
            else:
                y_space = np.linspace(y_start, y_end, n_draw)
            
            for j, y in enumerate(y_space):
                if n_draw > 4 and j == n_draw // 2 and i != len(all_layers)-1:
                    self.canvas.create_text(x, y, text="...", fill=self.fg_color, font=("Segoe UI", 16, "bold"))
                else:
                    r = 8
                    color = '#3fb950' if i==0 else ('#f78166' if i==len(all_layers)-1 else '#58a6ff')
                    self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=color, tags="node")
                    layer_coords.append((x, y))
            
            node_coords.append(layer_coords)
            
            if i > 0 and i < len(all_layers) - 1:
                btn_y = height - 25
                idx = i - 1
                
                self.canvas.create_rectangle(x-35, btn_y-12, x-15, btn_y+12, fill='#21262d', outline='#30363d', tags=f"minus_{idx}")
                self.canvas.create_text(x-25, btn_y, text="-", fill='white', font=("Segoe UI", 12), tags=f"minus_{idx}")
                self.canvas.tag_bind(f"minus_{idx}", "<Button-1>", lambda e, idx=idx: self.modify_neurons(idx, -1))
                self.canvas.tag_bind(f"minus_{idx}", "<Button-3>", lambda e, idx=idx: self.modify_neurons(idx, -8)) # Right click for fast
                
                self.canvas.create_rectangle(x+15, btn_y-12, x+35, btn_y+12, fill='#21262d', outline='#30363d', tags=f"plus_{idx}")
                self.canvas.create_text(x+25, btn_y, text="+", fill='white', font=("Segoe UI", 12), tags=f"plus_{idx}")
                self.canvas.tag_bind(f"plus_{idx}", "<Button-1>", lambda e, idx=idx: self.modify_neurons(idx, 1))
                self.canvas.tag_bind(f"plus_{idx}", "<Button-3>", lambda e, idx=idx: self.modify_neurons(idx, 8))
                
                self.canvas.create_text(x, btn_y, text=str(self.layers[idx]), fill=self.fg_color, font=("Segoe UI", 10, "bold"))
            
            if i < len(all_layers) - 1:
                mid_x = (x + layer_x[i+1]) / 2
                self.canvas.create_text(mid_x, 30, text="+ Layer", fill='#3fb950', font=("Segoe UI", 9, "underline"), tags=f"add_{i}")
                self.canvas.tag_bind(f"add_{i}", "<Button-1>", lambda e, idx=i: self.add_layer(idx))
                
            if i > 0 and i < len(all_layers) - 1 and len(self.layers) > 1:
                self.canvas.create_text(x, 50, text="×", fill='#f78166', font=("Segoe UI", 14, "bold"), tags=f"rem_{i-1}")
                self.canvas.tag_bind(f"rem_{i-1}", "<Button-1>", lambda e, idx=i-1: self.remove_layer(idx))
                
        for i in range(len(node_coords)-1):
            for (x1, y1) in node_coords[i]:
                for (x2, y2) in node_coords[i+1]:
                    self.canvas.create_line(x1, y1, x2, y2, fill='#30363d', width=1.5, tags="line")
        
        self.canvas.tag_lower("line")

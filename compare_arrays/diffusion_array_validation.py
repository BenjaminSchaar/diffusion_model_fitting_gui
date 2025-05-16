
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

arr1 = None
arr2 = None

def load_file(label, arr_index):
    global arr1, arr2
    file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
    if file_path:
        try:
            array = np.load(file_path)
            if array.ndim != 2:
                raise ValueError("Only 2D arrays are supported.")
            label.config(text=f"Loaded: {file_path.split('/')[-1]} (shape={array.shape})")
            if arr_index == 1:
                arr1 = array
            else:
                arr2 = array
            update_plot(slider.get())
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

def update_plot(index):
    global arr1, arr2
    if arr1 is None or arr2 is None:
        return
    if arr1.shape[0] != arr2.shape[0] or arr1.shape[1] != arr2.shape[1]:
        messagebox.showerror("Shape Mismatch", "Arrays must have the same shape.")
        return

    ax.clear()
    ax.plot(arr1[index], label='Array 1')
    ax.plot(arr2[index], label='Array 2')
    ax.set_title(f"Index {index}")
    ax.set_xlabel("Distance Index")
    ax.set_ylabel("Value")
    ax.legend()
    canvas.draw()

# GUI setup
root = tk.Tk()
root.title("2D Array Comparison Slider")

frame = tk.Frame(root)
frame.pack()

load1_btn = tk.Button(frame, text="Load Array 1", command=lambda: load_file(label1, 1))
load1_btn.grid(row=0, column=0)
label1 = tk.Label(frame, text="No file loaded")
label1.grid(row=0, column=1)

load2_btn = tk.Button(frame, text="Load Array 2", command=lambda: load_file(label2, 2))
load2_btn.grid(row=1, column=0)
label2 = tk.Label(frame, text="No file loaded")
label2.grid(row=1, column=1)

slider = tk.Scale(frame, from_=0, to=10799, orient=tk.HORIZONTAL, label="Time Index", command=lambda val: update_plot(int(val)))
slider.grid(row=2, column=0, columnspan=2, sticky="ew")

fig, ax = plt.subplots(figsize=(10, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()

from config import *
import tkinter as tk
from tkinter import ttk

window = tk.Tk()

window.title("Extracted Colors")
window.wm_geometry("1500x1000")

tree = ttk.Treeview(window, selectmode="browse", show="headings", columns=("thumbnail", "image-name", "color"))
tree.heading("thumbnail", text="Thumbnail", anchor="center")
tree.heading("image-name", text="Image Name")
tree.heading("color", text="Colour")
tree.pack(fill="both")

scrollbar = ttk.Scrollbar(window, command=tree.yview)
scrollbar.pack(side="right", fill="x")
tree.configure(yscrollcommand=scrollbar.set)

window.mainloop()

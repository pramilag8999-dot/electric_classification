# check_image.py
import os
from tkinter import Tk, filedialog, messagebox

# List of all images in train folder
train_images = []
for root, dirs, files in os.walk('train'):
    for file in files:
        if file.lower().endswith(('png','jpg','jpeg')):
            train_images.append(os.path.abspath(os.path.join(root,file)))

def is_image_in_train(img_path):
    return os.path.abspath(img_path) in train_images

# GUI for selecting random image
root = Tk()
root.withdraw()  # hide main window

file_path = filedialog.askopenfilename(title="Select Random Image")
if file_path:
    if is_image_in_train(file_path):
        messagebox.showinfo("Warning", "This image is already in train folder!")
    else:
        messagebox.showinfo("Info", "This image can be used!")

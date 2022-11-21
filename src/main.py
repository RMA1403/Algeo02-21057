import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from glob import glob
import asyncio
import threading
import functions.FaceRecognition as FaceRecon

dataset = ""
test_image = ""
dataset_weights = None
new_img = None

async_loop = asyncio.new_event_loop()

def input_dataset(event):
  global dataset
  new_dataset = filedialog.askdirectory(title="Select a dataset folder")
  if new_dataset != "":
    dataset = new_dataset

def start_train_dataset(event):
  global async_loop
  thread_target = lambda: async_loop.run_until_complete(train_dataset(dataset))
  threading.Thread(target=thread_target).start()

async def train_dataset(dataset):
  global dataset_weights
  dataset_weights = await FaceRecon.train_dataset(dataset)

def input_image(event):
  global test_image
  global new_img

  new_test_image = filedialog.askopenfilename(title="Select an image")
  if new_test_image != "":
    test_image = glob(new_test_image)[0]

    main_canvas.delete("input_img_image")
    new_img = ImageTk.PhotoImage(Image.open(new_test_image).resize((256, 256)))
    main_canvas.create_image(625, 150, image=new_img, tags="input_img_image2")

def start_recognition(event):
  global test_image
  global dataset

  print(dataset_weights)


window = tk.Tk(className=" Face Recognition by Never Tsurrender")
window.geometry("800x600")

main_canvas = tk.Canvas(window, highlightthickness=0, width=800, height=600)
main_canvas.pack(expand=True, fill="both")

bg_img = ImageTk.PhotoImage(Image.open("./src/assets/images/background.png"))
main_canvas.create_image(0, 0, anchor="nw", image=bg_img)

input_dataset_img = ImageTk.PhotoImage(Image.open("./src/assets/images/input_dataset.png"))
main_canvas.create_image(175, 155, image=input_dataset_img, tags="input_dataset_btn")
main_canvas.tag_bind("input_dataset_btn", '<Button-1>', input_dataset)

train_dataset_img = ImageTk.PhotoImage(Image.open("./src/assets/images/train_dataset.png"))
main_canvas.create_image(355, 155, image=train_dataset_img, tags="train_dataset_btn")
main_canvas.tag_bind("train_dataset_btn", '<Button-1>', start_train_dataset)

test_image_img = ImageTk.PhotoImage(Image.open("./src/assets/images/test_image.png"))
main_canvas.create_image(175, 300, image=test_image_img, tags="test_image_btn")
main_canvas.tag_bind("test_image_btn", '<Button-1>', input_image)

start_recognition_img = ImageTk.PhotoImage(Image.open("./src/assets/images/start_recognition.png"))
main_canvas.create_image(355, 300, image=start_recognition_img, tags="start_recognition_btn")
main_canvas.tag_bind("start_recognition_btn", '<Button-1>', start_recognition)

input_img = ImageTk.PhotoImage(Image.open("./src/assets/images/placeholder.jpg").resize((256, 256)))
main_canvas.create_image(625, 150, image=input_img, tags="input_img_image")

output_img = ImageTk.PhotoImage(Image.open("./src/assets/images/placeholder.jpg").resize((256, 256)))
main_canvas.create_image(625, 440, image=output_img, tags="output_img_image")

window.mainloop()
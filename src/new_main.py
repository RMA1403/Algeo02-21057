import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from glob import glob
import asyncio
import threading
import new_testing as nt

dataset = ""
test_image = ""
new_img = None

def _asyncio_thread(async_loop):
    async_loop.run_until_complete(nt.train_dataset(dataset))


def do_tasks():
    threading.Thread(target=_asyncio_thread, args=(async_loop,)).start()

def input_dataset(event):
  global dataset

  new_dataset = filedialog.askdirectory(title="Select a dataset folder")
  if new_dataset != "":
    dataset = new_dataset

def train_dataset(event):
  do_tasks()


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

  print(test_image, dataset)


window = tk.Tk(className=" Face Recognition by Never Tsurrender")
window.geometry("800x600")

main_canvas = tk.Canvas(window, highlightthickness=0, width=800, height=600)
main_canvas.pack(expand=True, fill="both")

bg_img = ImageTk.PhotoImage(Image.open("./src/background.png"))
main_canvas.create_image(0, 0, anchor="nw", image=bg_img)

input_dataset_img = ImageTk.PhotoImage(Image.open("./src/input_dataset.png"))
main_canvas.create_image(175, 155, image=input_dataset_img, tags="input_dataset_btn")
main_canvas.tag_bind("input_dataset_btn", '<Button-1>', input_dataset)

train_dataset_img = ImageTk.PhotoImage(Image.open("./src/train_dataset.png"))
main_canvas.create_image(355, 155, image=train_dataset_img, tags="train_dataset_btn")
main_canvas.tag_bind("train_dataset_btn", '<Button-1>', train_dataset)

test_image_img = ImageTk.PhotoImage(Image.open("./src/test_image.png"))
main_canvas.create_image(175, 300, image=test_image_img, tags="test_image_btn")
main_canvas.tag_bind("test_image_btn", '<Button-1>', input_image)

start_recognition_img = ImageTk.PhotoImage(Image.open("./src/start_recognition.png"))
main_canvas.create_image(355, 300, image=start_recognition_img, tags="start_recognition_btn")
main_canvas.tag_bind("start_recognition_btn", '<Button-1>', start_recognition)

input_img = ImageTk.PhotoImage(Image.open("./src/placeholder.jpg").resize((256, 256)))
main_canvas.create_image(625, 150, image=input_img, tags="input_img_image")

output_img = ImageTk.PhotoImage(Image.open("./src/placeholder.jpg").resize((256, 256)))
main_canvas.create_image(625, 440, image=output_img, tags="output_img_image")

async_loop = asyncio.new_event_loop()
window.mainloop()
import tkinter as tk
import asyncio
import threading
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
from glob import glob

import functions.FaceRecognition as FaceRecon


dataset = ""
test_image = ""
output_image = ""
curr_name = ""
dataset_weights = None
eigenfaces = None
mean_face = None
temp1 = None
temp2 = None
temp3 = None
temp4 = None
temp5 = None
temp6 = None
temp7 = None
temp8 = None
webcam = None
cvimage = None
isCameraOpen = False

def _asyncio_thread_train(async_loop):
  async_loop.run_until_complete(train_dataset(dataset))

def _asyncio_thread_match(async_loop):
  async_loop.run_until_complete(match_image(test_image, dataset_weights, eigenfaces, mean_face, dataset))

def _asyncio_thread_match_webcam(async_loop):
  async_loop.run_until_complete(match_image_webcam(cvimage, dataset_weights, eigenfaces, mean_face, dataset))

def get_name(filename):
  return filename[filename.rfind('/')+1:len(filename)-8]

def input_dataset(event):
  global dataset
  new_dataset = filedialog.askdirectory(title="Select a dataset folder")
  if new_dataset != "":
    dataset = new_dataset

def start_train_dataset(event):
  global async_loop
  main_canvas.create_text(265, 220, fill="#00C738", font="Inter 32 bold", text="Processing", tags="process_label")
  threading.Thread(target=_asyncio_thread_train, args=(async_loop,)).start()

async def train_dataset(dataset):
  global dataset_weights
  global eigenfaces
  global mean_face
  dataset_weights, eigenfaces, mean_face, train_time = await FaceRecon.train_dataset(dataset)
  
  main_canvas.delete("process_label")

  main_canvas.delete("train_time")
  text = "Dataset training time : " + str(round(train_time, 3)) + " second"
  main_canvas.create_text(245, 450, fill="#00C738", font="Inter 11", text=text, tags="train_time")


def input_image(event):
  global test_image
  global temp1
  new_test_image = filedialog.askopenfilename(title="Select an image")
  if new_test_image != "":
    test_image = glob(new_test_image)[0]

    main_canvas.delete("input_img_image")
    temp1 = ImageTk.PhotoImage(Image.open(new_test_image).resize((256, 256)))
    main_canvas.create_image(625, 150, image=temp1, tags="input_img_image")

def start_recognition(event):
  global async_loop
  threading.Thread(target=_asyncio_thread_match, args=(async_loop,)).start()

async def match_image(test_image, dataset_weights, eigenfaces, mean_face, dataset):
  global temp2
  global output_image
  global curr_name
  euc_min, output_image, match_time = await FaceRecon.match_image(test_image, dataset_weights, eigenfaces, mean_face, dataset)

  main_canvas.delete("output_img_image")
  temp2 = ImageTk.PhotoImage(Image.open(output_image).resize((256, 256)))
  main_canvas.create_image(625, 440, image=temp2, tags="output_img_image")

  main_canvas.delete("match_time")
  text = "Image recognition time : " + str(round(match_time, 3)) + " second"
  main_canvas.create_text(250, 475, fill="#00C738", font="Inter 11", text=text, tags="match_time")

  main_canvas.delete("euc_dis")
  text = "Euclidian distance : " + str(round(euc_min, 3))
  main_canvas.create_text(223, 500, fill="#00C738", font="Inter 11", text=text, tags="euc_dis")
  
  main_canvas.delete("name")
  text = get_name(output_image)
  curr_name = text
  main_canvas.create_text(265, 80, fill="white", font="Inter 32 bold", text=text, tags="name")

def show_webcam_frame():
  global webcam
  global async_loop
  global cvimage
  global isCameraOpen
  cvimage = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
  show_cvimage = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
  if isCameraOpen:
    threading.Thread(target=_asyncio_thread_match_webcam, args=(async_loop,)).start()
  if webcam != None:
    img = Image.fromarray(show_cvimage)
    imgtk = ImageTk.PhotoImage(image=img)
    webcam.imgtk = imgtk
    webcam.configure(image=imgtk)
    webcam.after(30, show_webcam_frame)

async def match_image_webcam(cvimage, dataset_weights, eigenfaces, mean_face, dataset):
  global temp3
  global output_image
  global curr_name
  euc_min, new_output_image, match_time = await FaceRecon.match_image_webcam(cvimage, dataset_weights, eigenfaces, mean_face, dataset)

  if new_output_image != output_image:
    output_image = new_output_image
    main_canvas.delete("output_img_image")
    temp3 = ImageTk.PhotoImage(Image.open(new_output_image).resize((256, 256)))
    main_canvas.create_image(625, 440, image=temp3, tags="output_img_image")

  main_canvas.delete("match_time")
  text = "Image recognition time : " + str(round(match_time, 3)) + " second"
  main_canvas.create_text(250, 475, fill="#00C738", font="Inter 11", text=text, tags="match_time")

  main_canvas.delete("euc_dis")
  text = "Euclidian distance : " + str(round(euc_min, 3))
  main_canvas.create_text(223, 500, fill="#00C738", font="Inter 11", text=text, tags="euc_dis")
  
  text = get_name(new_output_image)
  if text != curr_name:
    curr_name = text
    main_canvas.delete("name")
    main_canvas.create_text(265, 80, fill="white", font="Inter 32 bold", text=text, tags="name")

def open_camera(event):
  global webcam
  global temp4
  global temp8
  global isCameraOpen
  webcam = tk.Label(window, width=256, height=256)
  webcam.place(x=495, y=21)

  main_canvas.delete("open_camera_btn")
  temp4 = ImageTk.PhotoImage(Image.open("./src/assets/images/close_camera.png"))
  main_canvas.create_image(265, 20, image=temp4, tags="close_camera_btn")
  main_canvas.tag_bind("close_camera_btn", '<Button-1>', close_camera)

  main_canvas.delete("test_image_btn")
  main_canvas.delete("start_recognition_btn")

  temp8 = ImageTk.PhotoImage(Image.open("./src/assets/images/placeholder.jpg").resize((256, 256)))
  main_canvas.create_image(625, 150, image=input_img, tags="input_img_image")

  isCameraOpen = True
  show_webcam_frame()

def close_camera(event):
  global webcam
  global temp5
  global temp6
  global temp7
  global isCameraOpen
  global test_image
  webcam.destroy()

  main_canvas.delete("close_camera_btn")
  temp5 = ImageTk.PhotoImage(Image.open("./src/assets/images/open_camera.png"))
  main_canvas.create_image(265, 20, image=temp5, tags="open_camera_btn")
  main_canvas.tag_bind("open_camera_btn", '<Button-1>', open_camera)

  temp6 = ImageTk.PhotoImage(Image.open("./src/assets/images/test_image.png"))
  main_canvas.create_image(175, 300, image=temp6, tags="test_image_btn")
  main_canvas.tag_bind("test_image_btn", '<Button-1>', input_image)

  temp7 = ImageTk.PhotoImage(Image.open("./src/assets/images/start_recognition.png"))
  main_canvas.create_image(355, 300, image=temp7, tags="start_recognition_btn")
  main_canvas.tag_bind("start_recognition_btn", '<Button-1>', start_recognition)

  test_image = ""
  isCameraOpen = False

window = tk.Tk(className=" Face Recognition by Never Tsurrender")
window.geometry("800x600")

main_canvas = tk.Canvas(window, highlightthickness=0, width=800, height=600)
main_canvas.pack(expand=True, fill="both")

bg_img = ImageTk.PhotoImage(Image.open("./src/assets/images/background.png"))
main_canvas.create_image(0, 0, anchor="nw", image=bg_img)

open_camera_img = ImageTk.PhotoImage(Image.open("./src/assets/images/open_camera.png"))
main_canvas.create_image(265, 20, image=open_camera_img, tags="open_camera_btn")
main_canvas.tag_bind("open_camera_btn", '<Button-1>', open_camera)

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

main_canvas.create_text(265, 390, fill="#00C738", font="Inter 20 bold", text="", tags="name")
main_canvas.create_text(265, 410, fill="#00C738", font="Inter 20 bold", text="Results")
main_canvas.create_text(165, 450, fill="#00C738", font="Inter 12", text="", tags="train_time")
main_canvas.create_text(165, 470, fill="#00C738", font="Inter 12", text="", tags="match_time")
main_canvas.create_text(165, 490, fill="#00C738", font="Inter 12", text="", tags="euc_dis")

cap = cv2.VideoCapture(0)

async_loop = asyncio.new_event_loop()
window.mainloop()
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from glob import glob
import ImageLoader
import asyncio
import threading

filename = "./src/placeholder.jpg"
dataset = ""
result = "./src/placeholder.jpg"
res_img = None
result_image_frame = None
result_image = None
result_data_frame = None
euc_distance = None
exec_time_frame = None
exec_time = None


def _asyncio_thread(async_loop):
    async_loop.run_until_complete(loadImage(filename, dataset))


def do_tasks(async_loop):
    threading.Thread(target=_asyncio_thread, args=(async_loop,)).start()


async def loadImage(filename, dataset):
    global result
    global result_image
    global result_image_frame
    global res_img
    global result_data_frame
    global euc_distance
    global exec_time_frame
    global exec_time

    new_result = await ImageLoader.loadImage(filename, dataset)
    if new_result[0] != "":
        result = new_result[0]
        result_image.destroy()
        res_img = ImageTk.PhotoImage(Image.open(
            result).resize((256, 256)))
        result_image = tk.Canvas(
            result_image_frame, width=256, height=256, highlightthickness=0)
        result_image.create_image(0, 0, anchor='nw', image=res_img)
        result_image.configure(background='white')
        result_image.grid(row=1, column=0)

        if euc_distance != None:
            euc_distance.destroy()
        euc_distance = tk.Label(result_data_frame, text=(
            "Euclidian distance: " + str(new_result[1])), font=("Helvetica", 8))
        euc_distance.configure(background='white')
        euc_distance.grid(row=1, column=0)

        if exec_time != None:
            exec_time.destroy()
        exec_time = tk.Label(exec_time_frame, text=(
            "Execution time: " + str(new_result[2])), font=("Helvetica", 8))
        exec_time.configure(background='white')
        exec_time.grid(row=0, column=0)


def main(async_loop):
    global filename
    global dataset
    global result
    global result_image
    global result_image_frame
    global res_img
    global result_data_frame
    global exec_time_frame

    def addTestImage():
        """Fungsi untuk menambahkan test image"""
        nonlocal test_image
        nonlocal img
        global filename
        new_filename = filedialog.askopenfilename(title="Select an image")
        if new_filename != "":
            filename = new_filename
            test_image.destroy()
            img = ImageTk.PhotoImage(Image.open(
                filename).resize((256, 256)))
            test_image = tk.Canvas(
                test_image_frame, width=256, height=256, highlightthickness=0)
            test_image.create_image(0, 0, anchor='nw', image=img)
            test_image.configure(background=bg_color)
            test_image.grid(row=1, column=0)

    def addDataset():
        """Fungsi untuk menambahkan folder dataset"""
        global dataset
        dataset = filedialog.askdirectory(title="Select a dataset folder")

    def startAlgorithm():
        """Fungsi untuk memulai algoritma pengenalan wajah"""
        do_tasks(async_loop)

    # Konfigurasi warna
    bg_color = "white"

    window = tk.Tk(className=" Face Recognition by Never Tsurrender")
    window.configure(background=bg_color)

    main_frame = tk.Frame(window, padx=100, pady=30)
    main_frame.configure(background=bg_color)
    main_frame.pack()

    # Menampilkan header
    header_frame = tk.Frame(
        main_frame)
    header_frame.configure(background=bg_color)

    header_label_frame = tk.Frame(header_frame)
    header_label_frame.configure(background=bg_color)

    header_label = tk.Label(
        header_label_frame, text="Face Recognition",  font=("Helvetica", 36, "bold"))
    header_label.configure(background=bg_color)

    canvas_frame = tk.Frame(header_frame)
    canvas_frame.configure(background=bg_color)

    line = tk.Canvas(canvas_frame, width=900, height=20, highlightthickness=0)
    line.configure(background=bg_color)
    line.create_line(0, 17, 900, 17, width=3)
    line.pack(fill="both", expand=True)

    header_frame.grid(row=0, column=0, columnspan=3, pady=20)
    header_label_frame.grid(row=0, column=0)
    header_label.grid(row=0, column=0, padx=250)
    canvas_frame.grid(row=1, column=0)

    # Menampilkan input dataset
    dataset_input_frame = tk.Frame(
        main_frame, padx=10)
    dataset_input_frame.configure(background=bg_color)

    dataset_input_label = tk.Label(
        dataset_input_frame, text="Insert Your Dataset",  font=("Helvetica", 12))
    dataset_input_label.configure(background=bg_color)

    dataset_btn = tk.Button(
        dataset_input_frame, text="Choose dataset", command=addDataset)

    dataset_input_frame.grid(row=1, column=0)
    dataset_input_label.grid(row=0, column=0)
    dataset_btn.grid(row=1, column=0)

    # Menampilkan input test image
    image_input_frame = tk.Frame(
        main_frame, padx=10)
    image_input_frame.configure(background=bg_color)

    image_input_label = tk.Label(
        image_input_frame, text="Insert Your Image",  font=("Helvetica", 12))
    image_input_label.configure(background=bg_color)

    test_image_btn = tk.Button(
        image_input_frame, text="Choose Image", command=addTestImage)

    image_input_frame.grid(row=2, column=0)
    image_input_label.grid(row=0, column=0)
    test_image_btn.grid(row=1, column=0)

    # Menampilkan tombol start
    start_frame = tk.Frame(main_frame, padx=10)
    start_frame.configure(background=bg_color)

    start_btn = tk.Button(start_frame, text="START", command=startAlgorithm)

    start_frame.grid(row=3, column=0)
    start_btn.grid(row=0, column=0)

    # Menampilkan data hasil pengenalan wajah
    result_data_frame = tk.Frame(main_frame, padx=10)
    result_data_frame.configure(background=bg_color)

    result_label = tk.Label(
        result_data_frame, text="Result", font=("Helvetica", 12))
    result_label.configure(background=bg_color)

    result_data_frame.grid(row=4, column=0)
    result_label.grid(row=0, column=0)

    # Menampilkan test image
    test_image_frame = tk.Frame(main_frame, padx=10)
    test_image_frame.configure(background=bg_color)

    test_image_label = tk.Label(
        test_image_frame, text="Test Image", font=("Helvetica", 12))
    test_image_label.configure(background=bg_color)

    img = ImageTk.PhotoImage(Image.open(
        filename).resize((256, 256)))
    test_image = tk.Canvas(test_image_frame, width=256,
                           height=256, highlightthickness=0)
    test_image.create_image(0, 0, anchor='nw', image=img)
    test_image.configure(background=bg_color)

    test_image_frame.grid(sticky='E', row=1, column=1, rowspan=4)
    test_image_label.grid(sticky='W', row=0, column=0)
    test_image.grid(row=1, column=0)

    # Menampilkan gambar hasil
    result_image_frame = tk.Frame(main_frame, padx=10)
    result_image_frame.configure(background=bg_color)

    result_image_label = tk.Label(
        result_image_frame, text="Closest Result", font=("Helvetica", 12))
    result_image_label.configure(background=bg_color)

    res_img = ImageTk.PhotoImage(Image.open(
        result).resize((256, 256)))
    result_image = tk.Canvas(result_image_frame, width=256,
                             height=256, highlightthickness=0)
    result_image.create_image(0, 0, anchor='nw', image=res_img)
    result_image.configure(background=bg_color)

    result_image_frame.grid(sticky='E', row=1, column=2, rowspan=4)
    result_image_label.grid(sticky='W', row=0, column=0)
    result_image.grid(row=1, column=0)

    # Menampilkan waktu eksekusi algoritma
    exec_time_frame = tk.Frame(main_frame)
    exec_time_frame.configure(background=bg_color)

    exec_time_frame.grid(row=5, column=1)

    # Padding bottom
    padding_bottom = tk.Frame(main_frame, height=120)
    padding_bottom.configure(background=bg_color)
    padding_bottom.grid(row=6, column=0, columnspan=3)

    window.mainloop()


async_loop = asyncio.new_event_loop()
main(async_loop)

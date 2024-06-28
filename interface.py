import threading
import tkinter as tk
from tkinter import filedialog, Button
import numpy
import cv2
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Canvas, Scrollbar, BOTH, Label
import label_extr as ex
import model as mm
import model03 as m3
def open_new_window01():
    def select_directory():
        global directory
        directory = filedialog.askdirectory(initialdir="models")
        directory_label.config(text=f'model directory: {directory}')

    def select_directory1():
        global  directory1
        directory1 = filedialog.askdirectory(initialdir="dataset")
        directory_label1.config(text=f'dataset directory : {directory1}')
    def resume():


        """print( directory)
        print( directory1)"""
        m3.treat('dataset/data04')
        m = m3.get_mod("model-r")
        m3.resume_train(m, "model-r")
    fram = tk.Toplevel(root)
    fw = 600
    fh = 600
    fram.geometry(str(fw) + "x" + str(fh))
    fram.title('Drug label recognition ')
    print(fw)
    print(fh)
    fw0 = fw * 0.5
    fh0 = fh * 0.5
    frame1 = tk.Frame(fram,bg="white")
    frame1.place(x=0, y=0, height=fh0, width=fw0)
    button = tk.Button(frame1, text="Select Model ", command=select_directory)
    button.place(x=80, y=20, height=50, width=140)
    directory_label = tk.Label(frame1, text="No directory selected")
    directory_label.place(x=40, y=120, height=50, width=220)

    frame2 = tk.Frame(fram)
    frame2.place(x=fw0, y=0, height=fh0, width=fw0)
    button1 = tk.Button(frame2, text="Select Dataset", command=select_directory1)
    button1.place(x=80, y=20, height=50, width=140)
    directory_label1 = tk.Label(frame2, text="No directory selected")
    directory_label1.place(x=40, y=120, height=50, width=220)



    frame3 = tk.Frame(fram)
    frame3.place(x=0 , y=fh0, height=fh0, width=fw)
    button3 = tk.Button(frame3, text="resume learnign ", command=resume)
    button3.place(x=230, y=20, height=50, width=140)
    # Create a button that will open a file dialog
    """open_button1 = tk.Button(frame1, text="Open Image", command=open_dialog)
    open_button1.place(x=fw0 / 2 - 50, y=20, height=50, width=100)"""
def open_new_window02():
    # Create a new window
    def open_dialog():
        # Open a file dialog and get the path to the selected image
        global file_path
        file_path = filedialog.askopenfilename(initialdir="dedicine_label")


        # Open the image file with PIL and convert it to a PhotoImage object
        img = Image.open(file_path)
        img = img.resize((400, 300))
        img = ImageTk.PhotoImage(img)
        img_label1.configure(image=img)
          # keep a reference to the image to prevent garbage collection
        img_label1.image = img

        mser.configure(image=wait0)
        mser.image = wait0

        swt0.configure(image=wait0)
        swt0.image = wait0

        text.configure(image=wait0)
        text.image = wait0

        word.configure(image=wait0)
        word.image = wait0

        mergin.configure(image=wait0)
        mergin.image = wait0
        img_label2.configure(image=wait0)
        img_label2.image = wait0
        img_label3.configure(image=wait0)
        img_label3.image = wait0
        canvas.delete("all")

    def recog():


        img=cv2.imread(file_path)
        img = cv2.resize(img, (500, 300))
        if (pre):
            img_label3.configure(image=process)
            img_label3.image = process
            img_label3.update()
            result,images,lables=ex.write_on_image(img,pre,boxes)
            img = Image.fromarray(result)
            img= img.resize((400, 300))
            img = ImageTk.PhotoImage(img)
            img_label3.configure(image=img)
            img_label3.image = img
            conva(images,lables)

    def extr():
        global boxes
        img_label2.configure(image=process)
        img_label2.image = process
        img_label2.update()

        print(file_path)
        img=cv2.imread(file_path)
        img = cv2.resize(img, (500, 300))
        img1=numpy.copy(img)

        edgg, gray, bboxes = ex.get_canny_mser(img)
        ss = ex.np.copy(img)
        for box in bboxes:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        img1 = Image.fromarray(ss)
        #cv2.imwrite("000.png",ss )

        img1 = img1.resize(( 230,150))
        img1 = ImageTk.PhotoImage(img1)
        mser.configure(image=img1)
        mser.image = img1
        mser.update()

        swt = ex.swt_text(edgg, gray)
        ss = cv2.cvtColor(swt, cv2.COLOR_GRAY2BGR)

        #cv2.imwrite("111.png",ss )
        img1 = Image.fromarray(ss)
        img1 = img1.resize(( 230,150))
        img1 = ImageTk.PhotoImage(img1)
        swt0.configure(image=img1)
        swt0.image = img1
        swt0.update()

        bboxes = ex.np.array(ex.swt_filter(bboxes, swt))
        ss = ex.np.copy(img)
        for box in bboxes:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        img1 = Image.fromarray(ss)
        #cv2.imwrite("222",ss )

        img1 = img1.resize(( 230,150))
        img1 = ImageTk.PhotoImage(img1)
        mergin.configure(image=img1)
        mergin.image = img1
        mergin.update()

        bboxes = ex.group_rectangles(bboxes)
        ss = ex.np.copy(img)
        for box in bboxes:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        img1 = Image.fromarray(ss)
        #cv2.imwrite("333.png",ss )

        img1 = img1.resize(( 230,150))
        img1 = ImageTk.PhotoImage(img1)
        text.configure(image=img1)
        text.image = img1
        text.update()
        ss = ex.np.copy(img)
        boxes = ex.word_region(img, bboxes)
        for box in boxes:
            cv2.rectangle(ss, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        img1 = Image.fromarray(ss)
        #cv2.imwrite("444.png",ss )


        img1 = img1.resize((400, 300))
        img11 = ImageTk.PhotoImage(img1)
        img_label2.configure(image=img11)
        img_label2.image = img11

        img1 = img1.resize((230,150))
        img1 = ImageTk.PhotoImage(img1)
        word.configure(image=img1)
        word.image = img1

    def conva(images,lables):

        canvas.delete("all")
        # Add images and labels to the Canvas widget
        for i, img in enumerate(images):
            # Create a frame for each image
            image_frame = Frame(canvas)
            # Add the image to the frame
            image_label = Label(image_frame, image=img)
            image_label.image = img  # ensure the image object is not garbage collected
            image_label.pack(side="left")
            # Add a text label to the frame
            text_label = Label(image_frame, text=lables[i])
            text_label.pack(side="left")
            # Add the frame to the canvas
            canvas.create_window(0, i * img.height(), anchor="nw", window=image_frame)

        # Update the scroll region after adding images to the canvas
        canvas.configure(scrollregion=canvas.bbox("all"))
    fram = tk.Toplevel(root)
    fw = root.winfo_screenwidth()
    fh = root.winfo_screenheight()
    fram.geometry(str(fw)+"x"+str(fh))
    fram.title('Drug label recognition ')
    print(fw)
    print(fh)
    fw0=fw*0.33
    fh0=fh*0.5
    frame1 = tk.Frame(fram)
    frame1.place(x=0, y=0,height=fh0,width=fw0)

    frame2 = tk.Frame(fram)
    frame2.place(x=fw0, y=0,height=fh0,width=fw0)

    frame3 = tk.Frame(fram)
    frame3.place(x=fw0*2, y=0,height=fh0,width=fw0)
    frame4 = tk.Frame(fram)
    frame4.place(x=0, y=fh0, height=fh0, width=fw)

    # Create a button that will open a file dialog
    open_button1 = tk.Button(frame1, text="Open Image", command=open_dialog)
    open_button1.place(x=fw0/2-50, y=20,height=50,width=100)
    open_button2 = tk.Button(frame2, text="extraction ", command=extr)
    open_button2.place(x=fw0/2-50, y=20,height=50,width=100)
    open_button3 = tk.Button(frame3, text="recognition ", command=recog)
    open_button3.place(x=fw0/2-50, y=20, height=50, width=100)
    img_label1 = tk.Label(frame1)
    img_label2 = tk.Label(frame2)
    img_label3 = tk.Label(frame3)

    wait = Image.open("img.png")
    wait = wait.resize((400, 300))
    wait = ImageTk.PhotoImage(wait)
    img_label1.configure(image=wait)
    img_label1.image = wait
    img_label1.place(x=10, y=80, height=300, width=400 )
    img_label2.configure(image=wait)
    img_label2.image = wait
    img_label2.place(x=10, y=80, height=300, width=400)
    img_label3.configure(image=wait)
    img_label3.image = wait
    img_label3.place(x=10, y=80, height=300, width=400)

    wait0 = Image.open("img.png")
    wait0 = wait0.resize((230, 150))
    wait0 = ImageTk.PhotoImage(wait0)
    frame40 = tk.Frame(frame4 )
    frame40.place(x=0, y=0, height=fh0, width=fw/5)

    mser = tk.Label(frame40)
    mser.configure(image=wait0)
    mser.image = wait0
    mser.place(x=20, y=5, height=150, width=230)

    swt0 = tk.Label(frame40)
    swt0.configure(image=wait0)
    swt0.image = wait0
    swt0.place(x=20, y=160, height=150, width=230)

    frame41 = tk.Frame(frame4,bg="white")
    frame41.place(x=fw/5, y=0, height=fh0, width=fw/5)
    mergin = tk.Label(frame41)
    mergin.configure(image=wait0)
    mergin.image = wait0
    mergin.place(x=20, y=85, height=150, width=230)

    frame42 = tk.Frame(frame4)
    frame42.place(x=fw/5*2, y=0, height=fh0, width=fw/5)
    text = tk.Label(frame42)
    text.configure(image=wait0)
    text.image = wait0
    text.place(x=20, y=85, height=150, width=230)

    frame43 = tk.Frame(frame4,bg="white")
    frame43.place(x=fw/5*3, y=0, height=fh0, width=fw/5)
    word = tk.Label(frame43)
    word.configure(image=wait0)
    word.image = wait0
    word.place(x=20, y=85, height=150, width=230)

    frame44 = tk.Frame(frame4)
    frame44.place(x=fw/5*4, y=0, height=fh0, width=fw/5)
    canvas = Canvas(frame44)
    canvas.place(x=0, y=0, height=fh0, width=fw/5)
    scrollbar = Scrollbar(frame44, command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    process = Image.open("img.png")
    process = process.resize((400, 300))
    process = ImageTk.PhotoImage(process)
    process0 = Image.open("img.png")
    process0 = process0.resize((230, 150))
    process0 = ImageTk.PhotoImage(process0)
def open_new_window03():
    new_window = tk.Toplevel(root)
    new_window.geometry("1200x600")
    new_window.title("help ")
    label = tk.Label(new_window, text="you can contact with us \n eliasboulham@gmail.com",font=("Helvetica", 25, "bold"))
    label.pack(pady=20)
    new_window.mainloop()
def aa():
   global pre

   pre=mm.get_pr("models/model05")

root = tk.Tk()
fw = root.winfo_screenwidth()
fh = root.winfo_screenheight()
#fram.geometry(str(fw) + "x" + str(fh))
root.geometry(str(fw) + "x" + str(fh))  # Set the window size
threadd = threading.Thread(target=aa)
threadd.start()
#pred = threadd.result
# Create a label
label = tk.Label(root, text="drug label reader system ", font=("Helvetica", 20, "bold"))
label.pack(pady=20)  # Add some padding
label0 = tk.Label(root, text=" University of MOHAMMAD SEDDIK BENYAHIA of JIJEL ", font=("Helvetica", 10, "bold"))
label0.place(x=120, y=400,height=50,width=360)  # Add some padding
label01 = tk.Label(root, text=" By: \n Elias boulahm ", font=("Helvetica", 15, "bold"))
label01.place(x=20, y=500,height=50,width=220)  # Add some padding
label02 = tk.Label(root, text="Supervisor :\n Mokhtar TAFFAR", font=("Helvetica", 15, "bold"))
label02.place(x=400, y=500,height=50,width=200)  # Add some padding
label03 = tk.Label(root, text=" Year : 2022/2023 ", font=("Helvetica", 15, "bold"))
label03.place(x=250, y=600,height=20,width=170)  # Add some padding
button1 = tk.Button(root, text="Learn more",font=("Helvetica", 20, "bold"), command=open_new_window01)
button1.pack(pady=10)
button2 = tk.Button(root, text="Read labels ",font=("Helvetica", 20, "bold"), command=open_new_window02)
button2.pack(pady=10)
button3 = tk.Button(root, text="Helps ",font=("Helvetica", 20, "bold"), command=open_new_window03)
button3.pack(pady=10)
root.update_idletasks()  # Update the window's geometry information
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - root.winfo_width()) // 2
y = (screen_height - root.winfo_height()) // 2
root.geometry(f"+{x}+{y}")  # Center the window on the screen
root.mainloop()

import cv2
from PIL import ImageTk
import PIL.Image
from tkinter import *
from tkinter import filedialog
import tkinter.messagebox
from functools import partial
import numpy as np

# declare image path as a global variable


def browseImage():
    global image
    path = filedialog.askopenfilename()
    image = cv2.imread(path)
    print(type(image))

    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert the images to PIL format...
    origPIL = PIL.Image.fromarray(orig)

    # ...and then to ImageTk format
    origTK = ImageTk.PhotoImage(origPIL)

    orgImg.configure(image=origTK)
    orgImg.image = origTK

    imgLabel.configure(text="Original")

    image = orig


    #histogram.configure(command=histoEqualizer(image))
    #median.configure(command=medianFilter(image))
    #gaussian.configure(command=gaussianMask(image))


def showNewImage(img):
    newImg.configure(image=img)

    newImg.image = img


def histoEqualizer(img):
    global image
    print(type(image))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)

    equPIL = PIL.Image.fromarray(equ)

    equTk = ImageTk.PhotoImage(equPIL)

    editLabel.configure(text="Histogram Equalization")

    showNewImage(equTk)

    #Changes global image to filtered image
    image = equ


def gaussianMask(img):
    global image
    print(type(image))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #kernal = cv2.getGaussianKernel()

    gauss = cv2.GaussianBlur(img, (5, 5), 0)

    gaussPIL = PIL.Image.fromarray(gauss)

    gaussTk = ImageTk.PhotoImage(gaussPIL)

    editLabel.configure(text="Gaussian Mask")

    showNewImage(gaussTk)

    #Changes global image to filtered image
    image = gauss


def medianFilter(img):
    global image
    print(type(image))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(img, 3)

    medPIL = PIL.Image.fromarray(med)

    medTk = ImageTk.PhotoImage(medPIL)

    editLabel.configure(text="Median Filtering")

    showNewImage(medTk)

    #Changes global image to filtered image
    image = med



# opens a messgae box about the details of the program
def aboutProgram():
    tkinter.messagebox.showinfo("About this Program", "This program is about blah blah blah")


# opens a messgae box on how to use the program
def helpBox():
    tkinter.messagebox.showinfo("How to Use", "To use this program blah blah blah")


def doNothing():
    print("nothing")


# opens up a window, config title bar and size
root = Tk()
root.title("Blah")
root.geometry('1500x800')

# menu bar
menu = Menu(root)
root.config(menu=menu)

# add a "File" dropdown menu
fileMenu = Menu(menu)
menu.add_cascade(label="File", menu=fileMenu)
fileMenu.add_command(label="Browse Image...", command=browseImage)
#fileMenu.add_command(label="Save as..")
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=quit)

# add a "Help" dropdown menu
helpMenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpMenu)
helpMenu.add_command(label="Help", command=aboutProgram)
helpMenu.add_command(label="About", command=helpBox)

# add a toolbar to the top of the window
toolbar = Frame(root, bd=1, relief=SUNKEN)
toolbar.pack(side=TOP, fill=X)
# add a browse image button to the toolbar
browse = Button(toolbar, text="Browse...", command=browseImage)
browse.pack(side=LEFT, padx=2, pady=2)

# add a main frame
mainFrame = Frame(root, width=700, height=700)
mainFrame.pack()

# title label
welcomeLabel = Label(mainFrame, text="WELCOME", font="Helvetica", fg="purple").pack(side=TOP, pady=5)

imgLabel = Label(mainFrame, text="", font="Helvetica", fg="black")
imgLabel.pack(side=LEFT)
editLabel = Label(mainFrame, text="", font="Helvetica", fg="black")
editLabel.pack(side=RIGHT)

#Display original image
orgImg = Label(mainFrame, bg='white')
orgImg.pack(side=LEFT, expand=YES, fill=BOTH)

#Display after applying filter
newImg = Label(mainFrame, bg='white')
newImg.pack(expand=YES, fill=BOTH)

#Filter buttons
histogram = Button(toolbar, text="Histogram Equalization", command=lambda: histoEqualizer(image)).pack(side=LEFT, padx=2, pady=2)
median = Button(toolbar, text="Median Filtering", command=lambda: medianFilter(image)).pack(side=LEFT, padx=2, pady=2)
gaussian = Button(toolbar, text="Gaussian Mask", command=lambda: gaussianMask(image)).pack(side=LEFT, padx=2, pady=2)

#Unnecessary label?
#Label(root, text="Preparing to do nothing...", bd=1, relief=SUNKEN, anchor=W).pack(side=BOTTOM, fill=X)

root.mainloop()

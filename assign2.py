import cv2
from PIL import ImageTk
import PIL.Image
from tkinter import *
from tkinter import filedialog
import tkinter.messagebox
from functools import partial

def browseImage():
    global image
    global imageEdit
    global path

    path = filedialog.askopenfilename()
    imageEdit = cv2.imread(path)
    image = imageEdit

    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert the images to PIL format...
    origPIL = PIL.Image.fromarray(orig)

    # ...and then to ImageTk format
    origTK = ImageTk.PhotoImage(origPIL)

    orgImg.configure(image=origTK)
    orgImg.image = origTK

    imgLabel.configure(text="Original")

    image = orig

def saveImage(img):
    global image
    global path
    cv2.imwrite(path+".jpg", img)


def originalImage():
    global image
    global imageEdit

    #Reset image to original
    image = imageEdit

    #Convert to make it viewable in Tkinter
    imagePIL = PIL.Image.fromarray(imageEdit)
    imageTK = ImageTk.PhotoImage(imagePIL)

    #Label stuff
    orgImg.configure(image=imageTK)
    orgImg.image=imageTK
    editLabel.configure(text="Original")

    #Show image
    showNewImage(imageTK)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def histoEqualizer(img):
    global image
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)

    equPIL = PIL.Image.fromarray(equ)

    equTk = ImageTk.PhotoImage(equPIL)

    editLabel.configure(text="Histogram Equalization")

    showNewImage(equTk)

    #Changes global image to filtered image
    image = equ
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussianMask(img):
    global image
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gauss = cv2.GaussianBlur(img, (3, 3), 0)

    gaussPIL = PIL.Image.fromarray(gauss)

    gaussTk = ImageTk.PhotoImage(gaussPIL)

    editLabel.configure(text="Gaussian Mask")

    showNewImage(gaussTk)

    #Changes global image to filtered image
    image = gauss
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def medianFilter(img):
    global image
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(img, 3)

    medPIL = PIL.Image.fromarray(med)

    medTk = ImageTk.PhotoImage(medPIL)

    editLabel.configure(text="Median Filtering")

    showNewImage(medTk)

    #Changes global image to filtered image
    image = med
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def edgeCanny(img):
    global image

    edge = cv2.Canny(img, 100, 200)

    edgePIL = PIL.Image.fromarray(edge)

    edgeTK = ImageTk.PhotoImage(edgePIL)

    editLabel.configure(text="Edges (Canny)")

    showNewImage(edgeTK)

    image = edge

def edgeSobelX(img):
    global image

    edge = cv2.Sobel(img,cv2.COLOR_BGR2GRAY,1,0,ksize=5)
#
    edgePIL = PIL.Image.fromarray(edge)
#
    edgeTK = ImageTk.PhotoImage(edgePIL)
#
    editLabel.configure(text="Edges (Canny)")
#
    showNewImage(edgeTK)
#
    image = edge

def edgeSobelY(img):
    global image

    edge = cv2.Sobel(img,cv2.COLOR_BGR2GRAY,0,1,ksize=5)
#
    edgePIL = PIL.Image.fromarray(edge)
#
    edgeTK = ImageTk.PhotoImage(edgePIL)
#
    editLabel.configure(text="Edges (Canny)")
#
    showNewImage(edgeTK)
#
    image = edge

def showNewImage(img):
    newImg.configure(image=img)

    newImg.image = img

def undo():
    global image

# opens a messgae box about the details of the program
def aboutProgram():
    tkinter.messagebox.showinfo("About this Program", "This program is about blah blah blah")


# opens a messgae box on how to use the program
def helpBox():
    tkinter.messagebox.showinfo("1. Click Browse to choose image. 2. Apply desired filters. 3. Apply desired edge detection. 4. Save image if desired.")

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
fileMenu.add_command(label="Save as..", command=lambda: saveImage(image))
fileMenu.add_separator()
fileMenu.add_command(label="Exit", command=quit)

# add a "Help" dropdown menu
helpMenu = Menu(menu)
menu.add_cascade(label="Help", menu=helpMenu)
helpMenu.add_command(label="Help", command=aboutProgram)
helpMenu.add_command(label="About", command=helpBox)

#Add another toolbar for browsing/saving
fileButtons = Frame(root, bd=1, relief=SUNKEN)
fileButtons.pack(side=TOP, fill=X)

#Add buttons to toolbar
browse = Button(fileButtons, text="Browse...", command=browseImage)
browse.pack(side=LEFT, padx=2, pady=2)
save = Button(fileButtons, text="Save as...", command=lambda: saveImage(image))
save.pack(side=LEFT, padx=2, pady=2)

# add a filterButtons to the top of the window
filterButtons = Frame(root, bd=1, relief=SUNKEN)
filterButtons.pack(side=TOP, fill=X)
#Add buttons to toolbar
histogram = Button(filterButtons, text="Histogram Equalization", command=lambda: histoEqualizer(image)).pack(side=LEFT, padx=2, pady=2)
median = Button(filterButtons, text="Median Filtering", command=lambda: medianFilter(image)).pack(side=LEFT, padx=2, pady=2)
gaussian = Button(filterButtons, text="Gaussian Mask", command=lambda: gaussianMask(image)).pack(side=LEFT, padx=2, pady=2)
original = Button(filterButtons, text="Original", command=lambda: originalImage()).pack(side=LEFT, padx=2, pady=2)

#Add edge detection to a new toolbar
edgeButtons = Frame(root, bd=1, relief=SUNKEN)
edgeButtons.pack(side=TOP, fill=X)
#Add buttons to toolbar
canny = Button(edgeButtons, text="Canny Edges", command=lambda: edgeCanny(image)).pack(side=LEFT, padx=2, pady=2)
sobel = Button(edgeButtons, text="Sobel Filter", command=lambda: edgeSobelX(image)).pack(side=LEFT, padx=2, pady=2)

# add a main frame
mainFrame = Frame(root, width=700, height=700)
mainFrame.pack()

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

Label(root, text="When finished, File --> Exit...", bd=1, relief=SUNKEN, anchor=W).pack(side=BOTTOM, fill=X)

root.mainloop()

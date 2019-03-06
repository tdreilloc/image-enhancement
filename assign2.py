import cv2
from PIL import ImageTk
import PIL.Image
from tkinter import *
from tkinter import filedialog
import tkinter.messagebox

# declare image path as a global variable
image = ""


def browseImage():
    path = filedialog.askopenfilename()
    image = cv2.imread(path)

    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert the images to PIL format...
    orig = PIL.Image.fromarray(orig)

    # ...and then to ImageTk format
    orig = ImageTk.PhotoImage(orig)

    orgImg.configure(image=orig)
    orgImg.image = orig

    imgLabel.configure(text="Original")

    def histoEqualizer():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)

        equ = PIL.Image.fromarray(equ)

        equ = ImageTk.PhotoImage(equ)

        newImg.configure(image=equ)

        newImg.image = equ

        editLabel.configure(text="Histogram Equalization")

    def gaussianMask():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #kernal = cv2.getGaussianKernel()

        gauss = cv2.GaussianBlur(gray, (5, 5), 0)

        gauss = PIL.Image.fromarray(gauss)

        gauss = ImageTk.PhotoImage(gauss)

        newImg.configure(image=gauss)

        newImg.image = gauss

        editLabel.configure(text="Gaussian Mask")

    def medianFilter():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        med = cv2.medianBlur(gray, 3)

        med = PIL.Image.fromarray(med)

        med = ImageTk.PhotoImage(med)

        newImg.configure(image=med)

        newImg.image = med

        editLabel.configure(text="Median Filtering")

    histogram.configure(command=histoEqualizer)
    median.configure(command=medianFilter)
    gaussian.configure(command=gaussianMask)

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
welcomeLabel = Label(mainFrame, text="WELCOME", font="Helvetica", fg="purple")
welcomeLabel.pack(side=TOP, pady=5)
imgLabel = Label(mainFrame, text="", font="Helvetica", fg="black")
imgLabel.pack(side=LEFT)
editLabel = Label(mainFrame, text="", font="Helvetica", fg="black")
editLabel.pack(side=RIGHT)


orgImg = Label(mainFrame, bg='white')
orgImg.pack(side=LEFT, expand=YES, fill=BOTH)
newImg = Label(mainFrame, bg='white')
newImg.pack(expand=YES, fill=BOTH)

histogram = Button(toolbar, text="Histogram Equalization", command=doNothing)
histogram.pack(side=LEFT, padx=2, pady=2)
median = Button(toolbar, text="Median Filtering", command=doNothing)
median.pack(side=LEFT, padx=2, pady=2)
gaussian = Button(toolbar, text="Gaussian Mask", command=doNothing)
gaussian.pack(side=LEFT, padx=2, pady=2)

status = Label(root, text="Preparing to do nothing...", bd=1, relief=SUNKEN, anchor=W)
status.pack(side=BOTTOM, fill=X)

root.mainloop()

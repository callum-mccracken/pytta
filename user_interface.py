import sys
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import de_utilities as utils
import file_reader
sys.coinit_flags = 2  # COINIT_APARTMENTTHREADED


def makelabel(master,text):
    """Wrapper for tk.Label"""
    return tk.Label(
        master = master,
        text = text)

def makebutton(master,text,width,height,bg,fg,command):
    return tk.Button(
        master = master,
        text=text,
        width=width,
        height=height,
        bg=bg,
        fg=fg,
        command = command)

def callback():
    print("clicked")
    makelabel(window,"Button clicked")
    return 0


def callfile():
    filename = tk.filedialog.askopenfilename()
    time,data = file_reader.readfile(filename)
    plot(time,data)
    return 0

def plot(x,y):
    lines = []
    fig = plt.figure(figsize = (10,7.5))
    ax = fig.add_axes((0.1,0.1,0.8,0.8))
    lines.append(ax.plot(x,y))
    print(ax.lines)
    lines[0].remove(lines[0][0])
    canvas = FigureCanvasTkAgg(fig,master = window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,window)
    toolbar.update()
    canvas.get_tk_widget().pack()

    button = makebutton(window,"Analyze Data",10,1,"blue","yellow",utils.prepare_data(x,y,250))
    button.pack()

    return 0

def test():
    returned_values.append(2)
    print(returned_values)
    return returned_values

returned_values = []
window = tk.Tk()
window.title("TTA analyzis tool")
window.geometry("1000x1000")
greeting = makelabel(window,"Hello world")
button = makebutton(window,"Find File",25,5,"blue","yellow",callfile)
button.pack()
print(returned_values)
window.update()

#window.withdraw()
#folder_selected = filedialog.askdirectory()
#canvas = tk.Canvas(window,width = 1000, height = 750, bg = "white")
#canvas.create_text(300,50,text = "Hello world",fill = 'black', font = ('Helvetica 15 bold'))
#canvas.pack()
greeting.pack()


window.mainloop()


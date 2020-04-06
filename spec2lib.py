## Author: Gerrit Renner
## Contact: github.com/nihno

## Project description:
## this is a work-around to import, process and export
## IR spectra

## import external packages
from tkinter import *
from tkinter import Menu
from tkinter import filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import savgol_filter

## import internal packages
import importApit
import spec2lib_functions as spc

print('Hello World')

## code

""" CREATE MAIN GUI """
class main_gui_class:
    def __init__(self, master):
        self.master = master
        master.title("WELTBUNT spec2lib by Gerrit Renner")
        master.state('zoomed')
        # CREATE MENU
        master.menu = Menu(master)
        # MENU: FILE
        menu_file = Menu(master.menu, tearoff=0)
        menu_file.add_command(label='Load Spectra', command=lambda:open_spectra(master))
        menu_file.add_command(label='Save Spectra as', command=lambda:save_spectra(master))
        menu_file.add_separator()
        menu_file.add_command(label='Exit', command=lambda:exit())
        # MENU: CONFIG
        menu_config = Menu(master.menu, tearoff=0)
        menu_config.add_command(label='Configurations')
        # MENU: PLACE ON ROOT
        master.menu.add_cascade(label='File', menu=menu_file)
        master.menu.add_cascade(label='Configurations', menu=menu_config)
        master.config(menu=master.menu)
        # TREEVIEW:
        style = ttk.Style()
        style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Calibri', 11)) # Modify the font of the body
        style.configure("mystyle.Treeview.Heading", font=('Calibri', 13,'bold')) # Modify the font of the headings
        style.layout("mystyle.Treeview", [('mystyle.Treeview.treearea', {'sticky': 'nswe'})]) # Remove the borders
        master.tree = ttk.Treeview(master,height = 35, style="mystyle.Treeview")
        master.tree["columns"]=("path")
        master.tree.heading("#0", text="Name",anchor=W)
        master.tree.heading("path", text="Path",anchor=W)
        master.tree.bind("<<TreeviewSelect>>",self.show_spec)
        master.tree.pack(side=LEFT, fill=BOTH, expand=1)
        # PLOT
        master.fig = Figure(dpi=100)
        master.t = np.arange(0, 3, .01)
        master.ax1 = master.fig.add_subplot(211)
        master.ax1.plot(master.t, 2 * np.sin(2 * np.pi * master.t))
        master.ax1.set_xlabel('Wavenumber (1/cm)', fontsize=12, fontweight='bold')
        master.ax1.set_ylabel('Absorbance (a.u.)', fontsize=12, fontweight='bold')
        master.ax1.set_title('Raw Data',fontsize=12,fontweight='bold',loc='left',fontstyle='italic')
        master.ax2 = master.fig.add_subplot(212)
        master.ax2.plot(master.t, 2 * np.sin(2 * np.pi * master.t))
        master.ax2.set_xlabel('Wavenumber (1/cm)', fontsize=12, fontweight='bold')
        master.ax2.set_ylabel('Absorbance (a.u.)', fontsize=12, fontweight='bold')
        master.ax2.set_title('Processed Data',fontsize=12,fontweight='bold',loc='left',fontstyle='italic')
        master.fig.tight_layout()
        master.canvas = FigureCanvasTkAgg(master.fig, master=master)
        master.canvas.draw()
        master.canvas.get_tk_widget().pack(side=RIGHT,fill=BOTH,expand=1)

    def show_spec(self, event):
        curItem = self.master.tree.focus()
        file_path = self.master.tree.item(self.master.tree.selection(),"values")[0] + curItem
        # CHECK FILE TYPE
        cut = curItem.rfind('.')
        if cut > 0:
            format = curItem[cut+1:]
        if format == "csv":
            print('csv')
        if format == "txt":
            print('txt')
        if format == "apit":
            try:
                x,y,table_x,table_y = importApit.import_apit(file_path, calc_mean=False, KK_transform=False, cut=False, extend=True, mapping=True)
            except:
                x,y = importApit.import_apit(file_path, calc_mean=False, KK_transform=False, cut=False, extend=True, mapping=True)
        x_processed, y_processed, baseline = spec_process(x,y)
        # PLOT
        self.master.ax1.cla()
        self.master.ax1.plot(x, y)
        self.master.ax1.plot(x, baseline, linestyle='-.')
        self.master.ax1.set_xlabel('Wavenumber (1/cm)', fontsize=12, fontweight='bold')
        self.master.ax1.set_ylabel('Absorbance (a.u.)', fontsize=12, fontweight='bold')
        self.master.ax1.set_title('Raw Data',fontsize=12,fontweight='bold',loc='left',fontstyle='italic')
        self.master.ax1.set_xlim(np.max(x),np.min(x))

        self.master.ax2.cla()
        self.master.ax2.plot(x_processed, y_processed)
        self.master.ax2.set_xlabel('Wavenumber (1/cm)', fontsize=12, fontweight='bold')
        self.master.ax2.set_ylabel('Absorbance (a.u.)', fontsize=12, fontweight='bold')
        self.master.ax2.set_title('Raw Data',fontsize=12,fontweight='bold',loc='left',fontstyle='italic')
        self.master.ax2.set_xlim(np.max(x),np.min(x))
        self.master.canvas.draw()


""" FUNCTIONS """
""" Open Spectra """
def open_spectra(root):
    files = filedialog.askopenfilenames(filetypes=[("Text Files", ".csv .txt"),("Shimadzu Files", ".apit .amap")])
    create_spectra_list(root,files)

""" Save Spectra as """
def save_spectra(root):
    items = root.tree.get_children()
    processing = 'smoothing: ' + 'Savitzky-Golay(19,3)'
    processing += ', baseline: ' + 'ALS'
    processing += ', normalization: ' + 'Min-Max'
    for item in items:
        curItem = root.tree.item(item,"text")
        file_path = root.tree.item(item,"values")[0] + curItem
        # CHECK FILE TYPE
        cut = curItem.rfind('.')
        if cut > 0:
            format = curItem[cut+1:]
            name = curItem[:cut]
        if format == "csv":
            print('csv')
        if format == "txt":
            print('txt')
        if format == "apit":
            try:
                x,y,table_x,table_y = importApit.import_apit(file_path, calc_mean=False, KK_transform=False, cut=False, extend=True, mapping=True)
            except:
                x,y = importApit.import_apit(file_path, calc_mean=False, KK_transform=False, cut=False, extend=True, mapping=True)
            file_path = file_path[0:-4] + 'jdx'
        x_processed, y_processed, baseline = spec_process(x,y)
        spc.save_jcamp(x_processed,y_processed,x,y,file_path,processing)
    print(values)


""" Create Spectra List """
def create_spectra_list(root,files):
    root.tree.delete(*root.tree.get_children())
    for file in files:
        cut = file.rfind('/')
        if cut > 0:
            path = file[:cut+1]
            file = file[cut+1:]
        root.tree.insert('', 'end', file, text=file, values=(path))

""" DATA PROCESSING """
def spec_process(x,y):
    x_processed = x*1
    y_processed = y*1
    # smoothing
    y_processed = savgol_filter(y_processed,19,3)
    # baseline
    y2 = np.transpose(y_processed)
    y_processed,baseline = spc.baseline_als_auto(x_processed,y2)
    y_processed = y_processed - baseline
    baseline = baseline*(max(y2)-min(y2))+min(y2)
    # normalization
    y_processed = (y_processed-min(y_processed)) / (max(y_processed)-min(y_processed))
    return x_processed, y_processed, baseline

""" Create and Run Main root """
# create root
root = Tk()
main_gui = main_gui_class(root)
root.mainloop()

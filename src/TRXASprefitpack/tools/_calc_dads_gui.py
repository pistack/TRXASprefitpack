'''
calc_dads_gui:
Graphical User Interface for calc_dads utility

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from sys import platform
import re
import tkinter as tk
import tkinter.messagebox as msg
import tkinter.filedialog as fd
import numpy as np
from scipy.linalg import svd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from ..mathfun import calc_eta, calc_fwhm, solve_seq_model
from ..driver import dads, sads, dads_svd, sads_svd

float_sep_comma = re.compile('([\+\-]?[0-9]+[.]?[0-9]*[,]\s*)*[\+\-]?[0-9]+[.]?[0-9]*\s*')
isfloat = re.compile('[\+\-]?[0-9]+[.]?[0-9]*\s*')

# check font
if platform == "linux" or platform == "linux2":
    widgetfont = 'Liberation Sans'
else:
    widgetfont = 'Arial'


class PlotDataWidget:
    '''
    Class for widget which plot data
    '''
    # These codes are based on Matplotlib example "Embedding in Tk"

    def __init__(self, master):

        self.top = tk.Toplevel(master.root)
        self.top.title('Plot Data')

        self.fig = Figure(figsize=(8, 4), dpi=100)

        # immutable
        self.ax = self.fig.add_subplot()
        self.ax.set_xlabel('Energy')
        self.ax.set_ylabel('Intensity')
        self.ax.grid(True)

        # mutable
        self.ax.set_title(f'{master.t[0]}')
        self.ln, = self.ax.plot(master.e, master.escan_mat[:, 0], mfc='none',
        color='black', marker='o')
        self.poly = self.ax.fill_between(master.e,
        master.escan_mat[:, 0]-master.eps[:, 0],
        master.escan_mat[:, 0]+master.eps[:, 0], alpha=0.5, color='black')

        if master.fit:
            self.ln_fit, = self.ax.plot(master.e, master.escan_fit[:, 0],
            color='red', marker=None)

        # set canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()

        self.toolbar = \
            NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect('key_press_event', key_press_handler)

        self.slider_update = tk.Scale(self.top, from_=1, to_=len(master.t),
        orient=tk.HORIZONTAL, command=lambda val: self.update_plot(master, int(val)),
        label='Time Delay index')

        self.slider_update.pack(side=tk.BOTTOM)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.top.mainloop()

    # --- Update plot when one moves slider

    def update_plot(self, master, val):
        self.ax.set_title(f'{master.t[val-1]}')
        self.ln.set_data(master.e, master.escan_mat[:, val-1])
        self.poly.remove()
        self.poly = self.ax.fill_between(master.e,
        master.escan_mat[:, val-1]-master.eps[:, val-1],
        master.escan_mat[:, val-1]+master.eps[:, val-1], 
        alpha=0.5, color='black')
        if master.fit:
            self.ln_fit.set_data(master.e, master.escan_fit[:, val-1])

        ymin = np.min(master.escan_mat[:, val-1])
        ymax = np.max(master.escan_mat[:, val-1])
        errmax = np.max(master.eps[:, val-1])
        self.ax.set_ylim((ymin-5*errmax, ymax+5*errmax))
        self.canvas.draw()

class PlotSVDWidget:
    '''
    Class for widget which plot svd components
    '''
    # These codes are based on Matplotlib example "Embedding in Tk"

    def __init__(self, master):

        self.top = tk.Toplevel(master.root)
        self.top.title('Plot SVD result')

        self.fig = Figure(figsize=(12, 4), dpi=100)

        # immutable
        self.ax_e = self.fig.add_subplot(121)

        self.ax_e.set_xlabel('Energy')
        self.ax_e.set_ylabel('Intensity')
        self.ax_e.grid(True)
        self.ax_e.set_xlim((np.min(master.e), np.max(master.e)))
        self.ax_t = self.fig.add_subplot(122)
        self.ax_t.set_xlabel('Time Delay')
        self.ax_t.set_ylabel('Intensity')
        self.ax_t.grid(True)
        self.ax_t.set_xlim((np.min(master.t), np.max(master.t)))

        # mutable
        self.ax_e.set_title(f'Left Component: {1 :.3e}')
        self.ax_t.set_title(f'Right Component: {1 :.3e}')
        self.ln_e, = self.ax_e.plot(master.e, master.U[:, 0], mfc='none',
        color='black', marker='o')
        self.ln_t, = self.ax_t.plot(master.t, master.Vh[0, :], mfc='none',
        color='black', marker='o')

        # set canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()

        self.toolbar = \
            NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect('key_press_event', key_press_handler)

        self.slider_update = tk.Scale(self.top, from_=1, to_=len(master.sigma),
        orient=tk.HORIZONTAL, command=lambda val: self.update_plot(master, int(val)),
        label='Component index')

        self.slider_update.pack(side=tk.BOTTOM)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.top.mainloop()

    # --- Update plot when one moves slider

    def update_plot(self, master, val):
        self.ax_e.set_title(f'Left Component: {master.sigma[val-1]/master.sigma[0] :.3e}')
        self.ax_t.set_title(f'Right Component: {master.sigma[val-1]/master.sigma[0] :.3e}')
        self.ln_e.set_data(master.e, master.U[:, val-1])
        self.ln_t.set_data(master.t, master.Vh[val-1, :])
        y_emin = np.min(master.U[:, val-1])
        y_emax = np.max(master.U[:, val-1])
        y_tmin = np.min(master.Vh[val-1, :])
        y_tmax = np.max(master.Vh[val-1, :])
        self.ax_e.set_ylim(((y_emin+y_emax)/2-1.05*(y_emax-y_emin)/2,
        (y_emin+y_emax)/2+1.05*(y_emax-y_emin)/2))
        self.ax_t.set_ylim(((y_tmin+y_tmax)/2-1.05*(y_tmax-y_tmin)/2,
        (y_tmin+y_tmax)/2+1.05*(y_tmax-y_tmin)/2))
        self.canvas.draw()


class PlotDADSWidget:
    '''
    Class for wideget which plots DADS
    '''
    # These codes are based on Matplotlib example "Embedding in Tk"

    def __init__(self, master):

        self.top = tk.Toplevel(master.root)
        self.top.title('Plot DADS')

        self.fig = Figure(figsize=(8, 4), dpi=100)

        self.ax_dads = self.fig.add_subplot()
        self.ax_dads.set_title(f'{master.mode_var.get()}')
        self.ax_dads.set_xlabel('Energy')
        self.ax_dads.set_ylabel('Intensity')
        self.ax_dads.grid(True)

        for i in range(len(master.tau)):
            self.ax_dads.plot(master.e, master.dads[:, i], label=f'tau: {master.tau[i]}')
        
        if master.base_var.get():
            self.ax_dads.plot(master.e, master.dads[:, -1], label='long lived')

        self.ax_dads.legend()

        # set canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()

        self.toolbar = \
            NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect('key_press_event', key_press_handler)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.top.mainloop()

class CalcDADSGuiWidget:
    '''
    Class for calc_dads gui wrapper
    '''
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Calc DADS Gui')
        self.fit = False

        # -- define necessary variables
        self.irf_var = tk.StringVar()
        self.base_var = tk.IntVar()
        self.mode_var = tk.StringVar()
        self.escan_file = None
        self.time_file = None
        self.eps_file = None

        # --- Modes

        self.label_mode = tk.Label(self.root, text='Select Calculation Mode',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_mode.grid(column=0, row=0, columnspan=2)
        self.dads_mode = tk.Checkbutton(self.root, text='DADS', variable=self.mode_var,
        onvalue='dads', offvalue='')
        self.dads_mode.grid(column=0, row=1)
        self.dads_svd_mode = tk.Checkbutton(self.root, 
        text='DADS (SVD)', variable=self.mode_var,
        onvalue='dads_svd', offvalue='')
        self.dads_svd_mode.grid(column=1, row=1)
        self.eads_mode = tk.Checkbutton(self.root, text='EADS', variable=self.mode_var,
        onvalue='eads', offvalue='')
        self.eads_mode.grid(column=0, row=2)
        self.eads_svd_mode = tk.Checkbutton(self.root, 
        text='EADS (SVD)', variable=self.mode_var,
        onvalue='eads_svd', offvalue='')
        self.eads_svd_mode.grid(column=1, row=2)

        # --- irf model selection window
        self.irf_label = tk.Label(self.root, text='Select Type of irf',
        padx=90, pady=10, font=(widgetfont, 12))
        self.irf_label.grid(column=0, row=3, columnspan=3)
        self.irf_g = tk.Checkbutton(self.root, text='gaussian', variable=self.irf_var,
        onvalue='g', offvalue='')
        self.irf_g.grid(column=0, row=4)
        self.irf_c = tk.Checkbutton(self.root, text='cauchy', variable=self.irf_var,
        onvalue='c', offvalue='')
        self.irf_c.grid(column=1, row=4)
        self.irf_pv = tk.Checkbutton(self.root, text='pseudo voigt', variable=self.irf_var,
        onvalue='pv', offvalue='')
        self.irf_pv.grid(column=2, row=4)

        # --- SVD options
        self.option_label = tk.Label(self.root, text='SVD Options',
        padx=90, pady=10, font=(widgetfont, 12))
        self.option_label.grid(column=0, row=5, columnspan=3)
        self.label_cond_num = tk.Label(self.root, text='Conditional Number',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_cond_num.grid(column=0, row=6, columnspan=2)
        self.entry_cond_num = tk.Entry(self.root, width=10)
        self.entry_cond_num.grid(column=2, row=6)
        self.entry_cond_num.insert(0, '0')

        # --- miscellaneous options
        self.option_label = tk.Label(self.root, text='Miscellaneous Options',
        padx=120, pady=10, font=(widgetfont, 12))
        self.option_label.grid(column=0, row=7, columnspan=4)
        self.include_base_check = tk.Checkbutton(self.root, text='base',
        variable=self.base_var, onvalue=1, offvalue=0)
        self.include_base_check.grid(column=0, row=8)

        # --- Read file to fit
        self.label_file = tk.Label(self.root, text='Browse Files',
        padx=120, pady=10, font=(widgetfont, 12))
        self.label_file.grid(column=0, row=9, columnspan=4)
        self.print_file_num = tk.Canvas(self.root, width=320, height=20, bd=5,
        bg='white')
        self.print_file_num.grid(column=0, row=10, columnspan=2)
        self.button_file = tk.Button(self.root, width=30, bd=5, text='browse',
        command=self.browse_file)
        self.button_file.grid(column=2, row=10)
        self.button_plot = tk.Button(self.root, width=30, bd=5, text='plot',
        command=self.plot_file)
        self.button_plot.grid(column=3, row=10)

        # --- Parameters
        # 1. fwhm of irf function
        # 2. time zero of energy scan
        # 3. time constant

        self.label_paramter = tk.Label(self.root, text='Parameter',
        font=(widgetfont, 15), padx=90, pady=10)
        self.label_paramter.grid(column=0, row=11, columnspan=3)
        self.label_fwhm_G = tk.Label(self.root, text='fwhm_G (irf)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_fwhm_G.grid(column=0, row=12)
        self.entry_fwhm_G = tk.Entry(self.root, width=10, bd=1)
        self.entry_fwhm_G.grid(column=0, row=13)

        self.label_fwhm_L = tk.Label(self.root, text='fwhm_L (irf)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_fwhm_L.grid(column=1, row=12)
        self.entry_fwhm_L = tk.Entry(self.root, width=10, bd=1)
        self.entry_fwhm_L.grid(column=1, row=13)

        self.label_t0 = tk.Label(self.root,
        text='t0 (escan)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_t0.grid(column=2, row=12)
        self.entry_t0 = tk.Entry(self.root, width=10, bd=1)
        self.entry_t0.grid(column=2, row=13)

        self.label_tau = tk.Label(self.root,
        text='Insert life time parameter (tau1,tau2,...)',
        padx=90, pady=10, font=(widgetfont, 12))
        self.label_tau.grid(column=0, row=14, columnspan=3)
        self.entry_tau = tk.Entry(self.root, width=90, bd=5)
        self.entry_tau.grid(column=0, row=15, columnspan=3)

        # -- Buttons

        self.svd_button = tk.Button(self.root,
        text='SVD', command=self.svd_script,
        font=(widgetfont, 12), padx=30, pady=10, bd=5)
        self.svd_button.grid(column=0, row=16)

        self.run_button = tk.Button(self.root,
        text='Run', command=self.run_script,
        font=(widgetfont, 12), padx=30, pady=10, bd=5)
        self.run_button.grid(column=1, row=16)

        self.save_button = tk.Button(self.root,
        text='Save', command=self.save_script,
        font=(widgetfont, 12), padx=30, pady=10, bd=5)
        self.save_button.grid(column=2, row=16)

        self.root.mainloop()

    # --- read enery scan matrix file
    def browse_file(self):
        self.escan_file = fd.askopenfilename(parent=self.root, 
        title='Choose energy scan matrix file (row: energy, column: time)',
        filetypes=(('any files', '*'), ('text files', '*.txt'), ('data files', '*.dat')))
        self.eps_file = fd.askopenfilename(parent=self.root, 
        title='Choose error matrix file of energy scan (row: energy, column: time)',
        filetypes=(('any files', '*'), ('text files', '*.txt'), ('data files', '*.dat')))
        self.time_file = fd.askopenfilename(parent=self.root, 
        title='Choose file for time delay array of energy scan matrix',
        filetypes=(('any files', '*'), ('text files', '*.txt'), ('data files', '*.dat')))
        self.print_file_num.delete('all')
        self.print_file_num.create_text(200, 15, text='files are successfully loaded')

        self.fit = False
        tmp = np.genfromtxt(self.escan_file)
        self.t = np.genfromtxt(self.time_file)
        self.escan_mat = tmp[:, 1:]
        self.e = tmp[:, 0]
        self.eps = np.genfromtxt(self.eps_file)

        self.U, self.sigma, self.Vh = svd(self.escan_mat, full_matrices=False)

    def plot_file(self):

        if not self.escan_file or not self.time_file:
            msg.showerror('Error', 'Please load escan matrix file and time delay array file')
        else:
            PlotDataWidget(self)
    
    def handle_float(self, entry, entry_name):
        if entry.get():
            test = entry.get()
            if isfloat.match(test):
                value = float(test)
            else:
                msg.showerror('Error',
                f'{entry_name} should be single float number')
                return False
        else:
            msg.showerror('Error',
            f'Please enter {entry_name}')
            return False
        return value

    def handle_float_field(self, entry, entry_name):
        if entry.get():
            tst = entry.get()
            if float_sep_comma.match(tst):
                value = np.array(list(map(float, tst.split(','))))
            else:
                msg.showerror('Error',
                f'{entry_name} should be single float or floats seperated by comma.')
                return False
        else:
            msg.showerror('Error',
            f'Please enter {entry_name}')
            return False
        return value
                
    def handle_irf(self):
        eta = None
        if self.irf_var.get() == 'g':
            fwhm = self.handle_float(self.entry_fwhm_G, 'fwhm_G')
        elif self.irf_var.get() == 'c':
            fwhm = self.handle_float(self.entry_fwhm_L, 'fwhm_L')
        elif self.irf_var.get() == 'pv':
            if self.entry_fwhm_G.get() and self.entry_fwhm_L.get():
                fwhm_G_init = self.entry_fwhm_G.get()
                fwhm_L_init = self.entry_fwhm_L.get()
                if isfloat.match(fwhm_G_init) and isfloat.match(fwhm_L_init):
                    fwhm = calc_fwhm(float(fwhm_G_init), float(fwhm_L_init))
                    eta = calc_eta(float(fwhm_G_init, fwhm_L_init))
                else:
                    msg.showerror('Error',
                    'Both fwhm_G and fwhm_L field should be single float number')
                    return False
            else:
                msg.showerror('Error',
                'Please enter both fwhm_G and fwhm_L values')
                return False
        else:
            msg.showerror('Error',
            'Please select irf model')
            return
        return fwhm, eta

    def handle_t0(self):
        t0 = self.handle_float(self.entry_t0, 't0')
        return t0

    def handle_tau(self):
        if self.entry_tau.get():
            str_tau = self.entry_tau.get()
            if float_sep_comma.match(str_tau):
                tau = np.array(list(map(float, str_tau.split(','))))
            else:
                msg.showerror('Error',
                'life time constant tau should be single float or floats seperated by comma.')
                return False
        else:
            if self.base_var.get():
                return True
            else:
                msg.showerror('Error',
                'Please enter life time constant')
                return False
        return tau
    
    def svd_script(self):
        if not self.escan_file and not self.time_file:
            msg.showerror('Error', 
            'Please load escan matrix file and time delay array file')
        else:
            PlotSVDWidget(self)


    def run_script(self):

        # check files are loaded
        if not self.escan_file and not self.time_file:
            msg.showerror('Error', 
            'Please load escan matrix file and time delay array file')
            return
    

        # set fwhm
        irf = self.irf_var.get()
        fwhm, eta = self.handle_irf()
        if not fwhm:
            return

        # set t0 of energy scan
        t0 = self.handle_t0()
        if not isinstance(t0, float):
            return
        

        base = self.base_var.get()
        self.tau = self.handle_tau()
        if isinstance(self.tau, np.ndarray) or self.tau:
            if not isinstance(self.tau, np.ndarray):
                self.tau = None
        else:
            return
        
        cond_num = self.handle_float(self.entry_cond_num, 'Conditional Number')
        
        # reconstruct escan matrix based on SVD turncation filter
        N_servived = np.sum((self.sigma>cond_num*self.sigma[0]))
        escan_mat_turn = np.einsum('j,ij->ij', 
        self.sigma[:N_servived], self.U[:, :N_servived]) @ self.Vh[:N_servived, :]

        if self.mode_var.get() == 'dads':
            dads_spec, _, self.escan_fit = dads(self.t-t0, fwhm, self.tau, base, 
            irf=irf, eta=eta, intensity=escan_mat_turn)
            self.dads = dads_spec.T
        elif self.mode_var.get() == 'dads_svd':
            dads_spec, self.escan_fit = dads_svd(self.t-t0, fwhm, self.tau, base, 
            irf=irf, eta=eta, intensity=self.escan_mat, cond_num=cond_num)
            self.dads = dads_spec
        elif self.mode_var.get() in ['eads', 'eads_svd']:
            if not base:
                exclude = 'last'
            else:
                exclude = None
            y0 = np.zeros(self.tau.size+1)
            y0[0] = 1
            eigval, V, c = solve_seq_model(self.tau, y0) 
            if self.mode_var.get() == 'eads':
                dads_spec, _, self.escan_fit = sads(self.t-t0, fwhm,
                eigval=eigval, V=V, c=c, 
                exclude=exclude, irf=irf, eta=eta, intensity=escan_mat_turn)
                self.dads = dads_spec.T
            else:
                dads_spec, self.escan_fit = sads_svd(self.t-t0, fwhm,
                eigval=eigval, V=V, c=c, 
                exclude=exclude, irf=irf, eta=eta, intensity=escan_mat_turn,
                cond_num=cond_num)
                self.dads = dads_spec
        else:
            msg.showerror('Error', 'Please select type of calculation')
        self.fit = True
        PlotDADSWidget(self)
        return
    
    def save_script(self):

        if not self.escan_file and not self.time_file:
            msg.showerror('Error', 'Please load escan matrix file and time delay array file')
            return
        
        if not self.fit:
            msg.showerror('Error', 'Please click run before saving results')
            return 

        save_directory = fd.askdirectory(parent=self.root, title='Choose directory to save results')
        np.savetxt(f'{save_directory}/LSVD.txt', 
        np.vstack((self.e, self.U.T)).T)
        np.savetxt(f'{save_directory}/RSVD.txt', 
        np.vstack((self.t, self.Vh)).T)
        np.savetxt(f'{save_directory}/sigma.txt', 
        self.sigma)
        np.savetxt(f'{save_directory}/{self.mode_var.get()}.txt', 
        np.vstack((self.e, self.dads.T)).T)
        np.savetxt(f'{save_directory}/{self.mode_var.get()}_fit.txt', 
        np.vstack((self.e, self.escan_fit.T)).T)
        return

def calc_dads_gui():
    CalcDADSGuiWidget()



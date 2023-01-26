'''
fit_tscan_gui:
Graphical User Interface for fit_tscan utility

:copyright: 2021-2022 by pistack (Junho Lee).
:license: LGPL3.
'''

from sys import platform
import re
import tkinter as tk
import tkinter.messagebox as msg
import tkinter.filedialog as fd
import tkinter.scrolledtext as sc
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from ..res import set_bound_tau
from ..driver import fit_transient_exp, fit_transient_raise
from ..driver import fit_transient_dmp_osc, fit_transient_both
from ..driver import save_TransientResult

FITDRIVER = {'decay': fit_transient_exp, 'raise': fit_transient_raise,
'dmp_osc': fit_transient_dmp_osc, 'both': fit_transient_both}

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
        self.ax.set_xlabel('Time Delay')
        self.ax.set_ylabel('Intensity')
        self.ax.grid(True)

        # mutable
        self.ax.set_title(f'{master.fname[0]}')
        self.ln, = self.ax.plot(master.t[0], master.intensity[0][:, 0], mfc='none',
        color='black', marker='o')
        self.poly = self.ax.fill_between(master.t[0],
        master.intensity[0][:, 0]-master.eps[0][:, 0],
        master.intensity[0][:, 0]+master.eps[0][:, 0], alpha=0.5, color='black')

        # set canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()

        self.toolbar = \
            NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect('key_press_event', key_press_handler)

        self.slider_update = tk.Scale(self.top, from_=1, to_=len(master.fname),
        orient=tk.HORIZONTAL, command=lambda val: self.update_plot(master, int(val)),
        label='File index')

        self.slider_update.pack(side=tk.BOTTOM)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.top.mainloop()

    # --- Update plot when one moves slider

    def update_plot(self, master, val):
        self.ax.set_title(f'{master.fname[val-1]}')
        self.ln.set_data(master.t[val-1], master.intensity[val-1][:, 0])
        self.poly.remove()
        self.poly = self.ax.fill_between(master.t[val-1],
        master.intensity[val-1][:, 0]-master.eps[val-1][:, 0],
        master.intensity[val-1][:, 0]+master.eps[val-1][:, 0], alpha=0.5,
        color='black')
        xmin = np.min(master.t[val-1])
        xmax = np.max(master.t[val-1])
        ymin = np.min(master.intensity[val-1][:, 0])
        ymax = np.max(master.intensity[val-1][:, 0])
        errmax = np.max(master.eps[val-1][:, 0])
        self.ax.set_xlim((xmin, xmax))
        self.ax.set_ylim((ymin-5*errmax, ymax+5*errmax))
        self.canvas.draw()


class PlotFitWidget:
    '''
    Class for wideget which plots fitting result
    '''
    # These codes are based on Matplotlib example "Embedding in Tk"

    def __init__(self, master):

        self.top = tk.Toplevel(master.root)
        self.top.title('Plot Fitting Result')

        self.fig = Figure(figsize=(8, 8), dpi=100)

        # immutable
        # fit
        self.ax_fit = self.fig.add_subplot(211)
        self.ax_fit.set_xlabel('Time Delay')
        self.ax_fit.set_ylabel('Intensity')
        self.ax_fit.grid(True)

        # residual
        self.ax_res = self.fig.add_subplot(212)
        self.ax_res.set_xlabel('Time Delay')
        self.ax_res.set_ylabel('Residual')
        self.ax_res.grid(True)

        # mutable

        # fit
        self.ax_fit.set_title(f'{master.fname[0]}')
        self.ln_fit, = self.ax_fit.plot(master.t[0], master.result['fit'][0][:, 0],
        color='red')
        self.ln_data, = self.ax_fit.plot(master.t[0], master.intensity[0][:, 0], mfc='none',
        color='black', marker='o')
        self.poly_data = self.ax_fit.fill_between(master.t[0],
        master.intensity[0][:, 0]-master.eps[0][:, 0],
        master.intensity[0][:, 0]+master.eps[0][:, 0], alpha=0.5, color='black')

        # residual
        if master.fit_mode_var.get() in ['decay', 'raise', 'dmp_osc']:
            self.ln_res, = self.ax_res.plot(master.t[0],
            master.intensity[0][:, 0]-master.result['fit'][0][:, 0],
            mfc='none', color='black', marker='o')
            self.poly_res = self.ax_res.fill_between(master.t[0],
            master.intensity[0][:, 0]-master.result['fit'][0][:, 0]-master.eps[0][:, 0],
            master.intensity[0][:, 0]-master.result['fit'][0][:, 0]+master.eps[0][:, 0],
            alpha=0.5, color='black')
        else:
            self.ln_fit_osc, = self.ax_res.plot(master.t[0], master.result['fit_osc'][0][:, 0],
            color='red')
            self.ln_res, = self.ax_res.plot(master.t[0],
            master.intensity[0][:, 0]-master.result['fit_decay'][0][:, 0],
            mfc='none', color='black', marker='o')
            self.poly_res = self.ax_res.fill_between(master.t[0],
            master.intensity[0][:, 0]-master.result['fit_decay'][0][:, 0]-master.eps[0][:, 0],
            master.intensity[0][:, 0]-master.result['fit_decay'][0][:, 0]+master.eps[0][:, 0],
            alpha=0.5, color='black')



        # set canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()

        self.toolbar = \
            NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect('key_press_event', key_press_handler)

        self.slider_update = tk.Scale(self.top, from_=1, to_=len(master.fname),
        orient=tk.HORIZONTAL, command=lambda val: self.update_plot(master, int(val)),
        label='File index')

        self.slider_update.pack(side=tk.BOTTOM)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.top.mainloop()

    # --- Update plot when one moves slider

    def update_plot(self, master, val):

        # update fitting window
        self.ax_fit.set_title(f'{master.fname[val-1]}')
        self.ln_fit.set_data(master.t[val-1], master.result['fit'][val-1][:, 0])
        self.ln_data.set_data(master.t[val-1], master.intensity[val-1][:, 0])
        self.poly_data.remove()
        self.poly_data = self.ax_fit.fill_between(master.t[val-1],
        master.intensity[val-1][:, 0]-master.eps[val-1][:, 0],
        master.intensity[val-1][:, 0]+master.eps[val-1][:, 0], alpha=0.5,
        color='black')
        xmin = np.min(master.t[val-1])
        xmax = np.max(master.t[val-1])
        ymin_fit = np.min(master.intensity[val-1][:, 0])
        ymax_fit = np.max(master.intensity[val-1][:, 0])
        errmax = np.max(master.eps[val-1][:, 0])
        self.ax_fit.set_xlim((xmin, xmax))
        self.ax_fit.set_ylim((ymin_fit-5*errmax, ymax_fit+5*errmax))

        # update residual window

        if master.fit_mode_var.get() in ['decay', 'raise', 'dmp_osc']:
            self.ln_res.set_data(master.t[val-1],
            master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0])
            self.poly_res.remove()
            self.poly_res = self.ax_res.fill_between(master.t[val-1],
            master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0]-master.eps[val-1][:, 0],
            master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0]+master.eps[val-1][:, 0],
            alpha=0.5, color='black')
            ymax_res = np.max(master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0])
            ymin_res = np.min(master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0])
        else:
            self.ln_fit_osc.set_data(master.t[val-1], master.result['fit_osc'][val-1][:, 0])
            self.ln_res.set_data(master.t[val-1],
            master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0])
            self.poly_res.remove()
            self.poly_res = self.ax_res.fill_between(master.t[0],
            master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0]-master.eps[val-1][:, 0],
            master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0]+master.eps[val-1][:, 0],
            alpha=0.5, color='black')
            ymax_res = np.max(master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0])
            ymin_res = np.min(master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0])

        self.ax_res.set_xlim((xmin, xmax))
        self.ax_res.set_ylim((ymin_res-2*errmax, ymax_res+2*errmax))

        self.canvas.draw()

class FitReportWidget:
    '''
    Class for reporting fitting result
    '''
    def __init__(self, result):
        self.top = tk.Toplevel()
        self.top.title('Fitting Report')
        self.report_space = sc.ScrolledText(self.top,
        width = 80, height = 40)
        self.report_space.insert(tk.INSERT, str(result))
        self.report_space.configure(state='disabled')
        self.report_space.grid(column=0)
        self.top.mainloop()

class FitTscanGuiWidget:
    '''
    Class for fit_tscan gui wrapper
    '''
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Fit Tscan Gui')
        self.parameter_window = tk.Tk()
        self.parameter_window.title('Initial fitting parameter')
        self.result = None
        self.fit = False

        # -- define necessary variables
        self.irf_var = tk.StringVar()
        self.fit_mode_var = tk.StringVar()
        self.base_var = tk.IntVar()
        self.fix_irf_var = tk.IntVar()
        self.fix_raise_var = tk.IntVar()
        self.fix_t0_var = tk.IntVar()
        self.custom_bd_var = tk.IntVar()
        self.glb_opt_var = tk.StringVar()
        self.fname = []

        # --- fitting model selection window
        self.fit_mode_label = tk.Label(self.root, text='Select fitting mode',
        padx=90, pady=10, font=(widgetfont, 12))
        self.fit_mode_label.grid(column=0, row=0, columnspan=4)
        self.decay_mode = tk.Checkbutton(self.root, text='decay',
        variable = self.fit_mode_var, onvalue = 'decay', offvalue='')
        self.decay_mode.grid(column=0, row=1)
        self.raise_mode = tk.Checkbutton(self.root, text='raise',
        variable = self.fit_mode_var, onvalue = 'raise', offvalue='')
        self.raise_mode.grid(column=1, row=1)
        self.dmp_osc_mode = tk.Checkbutton(self.root, text='damped osc',
        variable = self.fit_mode_var, onvalue = 'dmp_osc', offvalue='')
        self.dmp_osc_mode.grid(column=2, row=1)
        self.both_mode = tk.Checkbutton(self.root, text='decay+dmp_osc',
        variable = self.fit_mode_var, onvalue = 'both', offvalue='')
        self.both_mode.grid(column=3, row=1)

        # --- irf model selection window
        self.irf_label = tk.Label(self.root, text='Select Type of irf',
        padx=90, pady=10, font=(widgetfont, 12))
        self.irf_label.grid(column=0, row=2, columnspan=3)
        self.irf_g = tk.Checkbutton(self.root, text='gaussian', variable=self.irf_var,
        onvalue='g', offvalue='')
        self.irf_g.grid(column=0, row=3)
        self.irf_c = tk.Checkbutton(self.root, text='cauchy', variable=self.irf_var,
        onvalue='c', offvalue='')
        self.irf_c.grid(column=1, row=3)
        self.irf_pv = tk.Checkbutton(self.root, text='pseudo voigt', variable=self.irf_var,
        onvalue='pv', offvalue='')
        self.irf_pv.grid(column=2, row=3)

        # --- global optimization Algorithm
        self.glb_opt_label = tk.Label(self.root, text='Global optimization Methods',
        padx=60, pady=10, font=(widgetfont, 12))
        self.glb_opt_label.grid(column=0, row=4, columnspan=2)
        self.glb_opt_ampgo = tk.Checkbutton(self.root, text='AMPGO', variable=self.glb_opt_var,
        onvalue='ampgo', offvalue='')
        self.glb_opt_ampgo.grid(column=0, row=5)
        self.glb_opt_basin = tk.Checkbutton(self.root, text='Basinhopping',
        variable=self.glb_opt_var,
        onvalue='basinhopping', offvalue='')
        self.glb_opt_basin.grid(column=1, row=5)

        # --- miscellaneous options
        self.option_label = tk.Label(self.root, text='Miscellaneous Options',
        padx=120, pady=10, font=(widgetfont, 12))
        self.option_label.grid(column=0, row=6, columnspan=5)
        self.include_base_check = tk.Checkbutton(self.root, text='base',
        variable=self.base_var, onvalue=1, offvalue=0)
        self.include_base_check.grid(column=0, row=7)
        self.fix_irf_check = tk.Checkbutton(self.root, text='fix_irf',
        variable=self.fix_irf_var, onvalue=1, offvalue=0)
        self.fix_irf_check.grid(column=1, row=7)
        self.fix_raise_check = tk.Checkbutton(self.root, text='fix_raise',
        variable=self.fix_raise_var, onvalue=1, offvalue=0)
        self.fix_raise_check.grid(column=2, row=7)
        self.fix_t0_check = tk.Checkbutton(self.root, text='fix_t0',
        variable=self.fix_t0_var, onvalue=1, offvalue=0)
        self.fix_t0_check.grid(column=3, row=7)
        self.custom_bd_check = tk.Checkbutton(self.root, text='custom bound',
        variable=self.custom_bd_var, onvalue=1, offvalue=0)
        self.custom_bd_check.grid(column=4, row=7)

        # --- Read file to fit
        self.label_file = tk.Label(self.root, text='Browse Files to fit',
        padx=120, pady=10, font=(widgetfont, 12))
        self.label_file.grid(column=0, row=8, columnspan=4)
        self.print_file_num = tk.Canvas(self.root, width=320, height=20, bd=5,
        bg='white')
        self.print_file_num.grid(column=0, row=9, columnspan=2)
        self.button_file = tk.Button(self.root, width=30, bd=5, text='browse',
        command=self.browse_file)
        self.button_file.grid(column=2, row=9)
        self.button_plot = tk.Button(self.root, width=30, bd=5, text='plot',
        command=self.plot_file)
        self.button_plot.grid(column=3, row=9)

        self.param_button = tk.Button(self.root,
        text='Parameters', command=self.view_param,
        font=(widgetfont, 12), bg='green', padx=30, pady=10, bd=5, fg='white')
        self.param_button.grid(column=0, row=10)

        self.run_button = tk.Button(self.root,
        text='Run', command=self.run_script,
        font=(widgetfont, 12), bg='blue', padx=30, pady=10, bd=5, fg='white')
        self.run_button.grid(column=1, row=10)

        self.save_button = tk.Button(self.root,
        text='Save', command=self.save_script,
        font=(widgetfont, 12), bg='black', padx=30, pady=10, bd=5, fg='white')
        self.save_button.grid(column=2, row=10)

        self.exit_button = tk.Button(self.root,
        text='Exit', command=self.exit_script,
        font=(widgetfont, 12), bg='red', padx=30, pady=10, bd=5, fg='white')
        self.exit_button.grid(column=3, row=10)

        # ---- Parameters Depending on fitting Model ----

        # Parameter
        self.label_paramter = tk.Label(self.parameter_window, text='Paramter',
        font=(widgetfont, 15), padx=120, pady=10)
        self.label_paramter.grid(column=0, row=0, columnspan=4)
        self.label_fwhm_G = tk.Label(self.parameter_window, text='fwhm_G (irf)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_fwhm_G.grid(column=0, row=1)
        self.entry_fwhm_G = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_fwhm_G.grid(column=0, row=2)

        self.label_fwhm_L = tk.Label(self.parameter_window, text='fwhm_L (irf)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_fwhm_L.grid(column=1, row=1)
        self.entry_fwhm_L = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_fwhm_L.grid(column=1, row=2)

        self.label_t0 = tk.Label(self.parameter_window,
        text='Insert initial time zero parameter (t01,t02,...)',
        padx=90, pady=10, font=(widgetfont, 12))
        self.label_t0.grid(column=0, row=3, columnspan=3)
        self.entry_t0 = tk.Entry(self.parameter_window, width=90, bd=5)
        self.entry_t0.grid(column=0, row=4, columnspan=3)

        self.label_tau = tk.Label(self.parameter_window,
        text='Insert initial life time parameter (tau1,tau2,...)',
        padx=90, pady=10, font=(widgetfont, 12))
        self.label_tau.grid(column=0, row=5, columnspan=3)
        self.entry_tau = tk.Entry(self.parameter_window, width=90, bd=5)
        self.entry_tau.grid(column=0, row=6, columnspan=3)

        self.label_osc = tk.Label(self.parameter_window,
        text='Insert initial osc period parameter (T_osc1,T_osc2,...)',
        padx=90, pady=10, font=(widgetfont, 12))
        self.label_osc.grid(column=0, row=7, columnspan=3)
        self.entry_osc = tk.Entry(self.parameter_window, width=90, bd=5)
        self.entry_osc.grid(column=0, row=8, columnspan=3)

        self.label_dmp = tk.Label(self.parameter_window,
        text='Insert initial damping lifetime parameter (dmp1,dmp2,...)',
        padx=90, pady=10, font=(widgetfont, 12))
        self.label_dmp.grid(column=0, row=9, columnspan=3)
        self.entry_dmp = tk.Entry(self.parameter_window, width=90, bd=5)
        self.entry_dmp.grid(column=0, row=10, columnspan=3)

        # Parameter Bound

        self.label_bound = tk.Label(self.parameter_window, text='Bounds',
        font=(widgetfont, 15), padx=120, pady=10)
        self.label_bound.grid(column=0, row=11, columnspan=4)

        # fwhm_G (lower)
        self.label_bd_l_fwhm_G = tk.Label(self.parameter_window, text='fwhm_G (lower)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_bd_l_fwhm_G.grid(column=0, row=12)
        self.entry_bd_l_fwhm_G = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_bd_l_fwhm_G.grid(column=0, row=13)

        # fwhm_G (upper)
        self.label_bd_u_fwhm_G = tk.Label(self.parameter_window, text='fwhm_G (upper)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_bd_u_fwhm_G.grid(column=1, row=12)
        self.entry_bd_u_fwhm_G = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_bd_u_fwhm_G.grid(column=1, row=13)

        # fwhm_L (lower)
        self.label_bd_l_fwhm_L = tk.Label(self.parameter_window, text='fwhm_L (lower)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_bd_l_fwhm_L.grid(column=2, row=14)
        self.entry_bd_l_fwhm_L = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_bd_l_fwhm_L.grid(column=2, row=15)

        # fwhm_L (upper)
        self.label_bd_u_fwhm_L = tk.Label(self.parameter_window, text='fwhm_L (upper)',
        padx=30, pady=10, font=(widgetfont, 12))
        self.label_bd_u_fwhm_L.grid(column=2, row=14)
        self.entry_bd_u_fwhm_L = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_bd_u_fwhm_L.grid(column=2, row=15)

        # bound time zero
        self.label_bd_l_t0 = tk.Label(self.parameter_window,
        text='Lower bound of time zero (t01,t02,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_l_t0.grid(column=0, row=16, columnspan=2)
        self.entry_bd_l_t0 = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_l_t0.grid(column=0, row=17, columnspan=2)

        self.label_bd_u_t0 = tk.Label(self.parameter_window,
        text='Upper bound of time zero (t01,t02,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_u_t0.grid(column=2, row=16, columnspan=2)
        self.entry_bd_u_t0 = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_u_t0.grid(column=2, row=17, columnspan=2)

        # bound tau
        self.label_bd_l_tau = tk.Label(self.parameter_window,
        text='Lower bound of life time (tau1,tau2,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_l_tau.grid(column=0, row=18, columnspan=2)
        self.entry_bd_l_tau = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_l_tau.grid(column=0, row=19, columnspan=2)

        self.label_bd_u_tau = tk.Label(self.parameter_window,
        text='Upper bound of life time (tau1,tau2,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_u_tau.grid(column=2, row=18, columnspan=2)
        self.entry_bd_u_tau = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_u_tau.grid(column=2, row=19, columnspan=2)

        # bound osc
        self.label_bd_l_osc = tk.Label(self.parameter_window,
        text='Lower bound of osc period (T_osc1,T_osc2,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_l_osc.grid(column=0, row=20, columnspan=2)
        self.entry_bd_l_osc = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_l_osc.grid(column=0, row=21, columnspan=2)

        self.label_bd_u_osc = tk.Label(self.parameter_window,
        text='Upper bound of osc period (T_osc1,T_osc2,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_u_osc.grid(column=2, row=20, columnspan=2)
        self.entry_bd_u_osc = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_u_osc.grid(column=2, row=21, columnspan=2)

        # bound dmp
        self.label_bd_l_dmp = tk.Label(self.parameter_window,
        text='Lower bound of osc life time (dmp1,dmp2,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_l_dmp.grid(column=0, row=22, columnspan=2)
        self.entry_bd_l_dmp = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_l_dmp.grid(column=0, row=23, columnspan=2)

        self.label_bd_u_dmp = tk.Label(self.parameter_window,
        text='Upper bound of osc life time (dmp1,dmp2,...)',
        padx=60, pady=10, font=(widgetfont, 12))
        self.label_bd_u_dmp.grid(column=2, row=22, columnspan=2)
        self.entry_bd_u_dmp = tk.Entry(self.parameter_window, width=60, bd=5)
        self.entry_bd_u_dmp.grid(column=2, row=23, columnspan=2)

        self.hide_init_param_option()

        self.root.mainloop()
        self.parameter_window.mainloop()

    # --- Browse root directory of fitting file
    def browse_file(self):
        self.file_lst = fd.askopenfilenames(parent=self.root, title='Choose files',
        filetypes=(('any files', '*'), ('text files', '*.txt'), ('data files', '*.dat')))
        self.print_file_num.delete('all')
        self.print_file_num.create_text(200, 15, text=f'{len(self.file_lst)} number of files are loaded')
        self.fname = []
        self.t = []
        self.intensity = []
        self.eps = []
        for fn in self.file_lst:
            tmp = np.genfromtxt(fn)
            self.t.append(tmp[:, 0])
            self.intensity.append(tmp[:, 1].reshape((tmp.shape[0], 1)))
            self.eps.append(tmp[:, 2].reshape((tmp.shape[0], 1)))
            self.fname.append(fn.split('/')[-1])
        self.fit = False

    def plot_file(self):

        if len(self.fname) == 0:
            msg.showerror('Error', 'Please load files')
        elif self.fit:
            PlotFitWidget(self)
        else:
            PlotDataWidget(self)


    # --- hide fitting parameter entry
    def hide_init_param_option(self):
        # bound
        self.label_bound.grid_remove()

        # irf option
        self.label_fwhm_G.grid_remove()
        self.entry_fwhm_G.grid_remove()
        self.label_fwhm_L.grid_remove()
        self.entry_fwhm_L.grid_remove()

        self.label_bd_l_fwhm_G.grid_remove()
        self.label_bd_u_fwhm_G.grid_remove()
        self.entry_bd_l_fwhm_G.grid_remove()
        self.entry_bd_u_fwhm_G.grid_remove()

        self.label_bd_l_fwhm_L.grid_remove()
        self.label_bd_u_fwhm_L.grid_remove()
        self.entry_bd_l_fwhm_L.grid_remove()
        self.entry_bd_u_fwhm_L.grid_remove()

        # t0 option
        self.label_t0.grid_remove()
        self.entry_t0.grid_remove()

        self.label_bd_l_t0.grid_remove()
        self.entry_bd_l_t0.grid_remove()
        self.label_bd_u_t0.grid_remove()
        self.entry_bd_u_t0.grid_remove()

        # tau option
        self.label_tau.grid_remove()
        self.entry_tau.grid_remove()

        self.label_bd_l_tau.grid_remove()
        self.entry_bd_l_tau.grid_remove()
        self.label_bd_u_tau.grid_remove()
        self.entry_bd_u_tau.grid_remove()

        # osc option
        self.label_osc.grid_remove()
        self.entry_osc.grid_remove()

        self.label_bd_l_osc.grid_remove()
        self.entry_bd_l_osc.grid_remove()
        self.label_bd_u_osc.grid_remove()
        self.entry_bd_u_osc.grid_remove()

        # dmp option
        self.label_dmp.grid_remove()
        self.entry_dmp.grid_remove()

        self.label_bd_l_dmp.grid_remove()
        self.entry_bd_l_dmp.grid_remove()
        self.label_bd_u_dmp.grid_remove()
        self.entry_bd_u_dmp.grid_remove()

    # --- prepare to fit
    def view_param(self):

        # hide all
        self.hide_init_param_option()

        if self.custom_bd_var.get():
            self.label_bound.grid()

        if not self.irf_var.get():
            msg.showerror('Error',
            'Please select the type of irf before clicking ready button')
            return

        # show irf option
        if self.irf_var.get() in ['g', 'pv']:
            self.label_fwhm_G.grid()
            self.entry_fwhm_G.grid()

            if self.custom_bd_var.get():
                self.label_bd_l_fwhm_G.grid()
                self.label_bd_u_fwhm_G.grid()
                self.entry_bd_l_fwhm_G.grid()
                self.entry_bd_u_fwhm_G.grid()
        
        if self.irf_var.get() in ['c', 'pv']:
            self.label_fwhm_L.grid()
            self.entry_fwhm_L.grid()

            if self.custom_bd_var.get():
                self.label_bd_l_fwhm_L.grid()
                self.label_bd_u_fwhm_L.grid()
                self.entry_bd_l_fwhm_L.grid()
                self.entry_bd_u_fwhm_L.grid()

        # show t0 option
        self.label_t0.grid()
        self.entry_t0.grid()

        if self.custom_bd_var.get():
            self.label_bd_l_t0.grid()
            self.label_bd_u_t0.grid()
            self.entry_bd_l_t0.grid()
            self.entry_bd_u_t0.grid()

        # show initial life time related option
        if not self.fit_mode_var.get():
            msg.showerror('Error',
            'Please select the fitting model before clicking ready button')
            return

        if self.fit_mode_var.get() in ['decay', 'raise', 'both']:
            self.label_tau.grid()
            self.entry_tau.grid()

            if self.custom_bd_var.get():
                self.label_bd_l_tau.grid()
                self.label_bd_u_tau.grid()
                self.entry_bd_l_tau.grid()
                self.entry_bd_u_tau.grid()

        if self.fit_mode_var.get() in ['dmp_osc', 'both']:

            self.label_osc.grid()
            self.entry_osc.grid()

            self.label_dmp.grid()
            self.entry_dmp.grid()

            if self.custom_bd_var.get():
                self.label_bd_l_osc.grid()
                self.label_bd_u_osc.grid()
                self.entry_bd_l_osc.grid()
                self.entry_bd_u_osc.grid()
                self.label_bd_l_dmp.grid()
                self.label_bd_u_dmp.grid()
                self.entry_bd_l_dmp.grid()
                self.entry_bd_u_dmp.grid()
    
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
            f'Please click Parameter Button and enter {entry_name}')
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
        if self.irf_var.get() == 'g':
            fwhm = self.handle_float(self.entry_fwhm_G, 'fwhm_G')
        elif self.irf_var.get() == 'c':
            fwhm = self.handle_float(self.entry_fwhm_L, 'fwhm_L')
        elif self.irf_var.get() == 'pv':
            if self.entry_fwhm_G.get() and self.entry_fwhm_L.get():
                fwhm_G_init = self.entry_fwhm_G.get()
                fwhm_L_init = self.entry_fwhm_L.get()
                if isfloat.match(fwhm_G_init) and isfloat.match(fwhm_L_init):
                    fwhm = [float(fwhm_G_init), float(fwhm_L_init)]
                else:
                    msg.showerror('Error',
                    'Both fwhm_G and fwhm_L field should be single float number')
                    return False
            else:
                msg.showerror('Error',
                'Please click Parameter button and enter both initial fwhm_G and fwhm_L values')
                return False
        else:
            msg.showerror('Error',
            'Please select irf model and click paramter button')
        return fwhm

    def handle_t0(self):
        t0 = self.handle_float_field(self.entry_t0, 't0')
        if not isinstance(t0, np.ndarray):
            return t0
        if t0.size != len(self.file_lst):
            msg.showerror('Error',
            'Number of initial time zero should be same as number of files to fit.')
            return False
        return t0

    def handle_tau(self):
        if self.entry_tau.get():
            str_tau = self.entry_tau.get()
            if float_sep_comma.match(str_tau):
                tau = np.array(list(map(float, str_tau.split(','))))
            else:
                msg.showerror('Error',
                'initial life time constant tau should be single float or floats seperated by comma.')
                return False
        else:
            if self.base_var.get():
                return True
            else:
                msg.showerror('Error',
                'Please enter initial life time constant')
                return False
        return tau

    def handle_osc(self):
        osc = self.handle_float_field(self.entry_osc, 'osc period')
        return osc

    def handle_dmp(self):
        dmp = self.handle_float_field(self.entry_dmp, 'dmp life time')
        return dmp
    
    def handle_bd_irf(self):
        bound_fwhm = None
        if self.irf_var.get() in ['g','pv']:
            bd_l_fwhm_G = self.handle_float(self.entry_bd_l_fwhm_G, 'fwhm_G (lower)')
            if not bd_l_fwhm_G:
                return None
            bd_u_fwhm_G = self.handle_float(self.entry_bd_u_fwhm_G, 'fwhm_G (upper)')
            if not bd_u_fwhm_G:
                return None
            bound_fwhm_G = (bd_l_fwhm_G, bd_u_fwhm_G)
        if self.irf_var.get() in ['c', 'pv']:
            bd_l_fwhm_L = self.handle_float(self.entry_bd_l_fwhm_L, 'fwhm_L (lower)')
            if not bd_l_fwhm_L:
                return None
            bd_u_fwhm_L = self.handle_float(self.entry_bd_u_fwhm_L, 'fwhm_L (upper)')
            if not bd_u_fwhm_L:
                return None
            bound_fwhm_L = (bd_l_fwhm_L, bd_u_fwhm_L)
        
        if self.irf_var.get() == 'g':
            bound_fwhm = [bound_fwhm_G]
        elif self.irf_var.get() == 'c':
            bound_fwhm = [bound_fwhm_L]
        elif self.irf_var.get() == 'pv':
            bound_fwhm = [bound_fwhm_G, bound_fwhm_L]
        return bound_fwhm

    def handle_bd_field(self, param, entry_bd_l, entry_bd_u, entry_name):
        bd_l = self.handle_float_field(entry_bd_l, f'{entry_name} (lower)')
        if not isinstance(bd_l, np.ndarray):
            return None
        bd_u = self.handle_float_field(entry_bd_u, f'{entry_name} (upper)')
        if not isinstance(bd_u, np.ndarray):
            return None
        
        if bd_l.size != bd_u.size:
            msg.showerror('Error', 'The size of lower and upper bound should be same')
            return None
        if bd_l.size != param.size:
            msg.showerror('Error', 'The size of bound should be same as size of initial parameter')
            return None

        return list(zip(bd_l, bd_u))

    def run_script(self):

        # check files are loaded
        if len(self.fname) == 0:
            msg.showerror('Error', 'Please read files before fitting')
            return
        
        dargs = []
        kwargs_key = []
        kwargs_val = []
        kwargs = {}

        bound_fwhm = None
        bound_t0 = None
        bound_tau = None
        bound_dmp = None
        bound_osc = None

        # set initial fwhm
        irf = self.irf_var.get()
        fwhm = self.handle_irf()
        if not fwhm:
            return
        # set bound fwhm
        if self.custom_bd_var.get():
            bound_fwhm = self.handle_bd_irf()
        kwargs_key.append('bound_fwhm')

        # set initial t0
        t0 = self.handle_t0()
        if not isinstance(t0, np.ndarray):
            return
        
        if self.custom_bd_var.get():
            bound_t0 = self.handle_bd_field(t0, self.entry_bd_l_t0,
        self.entry_bd_u_t0, 'time zero')
        kwargs_key.append('bound_t0')

        # handle fix_irf option
        if self.fix_irf_var.get():
            if irf in ['g', 'c']:
                bound_fwhm = [(fwhm, fwhm)]
            elif irf == 'pv':
                bound_fwhm = [(fwhm[0], fwhm[0]),
                (fwhm[1], fwhm[1])]
        
        kwargs_val.append(bound_fwhm)

        # handle fix_t0 option
        if self.fix_t0_var.get():
            bound_t0 = t0.size*[None]

            for i in range(t0.size):
                bound_t0[i] = (t0[i], t0[i])
        
        kwargs_val.append(bound_t0)

        mode = self.fit_mode_var.get()

        if mode in ['decay', 'raise', 'both']:
            base = self.base_var.get()
            tau = self.handle_tau()
            if isinstance(tau, np.ndarray) or tau:
                if not isinstance(tau, np.ndarray):
                    tau = None
                elif tau is None and mode == 'raise':
                    msg.showerror('Error', 
                    'Please give initial raise time constant for raising model')
                    return
                else:
                    if self.custom_bd_var.get():
                        bound_tau = self.handle_bd_field(tau,
                        self.entry_bd_l_tau, self.entry_bd_u_tau,
                        'tau')
                    elif self.fix_raise_var.get():
                        for k in tau:
                            bound_tau.append(set_bound_tau(k, fwhm))
                        bound_tau[0] = (tau[0], tau[0])
                    kwargs_key.append('bound_tau')
                    kwargs_val.append(bound_tau)
            else:
                return
            dargs.append(tau)

            if mode in ['decay', 'raise']:
                dargs.append(base)

        if mode in ['dmp_osc', 'both']:
            dmp = self.handle_dmp()
            if not dmp:
                return
            osc = self.handle_osc()
            if not osc:
                return
            if dmp.size != osc.size:
                msg.showerror('Error', 
                'The number of damping constant and oscillation period should be same')
                return
            
            if self.custom_bd_var.get():
                bound_dmp = self.handle_bd_field(dmp, self.entry_bd_l_dmp,
                self.entry_bd_u_dmp, 'dmp life time')
                bound_osc = self.handle_bd_field(osc, self.entry_bd_l_osc,
                self.entry_bd_u_osc, 'osc period')
            
            if mode == 'dmp_osc':
                kwargs_key.append('bound_tau')
                kwargs_val.append(bound_dmp)
                kwargs_key.append('bound_period')
                kwargs_val.append(bound_osc)
            else:
                kwargs_key.append('bound_tau_osc')
                kwargs_val.append(bound_dmp)
                kwargs_key.append('bound_period_osc')
                kwargs_val.append(bound_osc)

            dargs.append(dmp)
            dargs.append(osc)

        if mode == 'both':
            dargs.append(base)

        if not self.glb_opt_var.get():
            glb_opt = None
        else:
            glb_opt = self.glb_opt_var.get()
        
        for k,v in zip(kwargs_key, kwargs_val):
            kwargs[k] = v

        self.result = FITDRIVER[mode](irf, fwhm, t0, *dargs, method_glb=glb_opt,
        **kwargs,
        name_of_dset=np.array(self.fname), t=self.t, intensity=self.intensity, eps=self.eps)

        self.fit = True
        FitReportWidget(self.result)
        return
    
    def save_script(self):
        if self.result is None:
            msg.showerror('Error', 'Please click save button after click run button')
            return
        
        save_folder = fd.askdirectory()
        save_h5_name = save_folder + '/' + save_folder.split('/')[-1]
        save_TransientResult(self.result, save_h5_name)
        
        with open(f'{save_folder}/fit_summary.txt', 'w', encoding='utf-8') as f:
            f.write(str(self.result))
            
        for i in range(len(self.result['t'])):
            coeff_fmt = self.result['eps'][i].shape[1]*['%.8e']
            fit_fmt = (1+self.result['eps'][i].shape[1])*['%.8e']
            coeff_header_lst = []
            fit_header_lst = ['time_delay']
            res_save = np.vstack(
                (self.result['t'][i], self.result['res'][i][:, 0], self.result['eps'][i][:, 0])).T
            np.savetxt(f"{save_folder}/res_{self.result['name_of_dset'][i]}.txt", res_save,
                       fmt=['%.8e', '%.8e', '%.8e'],
                       header=f"time_delay \t res_{self.result['name_of_dset'][i]} \t eps")
            fit_header_lst.append(f"fit_{self.result['name_of_dset'][i]}")
            coeff_header_lst.append(f"tscan_{self.result['name_of_dset'][i]}")
            
            fit_header = '\t'.join(fit_header_lst)
            coeff_header = '\t'.join(coeff_header_lst)
            
            np.savetxt(f"{save_folder}/weight_{self.result['name_of_dset'][i]}.txt", 
            self.result['c'][i], fmt=coeff_fmt, header=coeff_header)
            
            if self.result['model'] in ['dmp_osc', 'both']:
                np.savetxt(f"{save_folder}/phase_{self.result['name_of_dset'][i]}.txt", 
                self.result['phase'][i], fmt=coeff_fmt, header=coeff_header)
            
            fit_save = np.vstack((self.result['t'][i], self.result['fit'][i].T)).T
            np.savetxt(f"{save_folder}/fit_{self.result['name_of_dset'][i]}.txt",
            fit_save, fmt=fit_fmt, header=fit_header)
            
            if self.result['model'] == 'both':
                fit_decay_save = np.vstack((self.result['t'][i], self.result['fit_decay'][i].T)).T
                np.savetxt(f"{save_folder}/fit_decay_{self.result['name_of_dset'][i]}.txt",
                fit_decay_save, fmt=fit_fmt, header=fit_header)
                fit_osc_save = np.vstack(
                    (self.result['t'][i], self.result['fit_osc'][i].T)).T
                np.savetxt(f"{save_folder}/fit_osc_{self.result['name_of_dset'][i]}.txt", 
                fit_osc_save, fmt=fit_fmt, header=fit_header)

    def exit_script(self):
        self.parameter_window.quit()
        self.root.quit()

def fit_tscan_gui():
    FitTscanGuiWidget()



import re
import tkinter as tk
import tkinter.messagebox as msg
import tkinter.filedialog as fd
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure 
from TRXASprefitpack import fit_transient_exp, fit_transient_dmp_osc, fit_transient_both

FITDRIVER = {'decay': fit_transient_exp, 'osc': fit_transient_dmp_osc, 
'both': fit_transient_both}

float_sep_comma = re.compile('([0-9]+[.]?[0-9]*[,]\s*)*[0-9]+[.]?[0-9]*\s*')

class plot_data_widegets:
    # These codes are based on Matplotlib example "Embedding in Tk"

    def __init__(self, master):

        self.top = tk.Toplevel(master.root)
        self.top.title('Plot Data')

        self.fig = Figure(figsize=(5, 4), dpi=100)

        # immutable 
        self.ax = self.fig.add_subplot()
        self.ax.set_xlabel('Time Delay')
        self.ax.set_ylabel('Intensity')

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

        self.canvas.mpl_connect("key_press_event", key_press_handler)

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
        self.canvas.draw()

class plot_fit_widegets:
    # These codes are based on Matplotlib example "Embedding in Tk"

    def __init__(self, master):

        self.top = tk.Toplevel(master.root)
        self.top.title('Plot Fitting Result')

        self.fig = Figure(figsize=(12, 4), dpi=100)

        # immutable 
        # fit
        self.ax_fit = self.fig.add_subplot(211)
        self.ax_fit.set_xlabel('Time Delay')
        self.ax_fit.set_ylabel('Intensity')

        # residual
        self.ax_res = self.fig.add_subplot(212)
        self.ax_res.set_xlabel('Time Delay')
        self.ax_res.set_ylabel('Residual')

        # mutable

        # fit
        self.ax_fit.set_title(f'{master.fname[0]}')
        self.ln_data, = self.ax_fit.plot(master.t[0], master.intensity[0][:, 0], mfc='none', 
        color='black', marker='o')
        self.poly_data = self.ax_fit.fill_between(master.t[0], 
        master.intensity[0][:, 0]-master.eps[0][:, 0],
        master.intensity[0][:, 0]+master.eps[0][:, 0], alpha=0.5, color='black')
        self.ln_fit, = self.ax_fit.plot(master.t[0], master.result['fit'][0][:, 0],
        color='red')

        # residual
        if master.fit_mode_var.get() in ['decay', 'osc']:
            self.ln_res, = self.ax_res.plot(master.t[0], 
            master.intensity[0][:, 0]-master.result['fit'][0][:, 0],
            mfc='none', color='black', marker='o')
            self.poly_res = self.ax_res.fill_between(master.t[0],
            master.intensity[0][:, 0]-master.result['fit'][0][:, 0]-master.eps[0][:, 0],
            master.intensity[0][:, 0]-master.result['fit'][0][:, 0]+master.eps[0][:, 0],
            alpha=0.5, color='black')
        else:
            self.ln_res, = self.ax_res.plot(master.t[0], 
            master.intensity[0][:, 0]-master.result['fit_decay'][0][:, 0],
            mfc='none', color='black', marker='o')
            self.poly_res = self.ax_res.fill_between(master.t[0],
            master.intensity[0][:, 0]-master.result['fit_decay'][0][:, 0]-master.eps[0][:, 0],
            master.intensity[0][:, 0]-master.result['fit_decay'][0][:, 0]+master.eps[0][:, 0],
            alpha=0.5, color='black')
            self.ln_fit, = self.ax_res.plot(master.t[0], master.result['fit_osc'][0][:, 0],
            color='red')
            


        # set canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()

        self.toolbar = \
            NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect("key_press_event", key_press_handler)

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
        self.ln_data.set_data(master.t[val-1], master.intensity[val-1][:, 0])
        self.poly_data.remove()
        self.poly_data = self.ax_fit.fill_between(master.t[val-1], 
        master.intensity[val-1][:, 0]-master.eps[val-1][:, 0],
        master.intensity[val-1][:, 0]+master.eps[val-1][:, 0], alpha=0.5,
        color='black')
        self.ln_fit.set_data(master.t[val-1], master.result['fit'][val-1][:, 0])

        # update residual window

        if master.fit_mode_var.get() in ['decay', 'osc']:
            self.ln_res.set_data(master.t[val-1], 
            master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0])
            self.poly_res.remove()
            self.poly_res = self.ax_res.fill_between(master.t[val-1],
            master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0]-master.eps[val-1][:, 0],
            master.intensity[val-1][:, 0]-master.result['fit'][val-1][:, 0]+master.eps[val-1][:, 0],
            alpha=0.5, color='black')
        else:
            self.ln_res.set_data(master.t[val-1], 
            master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0])
            self.poly_res.remove()
            self.poly_res = self.ax_res.fill_between(master.t[0],
            master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0]-master.eps[val-1][:, 0],
            master.intensity[val-1][:, 0]-master.result['fit_decay'][val-1][:, 0]+master.eps[val-1][:, 0],
            alpha=0.5, color='black')
            self.ln_fit.set_data(master.t[val-1], master.result['fit_osc'][val-1][:, 0])

        self.canvas.draw()


class fit_tscan_gui_widgets:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Fit Tscan Gui')
        self.parameter_window = tk.Tk()
        self.parameter_window.title('Initial fitting parameter')
        self.report_window = tk.Tk()
        self.report_window.title('Fitting Report')

        # -- define necessary variables
        self.irf_var = tk.StringVar()
        self.fit_mode_var = tk.StringVar()
        self.base_var = tk.IntVar()
        self.fix_irf_var = tk.IntVar()
        self.fix_t0_var = tk.IntVar() 
        self.glb_opt_var = tk.StringVar()
        self.fname = []
        
        # --- fitting model selection window
        self.fit_mode_label = tk.Label(self.root, text='Select fitting mode', 
        padx=90, pady=10, font=('Arial', 12))
        self.fit_mode_label.grid(column=0, row=0, columnspan=3)
        self.decay_mode = tk.Checkbutton(self.root, text='decay', 
        variable = self.fit_mode_var, onvalue = 'decay', offvalue='')
        self.decay_mode.grid(column=0, row=1)
        self.dmp_osc_mode = tk.Checkbutton(self.root, text='damped osc', 
        variable = self.fit_mode_var, onvalue = 'dmp_osc', offvalue='')
        self.dmp_osc_mode.grid(column=1, row=1)
        self.both_mode = tk.Checkbutton(self.root, text='decay+dmp_osc', 
        variable = self.fit_mode_var, onvalue = 'both', offvalue='')
        self.both_mode.grid(column=2, row=1)

        # --- irf model selection window
        self.irf_label = tk.Label(self.root, text='Select Type of irf', 
        padx=90, pady=10, font=('Arial', 12))
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
        padx=60, pady=10, font=('Arial', 12))
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
        padx=90, pady=10, font=('Arial', 12))
        self.option_label.grid(column=0, row=6, columnspan=3)
        self.include_base_check = tk.Checkbutton(self.root, text='base', 
        variable=self.base_var, onvalue=1, offvalue=0)
        self.include_base_check.grid(column=0, row=7)
        self.fix_irf_check = tk.Checkbutton(self.root, text='fix_irf', 
        variable=self.fix_irf_var, onvalue=1, offvalue=0)
        self.fix_irf_check.grid(column=1, row=7)
        self.fix_t0_check = tk.Checkbutton(self.root, text='fix_t0',
        variable=self.fix_t0_var, onvalue=1, offvalue=0)
        self.fix_t0_check.grid(column=2, row=7)
        
        # --- Read file to fit
        self.label_file = tk.Label(self.root, text='Browse Files to fit', 
        padx=120, pady=10, font=('Arial', 12))
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
    
        self.ready_button = tk.Button(self.root, 
        text='Parameters', command=self.view_param,
        font=('Arial', 12), bg='green', padx=30, pady=10, bd=5, fg='white')
        self.ready_button.grid(column=0, row=10)
        
        self.run_button = tk.Button(self.root, 
        text='Run', command=self.run_script, 
        font=('Arial', 12), bg='blue', padx=30, pady=10, bd=5, fg='white')
        self.run_button.grid(column=1, row=10)
        
        self.exit_button = tk.Button(self.root, 
        text='Exit', command=self.exit_script, 
        font=('Arial', 12), bg='red', padx=30, pady=10, bd=5, fg='white')
        self.exit_button.grid(column=2, row=10)
        
        # ---- Parameters Depending on fitting Model ----
        self.label_fwhm_G = tk.Label(self.parameter_window, text='fwhm_G (irf)',
        padx=30, pady=10, font=('Arial', 12))
        self.label_fwhm_G.grid(column=0, row=0)
        self.entry_fwhm_G = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_fwhm_G.grid(column=0, row=1)
        
        self.label_fwhm_L = tk.Label(self.parameter_window, text='fwhm_L (irf)',
        padx=30, pady=10, font=('Arial', 12))
        self.label_fwhm_L.grid(column=1, row=0)
        self.entry_fwhm_L = tk.Entry(self.parameter_window, width=10, bd=1)
        self.entry_fwhm_L.grid(column=1, row=1)
        
        self.label_t0 = tk.Label(self.parameter_window, 
        text='Insert initial time zero parameter (t01,t02,...)',
        padx=90, pady=10, font=('Arial', 12))
        self.label_t0.grid(column=0, row=2, columnspan=3)
        self.entry_t0 = tk.Entry(self.parameter_window, width=90, bd=5)
        self.entry_t0.grid(column=0, row=3, columnspan=3)
        
        self.label_tau = tk.Label(self.parameter_window, 
        text='Insert initial life time parameter (tau1,tau2,...)',
        padx=90, pady=10, font=('Arial', 12))
        self.label_tau.grid(column=0, row=4, columnspan=3)
        self.entry_tau = tk.Entry(self.parameter_window, width=90, bd=5)
        self.entry_tau.grid(column=0, row=5, columnspan=3)
        
        self.label_osc = tk.Label(self.parameter_window, 
        text='Insert initial osc period parameter (T_osc1,T_osc2,...)',
        padx=90, pady=10, font=('Arial', 12))
        self.label_osc.grid(column=0, row=6, columnspan=3)
        self.entry_osc = tk.Entry(self.parameter_window, width=90, bd=5)
        self.entry_osc.grid(column=0, row=7, columnspan=3)
        
        self.label_dmp = tk.Label(self.parameter_window, 
        text='Insert initial damping lifetime parameter (dmp1,dmp2,...)',
        padx=90, pady=10, font=('Arial', 12))
        self.label_dmp.grid(column=0, row=8, columnspan=3)
        self.entry_dmp = tk.Entry(self.root, width=90, bd=5)
        self.entry_dmp.grid(column=0, row=9, columnspan=3)

        self.hide_init_param_option()

        # --- canvas for fitting Report
        self.report = tk.Canvas(self.report_window, width=600, height=1200, bg='white')
        self.root.mainloop()
    
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
    
    def plot_file(self):

        if len(self.fname) == 0:
            msg.showerror('Error', 'Please load files')
        else:
            plot_data_widegets(self)


    # --- hide fitting parameter entry
    def hide_init_param_option(self):
        # irf option
        self.label_fwhm_G.grid_remove()
        self.entry_fwhm_G.grid_remove()
        self.label_fwhm_L.grid_remove()
        self.entry_fwhm_L.grid_remove()

        # t0 option
        self.label_t0.grid_remove()
        self.entry_t0.grid_remove()

        # tau option
        self.label_tau.grid_remove()
        self.entry_tau.grid_remove()

        # osc option
        self.label_osc.grid_remove()
        self.entry_osc.grid_remove()

        # dmp option
        self.label_dmp.grid_remove()
        self.entry_dmp.grid_remove()
        
    # --- prepare to fit 
    def view_param(self):
            
        # hide all
        self.hide_init_param_option()
            
        # show irf option
        if self.irf_var.get() == 'g':
            self.label_fwhm_G.grid()
            self.entry_fwhm_G.grid()
        elif self.irf_var.get() == 'c':
            self.label_fwhm_L.grid()
            self.entry_fwhm_L.grid()
        elif self.irf_var.get() == 'pv':
            self.label_fwhm_G.grid()
            self.label_fwhm_L.grid()
            self.entry_fwhm_G.grid()
            self.entry_fwhm_L.grid()
        else:
            msg.showerror('Error', 
            'Please select the type of irf before clicking ready button')
            return
            
        # show t0 option
        self.label_t0.grid()
        self.entry_t0.grid()
            
        # show initial life time related option
        if self.fit_mode_var.get() == 'decay':
            self.label_tau.grid()
            self.entry_tau.grid()
        elif self.fit_mode_var.get() == 'dmp_osc':
            self.label_osc.grid()
            self.entry_osc.grid()
            self.label_dmp.grid()
            self.entry_dmp.grid()
        elif self.fit_mode_var.get() == 'both':
            self.label_tau.grid()
            self.entry_tau.grid()
            self.label_osc.grid()
            self.entry_osc.grid()
            self.label_dmp.grid()
            self.entry_dmp.grid()
        else:
            msg.showerror('Error', 
            'Please select the fitting model before clicking ready button')
        return
    
    def handle_irf(self):
        if self.irf_var.get() == 'g':
            if self.entry_fwhm_G.get():
                fwhm_G_init = self.entry_fwhm_G.get()
                if fwhm_G_init.isnumeric():
                    fwhm = float(fwhm_G_init)
                else:
                    msg.showerror('Error',
                    'fwhm_G should be single float number.')
                    return False
            else:
                msg.showerror('Error', 'Please enter initial fwhm_G value')
                return False
        elif self.irf_var.get() == 'c':
            if self.entry_fwhm_L.get():
                fwhm_L_init = self.entry_fwhm_L.get()
                if fwhm_L_init.isnumeric():
                    fwhm = float(fwhm_L_init)
                else:
                    msg.showerror('Error',
                    'fwhm_L should be single float number')
                    return False
            else:
                msg.showerror('Error', 'Please enter initial fwhm_L value')
                return False
        elif self.irf_var.get() == 'pv':
            if self.entry_fwhm_G.get() and self.entry_fwhm_L.get():
                fwhm_G_init = self.entry_fwhm_G.get()
                fwhm_L_init = self.entry_fwhm_L.get()
                if fwhm_G_init.isnumeric() and fwhm_L_init.isnumeric():
                    fwhm = [float(fwhm_G_init), float(fwhm_L_init)]
                else:
                    msg.showerror('Error',
                    'Both fwhm_G and fwhm_L field should be single float number')
                    return False
            else:
                msg.showerror('Error',
                'Please enter both initial fwhm_G and fwhm_L values')
                return False
        return fwhm
    
    def handle_t0(self):
        if self.entry_t0.get():
            str_t0 = self.entry_t0.get()
            if float_sep_comma.match(str_t0):
                t0 = np.array(list(map(float, str_t0.split(','))))
                if t0.size != len(self.file_lst):
                    msg.showerror('Error',
                    'Number of initial time zero should be same as number of files to fit.')
                    return False
            else:
                msg.showerror('Error',
                'initial time zero should be single float or floats seperated by comma.')
                return False
        else:
            msg.showerror('Error',
            'Please enter initial time zero for each files')
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
            if self.base_var:
                return True
            else:
                msg.showerror('Error',
                'Please enter initial life time constant')
                return False
        return tau
    
    def handle_osc(self):
        if self.entry_osc.get():
            str_osc = self.entry_osc.get()
            if float_sep_comma.match(str_osc):
                osc = np.array(list(map(float, str_osc.split(','))))
            else:
                msg.showerror('Error',
                'initial oscillation period should be single float or floats seperated by comma.')
                return False
        else:
            msg.showerror('Error',
            'Please enter initial oscillation period')
            return False
        return osc

    def handle_dmp(self):
        if self.entry_dmp.get():
            str_dmp = self.entry_dmp.get()
            if float_sep_comma.match(str_dmp):
                dmp = np.array(list(map(float, str_dmp.split(','))))
            else:
                msg.showerror('Error',
                'initial oscillation period should be single float or floats seperated by comma.')
                return False
        else:
            msg.showerror('Error',
            'Please enter initial oscillation period')
            return False
        return dmp   
    
    def run_script(self):

        self.report.delete('all')

        # check files are loaded 
        if len(self.fname) == 0:
            msg.showerror('Error', 'Please read files before fitting')


        # set initial fwhm
        irf = self.irf_var.get()
        fwhm = self.handle_irf()
        if not fwhm:
            return

        # set initial t0
        t0 = self.handle_t0()
        if not isinstance(t0, np.ndarray):
            return
        
        # handle fix_irf option
        bound_fwhm = None
        if self.fix_irf_var.get():
            if irf in ['g', 'c']:
                bound_fwhm = [(fwhm, fwhm)]
            elif irf == 'pv':
                bound_fwhm = [(fwhm[0], fwhm[0]),
                (fwhm[1], fwhm[1])]
        
        # handle fix_t0 option
        bound_t0 = None
        if self.fix_t0_var.get():
            bound_t0 = t0.size*[None]
            
            for i in range(t0.size):
                bound_t0[i] = (t0[i], t0[i])
        
        dargs = []
        mode = self.fit_mode_var.get()
        
        if mode in ['decay', 'both']:
            base = self.base_var.get()
            tau = self.handle_tau()
            if isinstance(tau, np.ndarray) or tau:
                if not isinstance(tau, np.ndarray):
                    tau = None
            else:
                return
            dargs.append(tau)
            if mode == 'decay':
                dargs.append(base)
        
        if mode in ['osc', 'both']:
            dmp = self.handle_dmp()
            if not dmp:
                return
            osc = self.handle_osc()
            if not osc:
                return
            if dmp.size != osc.size:
                msg.showerror('Error', 'The number of damping constant and oscillation period should be same')
                return
            
            dargs.append(dmp)
            dargs.append(osc)
        
        if mode == 'both':
            dargs.append(base)
        
        if not self.glb_opt_var.get():
            glb_opt = None
        else:
            glb_opt = self.glb_opt_var.get()
        
        self.result = FITDRIVER[mode](irf, fwhm, t0, *dargs, method_glb=glb_opt,
        bound_fwhm=bound_fwhm, bound_t0=bound_t0,
        name_of_dset=self.fname, t=self.t, intensity=self.intensity, eps=self.eps)

        self.report.create_text(400, 800, text=self.result) # FIX NEEDED

        plot_fit_widegets(self)
        
        return
    
    def exit_script(self):
        self.root.quit()
        self.parameter_window.quit()

if __name__ == '__main__':
    fit_tscan_gui_widgets()

import tkinter as tk
import tkinter.messagebox as msg

root = tk.Tk()
root.title('Fit Tscan Gui')

result_window = tk.Tk()
result_window.title('Fitting Report')

canvas = tk.Canvas(result_window, width=300, height=600)
canvas.pack()

irf_var = tk.StringVar()
fit_mode_var = tk.StringVar()
base_var = tk.IntVar()
fix_irf_var = tk.IntVar()

# --- Select fitting model

fit_mode_label = tk.Label(root, text='Select fitting mode', padx=50, pady=10, font=('Arial', 12))
fit_mode_label.grid(column=0, row=0, columnspan=3)
decay_mode = tk.Checkbutton(root, text='decay', variable = fit_mode_var,
onvalue = 'decay', offvalue='', font=('Arial', 10))
decay_mode.grid(column=0, row=1)
dmp_osc_mode = tk.Checkbutton(root, text='damped osc', variable = fit_mode_var,
onvalue = 'dmp_osc', offvalue='', font=('Arial', 10))
dmp_osc_mode.grid(column=1, row=1)
both_mode = tk.Checkbutton(root, text='decay+dmp_osc', variable = fit_mode_var,
onvalue = 'both', offvalue='', font=('Arial', 10))
both_mode.grid(column=2, row=1)


irf_label = tk.Label(root, text='Select Type of irf', padx=50, pady=10, font=('Arial', 12))
irf_label.grid(column=0, row=2, columnspan=3)
irf_g = tk.Checkbutton(root, text='gaussian', variable=irf_var, 
onvalue='g', offvalue='')
irf_g.grid(column=0, row=3)
irf_c = tk.Checkbutton(root, text='cauchy', variable=irf_var, 
onvalue='c', offvalue='')
irf_c.grid(column=1, row=3)
irf_pv = tk.Checkbutton(root, text='pseudo voigt', variable=irf_var, 
onvalue='pv', offvalue='')
irf_pv.grid(column=2, row=3)

option_label = tk.Label(root, text='Other Options', padx=50, pady=10, font=('Arial', 12))
option_label.grid(column=0, row=4, columnspan=3)
include_base_check = tk.Checkbutton(root, text='base', variable=base_var, onvalue=1, offvalue=0)
include_base_check.grid(column=0, row=5)
fix_irf_check = tk.Checkbutton(root, text='fix_irf', variable=fix_irf_var, onvalue=1, offvalue=0)
fix_irf_check.grid(column=1, row=5)

# --- Read file to fit

label_file_path = tk.Label(root, text='Type root directory of file', padx=100, pady=10,
font=('Arial', 12))
label_file_path.grid(column=0, row=6, columnspan=3)
entry_file_path = tk.Entry(root, width=50, bd=5)
entry_file_path.grid(column=0, row=7, columnspan=3)

label_file_prefix = tk.Label(root, text='Prefix of filename to fit (pre,per,...)',
padx=100, pady=10, font=('Arial', 12))
label_file_prefix.grid(column=0, row=8, columnspan=3)
entry_file_prefix = tk.Entry(root, width=50, bd=5)
entry_file_prefix.grid(column=0, row=9, columnspan=3)

label_num_file = tk.Label(root, text='Number of file to fit (num1,num2,...)',
padx=100, pady=10, font=('Arial', 12))
label_num_file.grid(column=0, row=10, columnspan=3)
entry_num_file = tk.Entry(root, width=50, bd=5)
entry_num_file.grid(column=0, row=11, columnspan=3)

# ---- Parameters Depending on fitting Model ----

label_fwhm_G = tk.Label(root, text='fwhm_G (irf)',
padx=100, pady=10, font=('Arial', 12))
label_fwhm_G.grid(column=0, row=12)
entry_fwhm_G = tk.Entry(root, width=10, bd=1)
entry_fwhm_G.grid(column=0, row=13)



label_fwhm_L = tk.Label(root, text='fwhm_L (irf)',
padx=100, pady=10, font=('Arial', 12))
label_fwhm_L.grid(column=1, row=12)
entry_fwhm_L = tk.Entry(root, width=10, bd=1)
entry_fwhm_L.grid(column=1, row=13)



label_t0 = tk.Label(root, text='Insert initial time zero parameter (t01,t02,...)',
padx=100, pady=10, font=('Arial', 12))
label_t0.grid(column=0, row=14, columnspan=3)
entry_t0 = tk.Entry(root, width=50, bd=5)
entry_t0.grid(column=0, row=15, columnspan=3)



label_tau = tk.Label(root, text='Insert initial life time parameter (tau1,tau2,...)',
padx=100, pady=10, font=('Arial', 12))
label_tau.grid(column=0, row=16, columnspan=3)
entry_tau = tk.Entry(root, width=50, bd=5)
entry_tau.grid(column=0, row=17, columnspan=3)



label_osc = tk.Label(root, text='Insert initial osc period parameter (T_osc1,T_osc2,...)',
padx=100, pady=10, font=('Arial', 12))
label_osc.grid(column=0, row=18, columnspan=3)
entry_osc = tk.Entry(root, width=50, bd=5)
entry_osc.grid(column=0, row=19, columnspan=3)



label_dmp = tk.Label(root, text='Insert initial damping lifetime parameter (dmp1,dmp2,...)',
padx=100, pady=10, font=('Arial', 12))
label_dmp.grid(column=0, row=20, columnspan=3)
entry_dmp = tk.Entry(root, width=50, bd=5)
entry_dmp.grid(column=0, row=21, columnspan=3)



# in default options for inital parameters are not shown to user

def hide_init_param_option():
    label_fwhm_G.grid_remove()
    entry_fwhm_G.grid_remove()
    label_fwhm_L.grid_remove()
    entry_fwhm_L.grid_remove()
    label_t0.grid_remove()
    entry_t0.grid_remove()
    label_tau.grid_remove()
    entry_tau.grid_remove()
    label_osc.grid_remove()
    entry_osc.grid_remove()
    label_dmp.grid_remove()
    entry_dmp.grid_remove()


hide_init_param_option()

def ready_script():

    # hide all
    hide_init_param_option()

    # show irf option
    if irf_var.get() == 'g':
        label_fwhm_G.grid()
        entry_fwhm_G.grid()
    elif irf_var.get() == 'c':
        label_fwhm_L.grid()
        entry_fwhm_L.grid()
    elif irf_var.get() == 'pv':
        label_fwhm_G.grid()
        label_fwhm_L.grid()
        entry_fwhm_G.grid()
        entry_fwhm_L.grid()
    else:
        msg.showerror('Error', 'Please select the type of irf before clicking ready button')
        return

    
    # show t0 option
    label_t0.grid()
    entry_t0.grid()

    # show initial life time related option

    if fit_mode_var.get() == 'decay':
        label_tau.grid()
        entry_tau.grid()
    elif fit_mode_var.get() == 'dmp_osc':
        label_osc.grid()
        entry_osc.grid()
        label_dmp.grid()
        entry_dmp.grid()
    elif fit_mode_var.get() == 'both':
        label_tau.grid()
        entry_tau.grid()
        label_osc.grid()
        entry_osc.grid()
        label_dmp.grid()
        entry_dmp.grid()
    else:
        msg.showerror('Error', 'Please select the fitting model before clicking ready button')
        return

def check_sanity():
    if irf_var.get() == 'g' and not (entry_fwhm_G.get()):
        msg.showerror('Error', 'Please set initial fwhm_G parameter')
        return False

    if irf_var.get() == 'c' and not (entry_fwhm_L.get()):
        msg.showerror('Error', 'Please set initial fwhm_L parameter')
        return False

    if irf_var.get() == 'pv' and not ((entry_fwhm_G.get()) or entry_fwhm_L.get()):
        msg.showerror('Error', 'Please set initial fwhm_G, fwhm_L parameter')
        return False
    
    if not (entry_t0.get()):
        msg.showerror('Error', 'Please set inital time zero parameter')
        return False
    
    if fit_mode_var.get() == 'decay' and not ((entry_tau.get()) or base_var.get()):
        msg.showerror('Error', 'Please set inital life time parameter')
        return False
    
    if fit_mode_var.get() == 'dmp_osc' and \
        not ((entry_dmp.get()) and (entry_osc.get())):
        msg.showerror('Error', 'Please set both initial damping and period parameter')
        return False

    if fit_mode_var.get() == 'both' and \
        not (((entry_tau.get()) or base_var.get()) and (entry_dmp.get()) and (entry_osc.get())):
        msg.showerror('Error', 'Please set initial tau for decay component and damping, period for damping component')
        return False


def run_script():
    canvas.delete('all')
    if not check_sanity():
        return
    canvas.create_text(50, 50, text=fit_mode_var.get()+' '+irf_var.get())

def exit_script():
    result_window.quit()
    root.quit()

ready_button = tk.Button(root, text='Ready', command=ready_script,
font=('Arial', 12), bg='green', padx=10, pady=10, bd=5, fg='white')
ready_button.grid(column=0, row=22)

run_button = tk.Button(root, text='Run', command=run_script, 
font=('Arial', 12), bg='blue', padx=10, pady=10, bd=5, fg='white')
run_button.grid(column=1, row=22)

exit_button = tk.Button(root, text='Exit', command=exit_script, 
font=('Arial', 12), bg='red', padx=10, pady=10, bd=5, fg='white')
exit_button.grid(column=2, row=22)

root.mainloop()
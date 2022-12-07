import tkinter as tk
import tkinter.ttk as ttk



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

fit_mode_label = tk.Label(root, text='Select fitting mode', padx=50, pady=10, font=('Arial', 12))
decay_mode = tk.Checkbutton(root, text='decay', variable = fit_mode_var,
onvalue = 'decay', offvalue='')
dmp_osc_mode = tk.Checkbutton(root, text='damped osc', variable = fit_mode_var,
onvalue = 'dmp osc', offvalue='')
both_mode = tk.Checkbutton(root, text='decay+dmp_osc', variable = fit_mode_var,
onvalue = 'both', offvalue='')

# pack botton (fit mode)

fit_mode_label.pack()
decay_mode.pack()
dmp_osc_mode.pack()
both_mode.pack()


irf_label = tk.Label(root, text='Select Type of irf', padx=50, pady=10, font=('Arial', 12))
irf_g = tk.Checkbutton(root, text='gaussian', variable=irf_var, 
onvalue='g', offvalue='')
irf_c = tk.Checkbutton(root, text='cauchy', variable=irf_var, 
onvalue='c', offvalue='')
irf_pv = tk.Checkbutton(root, text='pseudo voigt', variable=irf_var, 
onvalue='pv', offvalue='')

# pack botton (irf)
irf_label.pack()
irf_g.pack()
irf_c.pack()
irf_pv.pack()

option_label = tk.Label(root, text='Other Options', padx=50, pady=10, font=('Arial', 12))
include_base = tk.Checkbutton(root, text='base', variable=base_var, onvalue=1, offvalue=0)
fix_irf_check = tk.Checkbutton



def run_script():
    canvas.delete('all')
    canvas.create_text(50, 50, text=fit_mode_var.get()+' '+irf_var.get())

def exit_script():
    result_window.quit()
    root.quit()

label_file_prefix = tk.Label(root, text='Prefix of filename to fit (pre,per,...)')
label_file_prefix.pack()
entry_file_prefix = tk.Entry(root)
entry_file_prefix.pack()

label_num_file = tk.Label(root, text='Number of file to fit (num1,num2,...)')
label_num_file.pack()
entry_num_file = tk.Entry(root)
entry_num_file.pack()

run_button = tk.Button(root, text='Run', command=run_script, 
font=('Arial', 20), bg='blue', padx=15, pady=15, bd=10, fg='white')
run_button.pack()

exit_button = tk.Button(root, text='Exit', command=exit_script, 
font=('Arial', 20), bg='red', padx=15, pady=15, bd=10, fg='white')
exit_button.pack()



root.mainloop()
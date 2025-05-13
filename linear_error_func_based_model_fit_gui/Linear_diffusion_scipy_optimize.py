import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from fpdf import FPDF
import os
import time

class DiffusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Slab Diffusion Coefficient Estimator")

        # Slab half-thickness in mm (user-editable)
        self.l_var = tk.DoubleVar(value=10.0)  # default 10 mm

        # Initial guesses
        self.D_init  = 7e-10   # m^2/s
        self.C0_init = 0.01414 # mol/L

        # Default data (multiple time snapshots)
        self.default_times         = [30, 60, 120]  # minutes
        self.default_distances     = [11, 13, 15, 17, 19]  # mm
        self.default_concentrations = [
            [0.0119, 0.0014, 0.0006, 0.0003, 0.0003],  # 30 min
            [0.0118, 0.0049, 0.0009, 0.0005, 0.0004],  # 60 min
            [0.0120, 0.0069, 0.0020, 0.0010, 0.0004]   # 120 min
        ]

        self.time_points = tk.IntVar(value=len(self.default_times))
        self.dist_points = tk.IntVar(value=len(self.default_distances))

        self.time_entries = []
        self.dist_entries = []
        self.conc_entries = []

        self.setup_ui()

    def setup_ui(self):
        control = ttk.Frame(self.root, padding="10")
        control.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Time controls
        tf = ttk.Frame(control)
        tf.grid(row=0, column=0, padx=5)
        ttk.Label(tf, text="Time snapshots:").grid(row=0, column=0)
        ttk.Spinbox(tf, from_=1, to=10, width=5, textvariable=self.time_points, command=self.generate_inputs).grid(row=0, column=1)
        ttk.Button(tf, text="+", width=3, command=self.add_time).grid(row=0, column=2)
        ttk.Button(tf, text="-", width=3, command=self.remove_time).grid(row=0, column=3)

        # Distance controls
        df = ttk.Frame(control)
        df.grid(row=0, column=1, padx=5)
        ttk.Label(df, text="Depths (mm):").grid(row=0, column=0)
        ttk.Spinbox(df, from_=1, to=10, width=5, textvariable=self.dist_points, command=self.generate_inputs).grid(row=0, column=1)
        ttk.Button(df, text="+", width=3, command=self.add_distance).grid(row=0, column=2)
        ttk.Button(df, text="-", width=3, command=self.remove_distance).grid(row=0, column=3)

        # Slab half-thickness control
        rf = ttk.Frame(control)
        rf.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(rf, text="Slab half-thickness (mm):").grid(row=0, column=0)
        ttk.Entry(rf, width=8, textvariable=self.l_var).grid(row=0, column=1)

        # Input frame
        self.input_frame = ttk.Frame(self.root, padding="10")
        self.input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.generate_inputs()

        # Progress bar
        pf = ttk.Frame(self.root, padding="10 0")
        pf.grid(row=2, column=0, sticky=(tk.W, tk.E))
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(pf, orient="horizontal", length=400, mode="determinate", variable=self.progress_var).grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.progress_label = ttk.Label(pf, text="")
        self.progress_label.grid(row=1, column=0, sticky=tk.W)

        # Buttons
        bf = ttk.Frame(self.root, padding="10")
        bf.grid(row=3, column=0, sticky=(tk.W, tk.E))
        ttk.Button(bf, text="Run Fit",   command=self.run_fit).grid(row=0, column=0, padx=5)
        ttk.Button(bf, text="Clear All", command=self.clear_all).grid(row=0, column=1, padx=5)

    def add_time(self):
        self.time_points.set(self.time_points.get() + 1)
        self.generate_inputs()
    def remove_time(self):
        if self.time_points.get()>1:
            self.time_points.set(self.time_points.get() - 1)
            self.generate_inputs()
    def add_distance(self):
        self.dist_points.set(self.dist_points.get() + 1)
        self.generate_inputs()
    def remove_distance(self):
        if self.dist_points.get()>1:
            self.dist_points.set(self.dist_points.get() - 1)
            self.generate_inputs()
    def clear_all(self):
        self.time_points.set(1)
        self.dist_points.set(1)
        self.generate_inputs()

    def generate_inputs(self):
        for w in self.input_frame.winfo_children(): w.destroy()
        self.time_entries.clear()
        self.dist_entries.clear()
        self.conc_entries.clear()

        # Headers
        ttk.Label(self.input_frame, text="Time (min)", font=('Arial',10,'bold')).grid(row=0, column=0, padx=5, pady=5)
        for d in range(self.dist_points.get()):
            ttk.Label(self.input_frame, text=f"Depth {d+1} (mm)", font=('Arial',10,'bold')).grid(row=0, column=d+1, padx=5)
        ttk.Label(self.input_frame, text="Concentration →", font=('Arial',9)).grid(row=1, column=0, sticky=tk.W)

        # Depth row
        for d in range(self.dist_points.get()):
            e = ttk.Entry(self.input_frame, width=8)
            if d < len(self.default_distances):
                e.insert(0, str(self.default_distances[d]))
            e.grid(row=1, column=d+1, padx=2)
            self.dist_entries.append(e)

        # Time & concentration rows
        for t in range(self.time_points.get()):
            te = ttk.Entry(self.input_frame, width=8)
            if t < len(self.default_times):
                te.insert(0, str(self.default_times[t]))
            te.grid(row=t+2, column=0, padx=2)
            self.time_entries.append(te)

            crow = []
            for d in range(self.dist_points.get()):
                ce = ttk.Entry(self.input_frame, width=8)
                if t < len(self.default_concentrations) and d < len(self.default_concentrations[t]):
                    ce.insert(0, str(self.default_concentrations[t][d]))
                ce.grid(row=t+2, column=d+1, padx=2)
                crow.append(ce)
            self.conc_entries.append(crow)

    def diffusion_errorfunction(self, z, D, C0, t, l):
        return 0.5 * C0 * (erf((l - z) / np.sqrt(4 * D * t)) + erf((l + z) / np.sqrt(4 * D * t)))

    def run_fit(self):
        try:
            D_vals  = []
            C0_vals = []

            for e in self.time_entries + self.dist_entries + sum(self.conc_entries, []):
                if not e.get().strip():
                    messagebox.showwarning("Missing Data","Fill in all fields.")
                    return

            depths = np.array([float(e.get()) for e in self.dist_entries]) / 1000
            times  = np.array([float(e.get()) for e in self.time_entries]) * 60
            C_matrix = np.array([[float(e.get()) for e in row] for row in self.conc_entries])
            l = self.l_var.get() / 1000

            # Prepare output directory & PDF
            out_dir = os.path.join(os.getcwd(), 'output')
            os.makedirs(out_dir, exist_ok=True)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", '', 12)
            pdf.cell(200,10,txt="Slab Diffusion Analysis Results",ln=True,align='C')
            pdf.cell(200,10,txt=f"Date: {time.strftime('%Y-%m-%d')}",ln=True)
            pdf.ln(5)

            total = len(times)
            for i, t in enumerate(times):
                self.progress_var.set(50 * i / total)
                self.progress_label.config(text=f"Fitting t={t/60:.0f} min ({i+1}/{total})")
                self.root.update_idletasks()

                y_data = C_matrix[i]
                popt, pcov = curve_fit(lambda z, D, C0: self.diffusion_errorfunction(z,D,C0,t,l),
                                        depths, y_data, p0=[self.D_init, self.C0_init])
                D_est, C0_est = popt
                D_err, C0_err = np.sqrt(np.diag(pcov))
                D_vals.append(D_est)
                C0_vals.append(C0_est)

                # Plot and save
                plt.figure()
                plt.plot(depths*1000, y_data, 'o', label='Data')
                z_fit = np.linspace(depths.min(), depths.max(), 200)
                plt.plot(z_fit*1000, self.diffusion_errorfunction(z_fit,D_est,C0_est,t,l), '-',
                         label=f"D={D_est:.2e}±{D_err:.2e}, C0={C0_est:.3f}±{C0_err:.3f}")
                plt.xlabel('Depth z (mm)')
                plt.ylabel('C (mol/L)')
                plt.legend()
                fp = os.path.join(out_dir, f'time_{i}.png')
                plt.savefig(fp, dpi=150)
                plt.close()

                # Add to PDF
                pdf.cell(200,10,txt=f"Time {t/60:.0f} min:",ln=True)
                pdf.cell(200,10,txt=f"  D = {D_est:.3e} ± {D_err:.3e} m²/s",ln=True)
                pdf.cell(200,10,txt=f"  C0 = {C0_est:.4f} ± {C0_err:.4f}",ln=True)
                pdf.image(fp, w=150)
                pdf.ln(5)

            # Summary page
            pdf.add_page()
            pdf.set_font("Arial", '', 12)
            pdf.cell(200,10,txt="Overall Averages",ln=True,align='C')
            pdf.ln(5)
            meanD, stdD = np.mean(D_vals), np.std(D_vals)
            meanC0, stdC0 = np.mean(C0_vals), np.std(C0_vals)
            pdf.cell(200,10,txt=f"Average D = {meanD:.3e} ± {stdD:.3e} m²/s",ln=True)
            pdf.cell(200,10,txt=f"Average C0 = {meanC0:.4f} ± {stdC0:.4f}",ln=True)

            # Save PDF
            out_pdf = os.path.join(out_dir, 'fit_summary.pdf')
            pdf.output(out_pdf)
            self.progress_var.set(100)
            self.progress_label.config(text="Done")
            messagebox.showinfo("Success", f"Saved: {out_pdf}")

        except Exception as e:
            messagebox.showerror("Fit Error", str(e))

if __name__ == '__main__':
    root = tk.Tk()
    app = DiffusionGUI(root)
    root.geometry("600x500")
    root.mainloop()

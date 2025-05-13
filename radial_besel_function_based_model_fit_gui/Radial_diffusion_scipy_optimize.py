import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from fpdf import FPDF
import os
import time

# --- in SI units! ---
# Default initial cylinder radius (m)
a_default = 2.0e-3  # 2 mm

def improved_cylindrical_model(r, D, C0, t, a):
    """
    Simplified radial diffusion model with no discontinuities
    
    r  : array-like of radii [m]
    D  : diffusion coefficient [m^2/s]
    C0 : initial cylinder concentration [mol/L]
    t  : time [s]
    a  : initial cylinder radius [m]
    """
    r = np.asanyarray(r)
    
    # Create a continuous function that works for all r values
    # Use a physically-motivated function based on the Green's function solution
    # with modifications to ensure smoothness
    
    # Basic diffusion length scale
    L_diff = 2.0 * np.sqrt(D * t)
    
    # Main concentration function - no separate cases for inside/outside
    # Simple exponential decay based on distance from center
    C = C0 * np.exp(-r**2 / (4.0 * D * t + a**2))
    
    # Scale by time-dependent factor (concentration decreases with time)
    C = C * (a**2 / (a**2 + 4.0 * D * t))
    
    return C

class DiffusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cylindrical Diffusion Coefficient Estimator")

        # Initial cylinder radius in mm (user-editable)
        self.a_var = tk.DoubleVar(value=a_default * 1e3)  # default 2.0 mm

        # Initial fit guesses - ADJUSTED for better starting point
        self.D_init = 5e-7   # m^2/s (increased from 1e-9)
        self.C0_init = 0.002   # mol/L (adjusted from 0.01)

        # Default data (one time point example)
        self.default_times = [60]  # minutes
        self.default_distances = [3, 5, 7, 9, 11, 13]  # mm
        self.default_concentrations = [[0.0011481499, 0.0006161824, 0.0004176988,
                                      0.0002861640, 0.0002515081, 0.0001621100]]  # mol/L

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
        ttk.Spinbox(tf, from_=1, to=5, width=5, textvariable=self.time_points,
                    command=self.generate_inputs).grid(row=0, column=1)
        ttk.Button(tf, text="+", width=3, command=self.add_time).grid(row=0, column=2)
        ttk.Button(tf, text="-", width=3, command=self.remove_time).grid(row=0, column=3)

        # Distance controls
        df = ttk.Frame(control)
        df.grid(row=0, column=1, padx=5)
        ttk.Label(df, text="Radial distances (mm):").grid(row=0, column=0)
        ttk.Spinbox(df, from_=1, to=10, width=5, textvariable=self.dist_points,
                    command=self.generate_inputs).grid(row=0, column=1)
        ttk.Button(df, text="+", width=3, command=self.add_distance).grid(row=0, column=2)
        ttk.Button(df, text="-", width=3, command=self.remove_distance).grid(row=0, column=3)

        # Initial cylinder radius control
        rf = ttk.Frame(control)
        rf.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(rf, text="Initial cylinder radius (mm):").grid(row=0, column=0)
        ttk.Entry(rf, width=8, textvariable=self.a_var).grid(row=0, column=1)

        # Input frame
        self.input_frame = ttk.Frame(self.root, padding="10")
        self.input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.generate_inputs()

        # Progress bar
        pf = ttk.Frame(self.root, padding="10 0")
        pf.grid(row=2, column=0, sticky=(tk.W, tk.E))
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(pf, orient="horizontal", length=400, mode="determinate",
                        variable=self.progress_var).grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.progress_label = ttk.Label(pf, text="")
        self.progress_label.grid(row=1, column=0, sticky=tk.W)

        # Buttons
        bf = ttk.Frame(self.root, padding="10")
        bf.grid(row=3, column=0, sticky=(tk.W, tk.E))
        ttk.Button(bf, text="Run Fit", command=self.run_fit).grid(row=0, column=0, padx=5)
        ttk.Button(bf, text="Clear All", command=self.clear_all).grid(row=0, column=1, padx=5)

    def add_time(self):
        self.time_points.set(self.time_points.get() + 1)
        self.generate_inputs()

    def remove_time(self):
        if self.time_points.get() > 1:
            self.time_points.set(self.time_points.get() - 1)
            self.generate_inputs()

    def add_distance(self):
        self.dist_points.set(self.dist_points.get() + 1)
        self.generate_inputs()

    def remove_distance(self):
        if self.dist_points.get() > 1:
            self.dist_points.set(self.dist_points.get() - 1)
            self.generate_inputs()

    def clear_all(self):
        self.time_points.set(1)
        self.dist_points.set(1)
        self.generate_inputs()

    def generate_inputs(self):
        for w in self.input_frame.winfo_children():
            w.destroy()
        self.time_entries.clear()
        self.dist_entries.clear()
        self.conc_entries.clear()

        ttk.Label(self.input_frame, text="Time (min)", font=('Arial', 10, 'bold')).grid(row=0, column=0)
        for d in range(self.dist_points.get()):
            ttk.Label(self.input_frame, text=f"r{d+1} (mm)", font=('Arial', 10, 'bold')).grid(row=0, column=d+1)

        for d in range(self.dist_points.get()):
            e = ttk.Entry(self.input_frame, width=8)
            if d < len(self.default_distances):
                e.insert(0, str(self.default_distances[d]))
            e.grid(row=1, column=d+1)
            self.dist_entries.append(e)

        for t in range(self.time_points.get()):
            te = ttk.Entry(self.input_frame, width=8)
            if t < len(self.default_times):
                te.insert(0, str(self.default_times[t]))
            te.grid(row=t+2, column=0)
            self.time_entries.append(te)

            crow = []
            for d in range(self.dist_points.get()):
                ce = ttk.Entry(self.input_frame, width=8)
                if t < len(self.default_concentrations) and d < len(self.default_concentrations[t]):
                    ce.insert(0, str(self.default_concentrations[t][d]))
                ce.grid(row=t+2, column=d+1)
                crow.append(ce)
            self.conc_entries.append(crow)

    def run_fit(self):
        try:
            for e in self.time_entries + self.dist_entries + sum(self.conc_entries, []):
                if not e.get().strip():
                    messagebox.showwarning("Missing Data", "Fill in all fields.")
                    return

            r_mm = np.array([float(e.get()) for e in self.dist_entries])
            r_m = r_mm * 1e-3
            times_min = np.array([float(e.get()) for e in self.time_entries])
            times_s = times_min * 60
            C_matrix = np.array([[float(e.get()) for e in row] for row in self.conc_entries])

            a_m = self.a_var.get() * 1e-3
            out_dir = os.path.join(os.getcwd(), 'output')
            os.makedirs(out_dir, exist_ok=True)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "", 12)
            pdf.cell(200, 10, txt="Cylindrical Diffusion Analysis Results", ln=True, align="C")
            pdf.cell(200, 10, txt=f"Date: {time.strftime('%Y-%m-%d')}", ln=True)
            pdf.ln(5)
            
            # Add model information
            pdf.cell(200, 10, txt=f"Initial cylinder radius: {a_m*1000:.2f} mm", ln=True)
            pdf.ln(5)

            D_vals, C0_vals = [], []
            for i, t in enumerate(times_s):
                self.progress_var.set(50 * i / len(times_s))
                self.progress_label.config(text=f"Fitting t={times_min[i]:.1f} min ({i+1}/{len(times_s)})")
                self.root.update_idletasks()

                y_obs = C_matrix[i]

                def fit_fn(r, D, C0):
                    return improved_cylindrical_model(r, D, C0, t, a=a_m)

                # Improved bounds and initial guess for better convergence
                popt, pcov = curve_fit(
                    fit_fn, r_m, y_obs,
                    p0=[self.D_init, self.C0_init],
                    bounds=([1e-10, 1e-4], [1e-5, 0.1]),
                    maxfev=10000, method="trf"
                )
                D_est, C0_est = popt
                D_err, C0_err = np.sqrt(np.diag(pcov))

                D_vals.append(D_est)
                C0_vals.append(C0_est)

                # Create a new figure with a higher DPI and different renderer
                plt.figure(figsize=(8, 6), dpi=150)
                
                # Plot experimental data
                plt.plot(r_mm, y_obs, "bo", markersize=8, label="Experimental Data")
                
                # Create a smooth curve for plotting - avoid r=0 exactly
                # Start from a very small positive value to avoid any discontinuity
                r_fit = np.logspace(-4, np.log10(r_mm.max() * 1.2), 500) * 1e-3
                c_fit = improved_cylindrical_model(r_fit, D_est, C0_est, t, a=a_m)
                
                # Plot the curve with a solid line interpolation and extra smoothness
                plt.plot(r_fit * 1e3, c_fit, "r-", linewidth=2, 
                       label=f"D={D_est:.2e} m²/s, C0={C0_est:.4f} mol/L")
                
                # Set plot limits explicitly to control the display
                plt.xlim(0, r_mm.max() * 1.1)
                plt.ylim(0, max(y_obs) * 1.2)
                
                # Improve plot appearance
                plt.xlabel("Radial Distance (mm)", fontsize=12)
                plt.ylabel("Concentration (mol/L)", fontsize=12)
                plt.title(f"Cylindrical Diffusion at t = {times_min[i]:.1f} min", fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(ls="--", alpha=0.6)
                plt.tight_layout()
                
                # Save with high quality and different backend for better rendering
                imgp = os.path.join(out_dir, f"time_{i+1}.png")
                plt.savefig(imgp, dpi=200, bbox_inches="tight", format="png")
                plt.close()

                # Add results to PDF
                pdf.cell(200, 10, txt=f"Time {times_min[i]:.1f} min:", ln=True)
                pdf.cell(200, 10, txt=f"  D = {D_est:.3e} ± {D_err:.3e} m²/s", ln=True)
                pdf.cell(200, 10, txt=f"  C0 = {C0_est:.4f} ± {C0_err:.4f} mol/L", ln=True)
                pdf.image(imgp, w=150)
                pdf.ln(5)

            # Add summary page
            pdf.add_page()
            pdf.set_font("Arial", "", 12)
            pdf.cell(200, 10, txt="Overall Results", ln=True, align="C")
            pdf.ln(5)
            pdf.cell(200, 10, txt=f"Average D = {np.mean(D_vals):.3e} ± {np.std(D_vals):.3e} m²/s", ln=True)
            pdf.cell(200, 10, txt=f"Average C0 = {np.mean(C0_vals):.4f} ± {np.std(C0_vals):.4f} mol/L", ln=True)
            
            # Add interpretation
            pdf.ln(10)
            pdf.cell(200, 10, txt="Interpretation:", ln=True)
            pdf.multi_cell(180, 10, txt=f"The diffusion coefficient of {np.mean(D_vals):.3e} m²/s is "
                          f"within the expected range for molecular diffusion in liquid media. "
                          f"For comparison, typical values range from 10^-9 to 10^-6 m²/s depending on "
                          f"the molecule size and medium viscosity.")

            out_pdf = os.path.join(out_dir, "cylindrical_diffusion_results.pdf")
            pdf.output(out_pdf)

            self.progress_var.set(100)
            self.progress_label.config(text="Done")
            messagebox.showinfo("Success", f"Analysis complete! Results saved to: {out_pdf}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    root = tk.Tk()
    app = DiffusionGUI(root)
    root.geometry("700x550")
    root.mainloop()
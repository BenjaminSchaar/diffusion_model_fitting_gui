import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure Tk backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.special import j0, j1
from scipy.integrate import quad_vec
from scipy.interpolate import interp2d
# Remove dependency on scikit-learn
import os
import threading
import time

# Manual implementation of R-squared calculation
def calculate_r2(y_true, y_pred):
    """Calculate the coefficient of determination (R²)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred)**2)
    
    # R-squared
    if ss_tot == 0:  # Handle the case where all y_true values are identical
        return 1.0
    else:
        return 1 - (ss_res / ss_tot)

# Integrand for the diffusion solution
def integrand(u, a, r, D, t):
    return j1(a * u) * j0(r * u) * np.exp(-D * t * u ** 2)

def f_diffusion(r, t, D, C0, a):
    def integrand_with_params(u):
        return a * C0 * integrand(u, a, r, D, t)
    C, err = quad_vec(integrand_with_params, 0, np.inf)
    return C

def interpolate_array(array, num_pts_time, num_pts_distance):
    num_time_original = array.shape[0]
    num_distance_original = array.shape[1]
    time_indices_original = np.arange(num_time_original)
    distance_indices_original = np.arange(num_distance_original)
    new_time_indices = np.linspace(0, num_time_original - 1, num_pts_time)
    new_distance_indices = np.linspace(0, num_distance_original - 1, num_pts_distance)
    interpolator = interp2d(distance_indices_original, time_indices_original, array, kind='linear')
    new_array = interpolator(new_distance_indices, new_time_indices)
    return new_array

class TimePointPlot:
    """Class to manage plots for a single time point"""
    def __init__(self, parent_frame, time_index, time_value):
        self.parent = parent_frame
        self.time_index = time_index
        self.time_min = time_value / 60  # Convert seconds to minutes
        
        # Create a frame for this time point
        self.frame = ttk.LabelFrame(parent_frame, text=f"Time: {self.time_min:.1f} min")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure with two subplots
        self.fig, (self.ax_line, self.ax_img) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Create canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()
        
        # Add fit quality metrics display
        self.metrics_frame = ttk.Frame(self.frame)
        self.metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.r2_var = tk.StringVar(value="R² = N/A")
        self.rmse_var = tk.StringVar(value="RMSE = N/A")
        
        ttk.Label(self.metrics_frame, textvariable=self.r2_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.metrics_frame, textvariable=self.rmse_var).pack(side=tk.LEFT, padx=5)
        
        # Initialize plots
        self.ax_line.set_xlabel('Radius (mm)')
        self.ax_line.set_ylabel('Concentration (mol/L)')
        self.ax_line.set_title(f'Concentration vs. Radius at t={self.time_min:.1f} min')
        
        self.ax_img.set_xlabel('X (mm)')
        self.ax_img.set_ylabel('Y (mm)')
        self.ax_img.set_title(f'Concentration Gradient at t={self.time_min:.1f} min')
        
        self.fig.tight_layout()
    
    def update_plot(self, radii, concs, r_line, c_line, C2d, extent, r2=None, rmse=None):
        """Update both plots with new data"""
        self.ax_line.clear()
        self.ax_img.clear()
        
        # Plot experimental points
        if concs is not None:
            self.ax_line.scatter(radii*1000, concs, color='red', label='Experimental Data')
        
        # Plot simulation line
        self.ax_line.plot(r_line*1000, c_line, 'b-', label='Simulation')
        
        # Plot heatmap
        im = self.ax_img.imshow(C2d, extent=extent, origin='lower', cmap='viridis')
        self.fig.colorbar(im, ax=self.ax_img)
        
        # Update labels and title
        self.ax_line.set_xlabel('Radius (mm)')
        self.ax_line.set_ylabel('Concentration (mol/L)')
        self.ax_line.set_title(f'Concentration vs. Radius at t={self.time_min:.1f} min')
        self.ax_line.legend()
        
        self.ax_img.set_xlabel('X (mm)')
        self.ax_img.set_ylabel('Y (mm)')
        self.ax_img.set_title(f'Concentration Gradient at t={self.time_min:.1f} min')
        
        # Update fit metrics if provided
        if r2 is not None:
            self.r2_var.set(f"R² = {r2:.4f}")
        if rmse is not None:
            self.rmse_var.set(f"RMSE = {rmse:.6f}")
        
        self.fig.tight_layout()
        self.canvas.draw()

class MultiPlotFrame(ttk.Frame):
    """Scrollable frame containing multiple time point plots"""
    def __init__(self, parent):
        super().__init__(parent)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # List to keep track of plot frames
        self.plot_frames = []
    
    def clear_plots(self):
        """Remove all plot frames"""
        for plot in self.plot_frames:
            plot.frame.destroy()
        self.plot_frames = []
    
    def add_plot(self, time_index, time_value):
        """Add a new plot for a time point"""
        plot = TimePointPlot(self.scrollable_frame, time_index, time_value)
        self.plot_frames.append(plot)
        return plot

class DiffusionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Enhanced Diffusion Simulator & Data Fit")
        self.geometry("1200x800")  # Larger window for the enhanced GUI
        self._build_ui()
        
    def _build_ui(self):
        # Create main frames
        control_pane = ttk.Frame(self)
        control_pane.pack(side="left", fill="y", padx=10, pady=10)
        
        plot_pane = ttk.Frame(self)
        plot_pane.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Build control panel
        data_frame = ttk.LabelFrame(control_pane, text="Real-World Data")
        data_frame.pack(fill="x", padx=5, pady=5)
        
        sim_frame = ttk.LabelFrame(control_pane, text="Simulation Parameters")
        sim_frame.pack(fill="x", padx=5, pady=5)
        
        control_frame = ttk.LabelFrame(control_pane, text="Controls & Export")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        progress_frame = ttk.LabelFrame(control_pane, text="Progress")
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        # Build data frame components
        ttk.Label(data_frame, text="Times (min):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.times_var = tk.StringVar(value="60,120")
        ttk.Entry(data_frame, textvariable=self.times_var, width=25).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(data_frame, text="Radii (mm):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.radii_var = tk.StringVar(value="3,5,7,9,11,13")
        ttk.Entry(data_frame, textvariable=self.radii_var, width=25).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(data_frame, text="Concs (mol/L):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.conc_frame = ttk.Frame(data_frame)
        self.conc_frame.grid(row=2, column=1, padx=5, pady=2)
        
        self.conc_text = scrolledtext.ScrolledText(self.conc_frame, width=23, height=4)
        self.conc_text.insert('1.0', "[[0.001148,0.000616,0.000418,0.000286,0.000252,0.000162],\n[0.000948,0.000516,0.000318,0.000186,0.000152,0.000062]]")
        self.conc_text.pack(fill="both", expand=True)
        
        # Build simulation frame components
        ttk.Label(sim_frame, text="D (m²/s):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.D_var = tk.DoubleVar(value=5e-7)
        ttk.Entry(sim_frame, textvariable=self.D_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(sim_frame, text="C₀ (mol/L):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.C0_var = tk.DoubleVar(value=0.02)
        ttk.Entry(sim_frame, textvariable=self.C0_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(sim_frame, text="Well Radius (mm):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.well_var = tk.DoubleVar(value=2.0)
        ttk.Entry(sim_frame, textvariable=self.well_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(sim_frame, text="Source X₀,Y₀ (mm):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        coord_frame = ttk.Frame(sim_frame)
        coord_frame.grid(row=3, column=1, padx=5, pady=2)
        
        self.x0_var = tk.DoubleVar(value=20.297)
        self.y0_var = tk.DoubleVar(value=33.841)
        ttk.Entry(coord_frame, textvariable=self.x0_var, width=5).pack(side="left")
        ttk.Entry(coord_frame, textvariable=self.y0_var, width=5).pack(side="left")
        
        ttk.Label(sim_frame, text="Arena Size (mm):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.size_var = tk.DoubleVar(value=40.5)
        ttk.Entry(sim_frame, textvariable=self.size_var, width=10).grid(row=4, column=1, padx=5, pady=2)
        
        # Build control frame components
        ttk.Label(control_frame, text="Grid N:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.N_var = tk.IntVar(value=200)
        ttk.Entry(control_frame, textvariable=self.N_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Conv Factor:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.conv_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.conv_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Export Time (h):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.export_time_h_var = tk.DoubleVar(value=3.0)
        ttk.Entry(control_frame, textvariable=self.export_time_h_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Button(button_frame, text="Simulate & Plot", command=self.run_and_plot).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Export Grids", command=self.start_export).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Plots", command=self.save_all_plots).pack(side="left", padx=5)
        
        # Build progress frame components
        ttk.Label(progress_frame, text="Calculation:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.calc_progress = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")
        self.calc_progress.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
        ttk.Label(progress_frame, text="Interpolation:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.interp_progress = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")
        self.interp_progress.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        ttk.Label(progress_frame, text="Export:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.export_progress = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")
        self.export_progress.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Build plot pane
        self.plot_container = MultiPlotFrame(plot_pane)
        self.plot_container.pack(fill="both", expand=True)
        
        # Initialize progress bars
        self.reset_progress_bars()

    def reset_progress_bars(self):
        """Reset all progress bars to 0"""
        self.calc_progress["value"] = 0
        self.interp_progress["value"] = 0
        self.export_progress["value"] = 0
        self.status_var.set("Ready")
        self.update_idletasks()

    def update_status(self, message):
        """Update status message and refresh GUI"""
        self.status_var.set(message)
        self.update_idletasks()

    def run_and_plot(self):
        """Run simulation and create plots for each time point"""
        try:
            # Clear existing plots
            self.plot_container.clear_plots()
            
            # Reset progress
            self.reset_progress_bars()
            self.update_status("Parsing input data...")
            
            # Parse input data
            times_min = [float(t) for t in self.times_var.get().split(',')]
            times_s = [t * 60 for t in times_min]  # Convert to seconds
            
            radii_mm = [float(r) for r in self.radii_var.get().split(',')]
            radii_m = [r / 1000.0 for r in radii_mm]  # Convert to meters
            
            # Parse concentrations - handle both single and multiple time points
            conc_text = self.conc_text.get('1.0', 'end').strip()
            try:
                concs = eval(conc_text)
                # Convert to list of lists if it's just a single list
                if isinstance(concs[0], (int, float)):
                    concs = [concs]
            except:
                messagebox.showerror("Input Error", "Invalid concentration format. Use format [[val1,val2,...],[val1,val2,...]]")
                return
            
            # Check dimensions
            if len(concs) != len(times_s):
                resp = messagebox.askquestion("Data Mismatch", 
                                             f"Number of concentration sets ({len(concs)}) doesn't match number of time points ({len(times_s)}).\n\nContinue anyway?")
                if resp != 'yes':
                    return
            
            # Get parameters
            D = self.D_var.get()
            C0 = self.C0_var.get()
            a_mm = self.well_var.get()
            a_m = a_mm / 1000.0  # Convert to meters
            
            rx0_mm = self.x0_var.get()
            ry0_mm = self.y0_var.get()
            rx0_m = rx0_mm / 1000.0  # Convert to meters
            ry0_m = ry0_mm / 1000.0  # Convert to meters
            
            size_mm = self.size_var.get()
            size_m = size_mm / 1000.0  # Convert to meters
            
            N = self.N_var.get()
            conv = self.conv_var.get()
            
            # Set up grid
            X = np.linspace(0, size_m, N)
            Y = np.linspace(0, size_m, N)
            Xg, Yg = np.meshgrid(X, Y)
            Rg = np.sqrt((Xg - rx0_m)**2 + (Yg - ry0_m)**2)
            
            # Create plots for each time point
            for i, t in enumerate(times_s):
                self.update_status(f"Calculating for t = {times_min[i]:.1f} min ({i+1}/{len(times_s)})...")
                progress = (i + 1) / len(times_s) * 100
                self.calc_progress["value"] = progress
                self.update_idletasks()
                
                # Calculate 2D concentration
                C2d = f_diffusion(Rg, t, D, C0, a_m)
                
                # Create 1D profile along a line through the center
                center = N // 2
                r_line = np.sqrt((Xg[:, center] - rx0_m)**2 + (Yg[:, center] - ry0_m)**2)
                sort_idx = np.argsort(r_line)
                r_sorted = r_line[sort_idx]
                c_sorted = C2d[:, center][sort_idx]
                
                # Create a new plot for this time point
                plot = self.plot_container.add_plot(i, t)
                
                # Calculate fit metrics if experimental data is available for this time
                r2, rmse = None, None
                if i < len(concs):
                    # Calculate model predictions at experimental points
                    c_pred = f_diffusion(np.array(radii_m), t, D, C0, a_m)
                    
                    # Calculate R² and RMSE
                    r2 = calculate_r2(concs[i], c_pred)
                    rmse = np.sqrt(np.mean((np.array(concs[i]) - c_pred) ** 2))
                
                # Update the plot
                plot.update_plot(
                    np.array(radii_m),
                    concs[i] if i < len(concs) else None,
                    r_sorted,
                    c_sorted,
                    C2d,
                    (0, size_mm, 0, size_mm),
                    r2,
                    rmse
                )
            
            self.calc_progress["value"] = 100
            self.update_status("Plotting complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during simulation: {str(e)}")
            self.update_status(f"Error: {str(e)}")

    def start_export(self):
        """Start the export process in a separate thread to keep UI responsive"""
        # Reset progress bars
        self.reset_progress_bars()
        # Disable the export button during export
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Controls & Export":
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Frame):
                                for button in grandchild.winfo_children():
                                    if isinstance(button, ttk.Button) and button.cget("text") == "Export Grids":
                                        button.config(state="disabled")
        
        # Start export in separate thread
        export_thread = threading.Thread(target=self.export_grids)
        export_thread.daemon = True
        export_thread.start()

    
    def export_grids(self):
        """Export radial slices from 2D diffusion fields over time, with interpolation."""
        try:
            # Prepare metadata to write to file
            metadata = {
                'Diffusion Coefficient (D)': self.D_var.get(),
                'Initial Concentration (C0)': self.C0_var.get(),
                'Conversion Factor': self.conv_var.get(),
                'Well Radius (mm)': self.well_var.get(),
                'Source X0 (mm)': self.x0_var.get(),
                'Source Y0 (mm)': self.y0_var.get(),
                'Arena Size (mm)': self.size_var.get(),
                'Grid Resolution': self.N_var.get(),
                'Time Duration (s)': int(self.export_time_h_var.get() * 3600),
                'Frame Rate (Hz)': 0.1
            }

            script_dir = os.path.dirname(os.path.abspath(__file__))
            outdir = os.path.join(script_dir, 'output')
            os.makedirs(outdir, exist_ok=True)

            self.update_status("Preparing calculations...")
            total_seconds = int(self.export_time_h_var.get() * 3600)
            times = [t for t in range(10, total_seconds + 1, 10)]  # 0.1 fps
            N = self.N_var.get()
            a = self.well_var.get() / 1000.0
            D = self.D_var.get()
            C0 = self.C0_var.get()

            rx0 = self.x0_var.get() / 1000.0
            ry0 = self.y0_var.get() / 1000.0
            size_m = self.size_var.get() / 1000.0
            X = np.linspace(0, size_m, N)
            Y = np.linspace(0, size_m, N)
            Xg, Yg = np.meshgrid(X, Y)
            Rg = np.sqrt((Xg - rx0) ** 2 + (Yg - ry0) ** 2)

            # Find column closest to source x position
            x_index = np.argmin(np.abs(X - rx0))

            conc_gradient = np.zeros((len(times), N))
            distance_array = np.zeros((len(times), N))

            self.update_status("Calculating diffusion and extracting radial slices...")
            for i, t in enumerate(times):
                C2d = f_diffusion(Rg, t, D, C0, a)
                conc_gradient[i, :] = C2d[:, x_index]
                distance_array[i, :] = Rg[:, x_index]

                # Update progress
                progress = (i + 1) / len(times) * 100
                self.calc_progress["value"] = progress
                if i % 10 == 0:
                    self.update_status(f"Exporting: {i+1}/{len(times)} frames ({progress:.1f}%)")
                    self.update_idletasks()

            # === Interpolation step ===
            self.update_status("Interpolating arrays...")

            from scipy.interpolate import interp2d

            def interpolate_array(array, num_pts_time, num_pts_distance):
                num_time_original = array.shape[0]
                num_distance_original = array.shape[1]
                time_indices_original = np.arange(num_time_original)
                distance_indices_original = np.arange(num_distance_original)
                new_time_indices = np.linspace(0, num_time_original - 1, num_pts_time)
                new_distance_indices = np.linspace(0, num_distance_original - 1, num_pts_distance)
                interpolator = interp2d(distance_indices_original, time_indices_original, array, kind='linear')
                new_array = interpolator(new_distance_indices, new_time_indices)
                return new_array

            conc_gradient = interpolate_array(conc_gradient, 10800, 40500)
            distance_array = interpolate_array(distance_array, 10800, 40500)

            distance_array *= 1000  # Convert from meters to millimeters

            self.interp_progress["value"] = 100
            self.update_idletasks()
            # ==========================

            # Save arrays
            self.update_status("Saving arrays...")
            np.save(os.path.join(outdir, 'conc_gradient.npy'), conc_gradient)
            np.save(os.path.join(outdir, 'distance_array.npy'), distance_array)

            with open(os.path.join(outdir, 'export_info.txt'), 'w') as f:
                for key, val in metadata.items():
                    f.write(f"{key}: {val}\n")

            self.update_status("Export complete.")
            self.export_progress["value"] = 100

            # Re-enable export button
            def enable_button():
                for widget in self.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Controls & Export":
                                for grandchild in child.winfo_children():
                                    if isinstance(grandchild, ttk.Frame):
                                        for button in grandchild.winfo_children():
                                            if isinstance(button, ttk.Button) and button.cget("text") == "Export Grids":
                                                button.config(state="normal")
                messagebox.showinfo("Export Complete", f"Saved to {outdir}")

            self.after(100, enable_button)

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Export Error", str(e))


    def save_all_plots(self):
        """Save all plots to files"""
        try:
            if not self.plot_container.plot_frames:
                messagebox.showinfo("No Plots", "No plots to save. Run simulation first.")
                return
            
            # Ask for directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            outdir = filedialog.askdirectory(initialdir=script_dir)
            
            if not outdir:  # User cancelled
                return
            
            self.update_status("Saving plots...")
            
            # Create plots directory
            plots_dir = os.path.join(outdir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Save each plot
            for i, plot in enumerate(self.plot_container.plot_frames):
                # Save figure
                filename = os.path.join(plots_dir, f"plot_time_{i+1}.png")
                plot.fig.savefig(filename, dpi=300, bbox_inches="tight")
                
                # Update progress
                progress = (i + 1) / len(self.plot_container.plot_frames) * 100
                self.export_progress["value"] = progress
                self.update_status(f"Saving: {i+1}/{len(self.plot_container.plot_frames)} plots ({progress:.1f}%)")
                self.update_idletasks()
            
            self.export_progress["value"] = 100
            self.update_status(f"Plots saved to {plots_dir}")
            messagebox.showinfo("Export Complete", f"All plots saved to {plots_dir}")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving plots: {str(e)}")
            self.update_status(f"Error: {str(e)}")

if __name__ == '__main__':
    app = DiffusionGUI()
    app.mainloop()
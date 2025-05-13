**Slab Diffusion Coefficient Estimator**  

A Python-based GUI application for estimating diffusion coefficients in a one-dimensional slab using concentration profiles over time. This tool fits experimental concentration vs. depth data to the analytical solution of Fick's second law for a slab, computing best-fit values and uncertainties for the diffusion coefficient \(D\) and initial surface concentration \(C_0\).

---

## Features
- Interactive Tkinter GUI to input time snapshots, depths, and concentrations.  
- Automated non-linear least squares fitting of concentration vs. depth using SciPy.  
- Analytical solution uses the error function for diffusion in a finite slab.  
- Generates plots for each time point and compiles a PDF report with fitted parameters and images.  
- Configurable slab half-thickness and default example dataset.  

---

## Usage
1. Launch the GUI:  
   ```bash
   python diffusion_gui.py
   ```
2. In the GUI:  
   - Adjust the number of time snapshots and depth measurements.  
   - Enter time values (in minutes), depths (in mm), and concentration data (in mol/L).  
   - Set the slab half-thickness (in mm).  
   - Click **Run Fit** to perform the fitting procedure.  

3. Results:  
   - Progress bar updates during fitting.  
   - A PDF report (`fit_summary.pdf`) and plot images are saved in an `output/` folder.  

---

## Theory and Mathematical Model
Diffusion in a one-dimensional slab of half-thickness \(l\) with a constant surface concentration . The analytical solution for the concentration profile \(C(z,t)\) as a function of depth \(z\) and time \(t\) is given by the error function formulation of Fick's second law:

\[
C(z,t) = \frac{C_0}{2} \left[ \operatorname{erf}\!\biggl(\frac{l - z}{\sqrt{4 D t}}\biggr) + \operatorname{erf}\!\biggl(\frac{l + z}{\sqrt{4 D t}}\biggr) \right]
\]

- \(D\): Diffusion coefficient (m²/s).  
- \(C_0\): Surface concentration at \(t=0\) (mol/L).  
- \(l\): Slab half-thickness (m).  
- \(z\): Depth coordinate from the center (m).  
- \(\operatorname{erf}(x)\): Gauss error function.  

### Non-linear Least Squares Fit
For each time snapshot \(t_i\) with measured concentrations \(C_{ij}\) at depths \(z_j\), we fit the model:

\[
C_{ij} = \frac{C_0}{2} \bigl[ \operatorname{erf}(\tfrac{l - z_j}{\sqrt{4 D t_i}}) + \operatorname{erf}(\tfrac{l + z_j}{\sqrt{4 D t_i}})\bigr] + \varepsilon_{ij}
\]

Minimizing the residuals \(\varepsilon_{ij}\) via `scipy.optimize.curve_fit` yields estimates \(\hat{D}_i\) and \(\hat{C}_{0,i}\) with covariance-derived uncertainties.

---

## Code Structure
```
diffusion_gui.py   # Main application
requirements.txt   # Dependencies
output/            # Generated plots and PDF report
```
- **DiffusionGUI**: Class handling GUI elements, data validation, fitting loops, and report generation.  
- **diffusion_errorfunction**: Implements the analytical model using `scipy.special.erf`.  
- **run_fit**: Orchestrates data collection, fitting per time point, plotting (Matplotlib), and PDF compilation (FPDF).

---

## Examples
Default example uses three time points (30, 60, 120 min) and five depths (11–19 mm) with synthetic concentration data.  

![Example plot](output/time_0.png)  
*Concentration vs. depth fit at 30 min.*

---

## Output Files
- **output/time_*.png**: Individual plots for each time snapshot.  
- **output/fit_summary.pdf**: Comprehensive report with fitted parameters, uncertainties, and summary statistics.

---

## Dependencies
- Python 3.7+  
- numpy  
- scipy  
- matplotlib  
- tk (tkinter)  
- fpdf

---

## License
MIT License. See [LICENSE](LICENSE) for details.

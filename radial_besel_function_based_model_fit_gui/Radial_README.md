# Cylindrical Diffusion Coefficient Estimation

A Python application with GUI for estimating diffusion coefficients in cylindrical geometry by fitting experimental concentration vs. radius data to a modified analytical solution.

## Features

- **User-friendly GUI** for inputting experimental data with distance, time, and concentration fields
- **Fast parameter estimation** of diffusion coefficient (D) and initial concentration (C0)
- **Smooth visualization** of the diffusion profile with special handling of the center region
- **Comprehensive PDF reports** with fitting results, uncertainties, and graphical visualization
- **Multi-timepoint support** for analyzing diffusion at different time snapshots
- **Flexible input options** for configuring the initial cylinder radius and measurement distances

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:
   ```
   pip install numpy scipy matplotlib tkinter fpdf
   ```
3. Run the application:
   ```
   python cylindrical_diffusion.py
   ```

## Usage Guide

### Data Input Format

1. **Radial Distance (r)**: 
   - Measure from the **center** of the initial cylinder (r=0)
   - Input values in millimeters (mm)
   - Example: For a cylinder with 2mm radius, measurements might be at r = 1, 3, 5, 7, 9, 11 mm
   - First point (1mm) is inside the cylinder, 2mm is at the boundary, and the rest are outside

2. **Time Points**:
   - Input times in minutes since the start of diffusion
   - Multiple time points can be added for time-series analysis

3. **Concentration**:
   - Input in mol/L (or other consistent concentration units)
   - Values should decrease with distance from center

4. **Initial Cylinder Radius**:
   - Set the radius of the initial concentration cylinder in mm
   - Default value is 2.0 mm

### Step-by-Step Instructions

1. **Start the application**
   - Run `python cylindrical_diffusion.py`
   
2. **Configure your data points**
   - Set the number of time snapshots using the spinner or +/- buttons
   - Set the number of radial distance points using the spinner or +/- buttons
   - Enter the initial cylinder radius (in mm)
   
3. **Enter your measurements**
   - Fill in all time values (min)
   - Fill in all radial distances (mm) from center outward
   - Fill in all concentration values (mol/L)
   
4. **Run the analysis**
   - Click "Run Fit" to perform the calculation
   - The progress bar will show the fitting progress
   
5. **View and save results**
   - Results are automatically saved to the "output" folder
   - A PDF report contains all graphs and fitting parameters
   - The diffusion coefficient (D) and initial concentration (C0) are reported with uncertainties

## Understanding the Output

1. **Individual Time Plots**:
   - Blue dots: Your experimental data
   - Red line: Fitted model curve
   - Top right legend: Estimated D and C0 values
   
2. **PDF Report**:
   - First page: Individual time point results with plots
   - Summary page: Average values of D and C0 across all time points with statistical uncertainties
   - Interpretation section with context for the results

## Tips for Best Results

- Ensure measurements include points both inside and outside the initial cylinder
- For best accuracy, include at least one measurement point near the center (r < a)
- More radial distance points lead to better fit quality
- Multiple time points help verify the consistency of the diffusion coefficient
- Fill all fields completely before running the fit
- The model works best for systems that follow Fick's laws of diffusion in cylindrical geometry

## Output Files

All results are saved to the `output` directory:
- `time_1.png`, `time_2.png`, etc.: Plot for each time snapshot
- `cylindrical_diffusion_results.pdf`: Comprehensive report with all fits and analysis
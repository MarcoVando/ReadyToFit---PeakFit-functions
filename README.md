# 📊 ReadyToFit — PeakFit Functions

A flexible and modular Python toolkit for multi-peak curve fitting, parameter management, visualization, and area analysis using scipy.optimize.curve_fit.

This package is designed to handle arbitrary numbers of peaks with different models (Gaussian, Voigt, Asymmetric, Skewed), including support for fixed parameters (e.g., fixed peak centers μ).

## 🚀 Features
🔹 Multi-peak fitting with arbitrary peak count  
🔹 Multiple peak models:  
&emsp;- Gaussian  
&emsp;- Voigt  
&emsp;- Asymmetric  
&emsp;- Skewed  
🔹 Support for fixed parameters (e.g., μ fixed per peak)  
🔹 Automatic parameter flattening/unflattening  
🔹 Robust initial guess generation  
🔹 Flexible bounds handling  
🔹 Full peak decomposition after fitting  
🔹 Area integration for peaks and total signal  
🔹 Clean visualization with residuals and RMSE  

## ⚙️ Installation

Clone the repository:
```
git clone https://github.com/your-username/ReadyToFit---PeakFit-functions.git
cd ReadyToFit---PeakFit-functions
```

Install dependencies:
```
pip install numpy scipy matplotlib
```

## 🧠 Core Concept
Each peak is defined as a dictionary:
```python
peaks = [
    {"model": "gauss"},
    {"model": "voigt", "mu": 50},  # fixed center
    {"model": "skew"}
]
```
The system automatically:  
- builds the composite model  
- lattens parameters for optimization  
- handles fixed parameters  
- reconstructs fitted peaks  

## 📈 Usage Example  
**1. Fit a multi-peak signal**

```python
from fit_models import fit_model
import numpy as np

# synthetic data
x = np.linspace(0, 100, 500)
y = np.sin(x/10) + np.random.normal(0, 0.05, len(x))

# define peaks
peaks = [
    {"model": "gauss"},
    {"model": "voigt"},
    {"model": "skew"}
]

result = fit_model(x, y, peaks)
```

**2. Plot results**
```python
from plot_fit import plot_fit_result
plot_fit_result(x, y, result)
```
**3. Compute peak areas**
```python
from area_integration import evaluate_peak_areas

areas = evaluate_peak_areas(x, result)

print("Total area:", areas["total"])
print("Peak areas:", areas["peaks"])
```

**4. Access fitted parameters**
```python
from parameters import unflatten_params

params = unflatten_params(peaks, result["popt"])

for i, p in enumerate(params):
    print(f"Peak {i}:", p)
```
## 📊 Output Example

After fitting, result contains:
```python
{
    "popt": [...],              # optimized parameters
    "total_fit": [...],        # full reconstructed signal
    "peak_fits": [...],        # individual peaks
    "residual": [...],         # y - fit
    "param_names": [...],      # parameter labels
    "param_slices": [...],     # index mapping
}
```

## 🔬 Supported Peak Models
|Model	|Parameters  |
|-------|------------|
|gauss	|A, μ, σ|
|voigt	|A, μ, σ, γ|
|asym	|A, μ, σL, σR, γ|
|skew	|A, μ, σ, γ, α|

If mu is fixed:
```python
{"model": "gauss", "mu": 50}
```
→ μ is removed from optimization.

## 📐 Key Features in Detail
🔹 Automatic parameter handling  
No manual indexing required — parameters are flattened internally.

🔹 Robust fitting pipeline
Handles:
- missing p0
- partial bounds
i- nvalid input gracefully

🔹 Full decomposition
You can inspect:
- total fit
- individual peaks
- residuals
- peak areas

## 📉 Visualization

The plotting utility includes:

- raw data
- total fit
- individual peaks (dashed)
- filled peak areas
- residual curve
- RMSE annotation


## 📦 Dependencies
numpy
scipy
matplotlib


## ⭐ Author Notes

This project is designed as a modular research-grade fitting system, not a black-box tool.
Each component can be reused independently in scientific workflows.

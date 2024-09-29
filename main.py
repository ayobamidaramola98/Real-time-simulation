# ===================================================================
# Code written by Ayobami Daniel Daramola from 7-Sept to 15-Sept 2024
# Replica of Digital twin for Monoblock Divertor 
# ==================================================================

# Line 7-24 helps to install the necessary modules if absents in your computer#
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "numpy", "matplotlib", "scikit-learn", "scipy", "torch", "joblib"
]

# Try to import and install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        install(package)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import GradientBoostingRegressor
from scipy.interpolate import interp1d
import torch
from torch.distributions import Normal
from joblib import Parallel, delayed

# =================================================================================================================
# Step 1: Material properties (For divertor monoblock)
# Data obtained from L. Humphrey, A.J Dubas, L.C Fletcher and A. Davis 2024, Plasma Phys. Control. Fusion 66 025002
# =================================================================================================================

def get_material_properties(temperature, material="tungsten"):
    if material == "tungsten":
        temp_points = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        conductivity_values = [173, 170, 165, 160, 156, 151, 147, 143, 140, 136, 133, 130, 127, 125, 122]
        expansion_values = [4.50, 4.50, 4.50, 4.53, 4.53, 4.58, 4.72, 4.76, 4.63, 4.68, 4.68, 4.72, 4.54, 4.40, 4.20]
        modulus_values = [398, 398, 397, 397, 396, 396, 395, 394, 393, 391, 390, 388, 387, 385, 383]

    elif material == "copper":
        temp_points = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
        conductivity_values = [401, 398, 395, 391, 388, 384, 381, 378, 374, 371, 367, 364, 360, 357, 354, 350, 347, 344, 340, 337, 334]
        expansion_values = [16.7, 17.0, 17.2, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4, 18.5, 18.7, 18.8, 19.0, 19.1, 19.3, 19.6, 19.8, 20.1, 20.3, 20.5]
        modulus_values = [117, 116, 114, 112, 110, 108, 105, 102, 98, 95, 92, 90, 87, 84, 82, 80, 78, 76, 74, 72, 70]

    elif material == "CuCrZr":
        temp_points = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        conductivity_values = [318, 324, 333, 339, 343, 345, 346, 347, 347, 346, 346, 345, 343, 339, 336]
        expansion_values = [16.7, 17.0, 17.3, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4, 18.5, 18.7, 18.8, 19.0, 19.2]
        modulus_values = [128, 127, 127, 125, 123, 121, 118, 116, 113, 110, 106, 102, 98, 95, 90]

    else:
        raise ValueError("Unknown material specified!")

    conductivity = interp1d(temp_points, conductivity_values, kind='linear', fill_value="extrapolate")(temperature)
    expansion = interp1d(temp_points, expansion_values, kind='linear', fill_value="extrapolate")(temperature)
    modulus = interp1d(temp_points, modulus_values, kind='linear', fill_value="extrapolate")(temperature)

    return conductivity, expansion, modulus

# ==============================================================================
# Step 2: in the absent of FEM (OPENFOAM and MOOSE) divertor monoblock design
# =============================================================================

def design_monoblock_divertor(nx=100, ny=100):
    # Generate a 2D grid for the divertor monoblock
    X, Y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    X, Y = np.meshgrid(X, Y)

    # Monoblock Divertor with cooling pipe (simulating the divertor structure)
    cooling_pipe_radius = 0.1
    cooling_pipe_center = (0.5, 0.5)

    pipe_mask = (X - cooling_pipe_center[0])**2 + (Y - cooling_pipe_center[1])**2 < cooling_pipe_radius**2

    # Simulated data for temperature, stress, and heat flux
    temperature = np.random.uniform(300, 1200, (ny, nx))  # K
    stress = np.random.uniform(0, 1000, (ny, nx))  # MPa
    heat_flux = np.random.uniform(10, 20, (ny, nx))  # MW/m²

    # Apply lower temperature near cooling pipe
    temperature[pipe_mask] = np.random.uniform(300, 400, size=temperature[pipe_mask].shape)

    # Material properties for tungsten armor, copper interlayer, and CuCrZr pipe
    material_props = {}
    material_props['tungsten'] = get_material_properties(temperature, material="tungsten")
    material_props['copper'] = get_material_properties(temperature, material="copper")
    material_props['CuCrZr'] = get_material_properties(temperature, material="CuCrZr")

    return X, Y, temperature, stress, heat_flux, material_props, pipe_mask


# ===============================================================
# Step 2b: Elasto-plasticity identification: plastic deformation
# ================================================================

def compute_plastic_strain(stress, elastic_modulus, yield_stress):
    plastic_strain = np.zeros_like(stress)
    strain = stress / elastic_modulus
    plastic_region = stress > yield_stress
    plastic_strain[plastic_region] = strain[plastic_region] - yield_stress / elastic_modulus
    return plastic_strain


# =================================
# Step 3: Real-time data simulation
# =================================

def get_real_time_data(temperature, stress, heat_flux):
    # Simulating real-time fluctuations in temperature, stress, and heat flux
    temperature += np.random.uniform(-10, 10, temperature.shape)  # K fluctuations
    stress += np.random.uniform(-50, 50, stress.shape)  # MPa fluctuations
    heat_flux += np.random.uniform(-0.5, 0.5, heat_flux.shape)  # MW/m² fluctuations

    return temperature, stress, heat_flux

# =================================================
# Step 4: Heat conduction simulation (Heat equation)
# =================================================

def simulate_heat_conduction(temperature, time_step=0.1, alpha=1e-5):
    """
    Solves the 2D heat equation using an explicit finite-difference method.
    """
    ny, nx = temperature.shape
    laplacian_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian = np.zeros_like(temperature)
    
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            laplacian[i, j] = np.sum(temperature[i-1:i+2, j-1:j+2] * laplacian_matrix)
    
    temperature_new = temperature + alpha * time_step * laplacian
    return temperature_new

# =================================================================================
# Step 5: Predictive maintenance with time-series data: very short machine learning
# =================================================================================

def predictive_maintenance(time_series_stress, time_series_temperature):
    """
    Uses time-series data to predict failure points over time.
    """
    X = np.column_stack((time_series_stress.flatten(), time_series_temperature.flatten()))
    y = (time_series_stress + time_series_temperature).flatten()

    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(X, y)
    
    future_predictions = model.predict(X).reshape(time_series_stress.shape)
    
    prior_mean = torch.tensor(future_predictions)
    prior_std = torch.tensor(0.1)
    
    prior_mean_np = prior_mean.numpy()
    threshold = np.percentile(prior_mean_np, 95)
    failure_points = prior_mean_np > threshold
    
    return failure_points, future_predictions

# ==========================================================
# Step 6: Real-time visualization with animation and analysis
# ============================================================

def analyze_results(temperature, stress, heat_flux, failure_points):
    """
    Analyzes the results from the simulation and provides statistical summaries.
    """
    avg_temp = np.mean(temperature)
    max_stress = np.max(stress)
    avg_heat_flux = np.mean(heat_flux)
    num_failures = np.sum(failure_points)

    print(f"Average Temperature: {avg_temp:.2f} K")
    print(f"Maximum Stress: {max_stress:.2f} MPa")
    print(f"Average Heat Flux: {avg_heat_flux:.2f} MW/m²")
    print(f"Number of Predicted Failures: {num_failures}")

yield_stresses = {
    "tungsten": 500,
    "copper": 210,
    "CuCrZr": 300
}

elastic_moduli = {
    "tungsten": 400e3,
    "copper": 117e3,
    "CuCrZr": 128e3
}

plastic_strain_over_time = []
max_stress_over_time = []

def animate_real_time(i, X, Y, temperature, stress, heat_flux, pipe_mask, ims, ax_stress, ax_plastic_strain, material_props, colorbars):
    temperature, stress, heat_flux = get_real_time_data(temperature, stress, heat_flux)
    temperature = simulate_heat_conduction(temperature)
    failure_points, future_predictions = predictive_maintenance(stress, temperature)

    plastic_strain = np.zeros_like(stress)
    tungsten_mask = ~pipe_mask
    plastic_strain[tungsten_mask] = compute_plastic_strain(stress[tungsten_mask], elastic_moduli["tungsten"], yield_stresses["tungsten"])

    copper_mask = (X > 0.4) & (X < 0.6) & (Y > 0.4) & (Y < 0.6) & ~pipe_mask
    plastic_strain[copper_mask] = compute_plastic_strain(stress[copper_mask], elastic_moduli["copper"], yield_stresses["copper"])

    plastic_strain[pipe_mask] = compute_plastic_strain(stress[pipe_mask], elastic_moduli["CuCrZr"], yield_stresses["CuCrZr"])

    analyze_results(temperature, stress, heat_flux, failure_points)

    # Update the data in the existing plots
    for ax in ims:
        ax.clear()

    c1 = ims[0].contourf(X, Y, stress, cmap='coolwarm')
    ims[0].set_title('Real-time Stress Distribution')
    colorbars[0].update_normal(c1)

    c2 = ims[1].contourf(X, Y, temperature, cmap='hot')
    ims[1].set_title('Real-time Temperature Distribution')
    colorbars[1].update_normal(c2)

    c3 = ims[2].contourf(X, Y, heat_flux, cmap='inferno')
    ims[2].set_title('Real-time Heat Flux Distribution')
    colorbars[2].update_normal(c3)

    # Adding the pipe overlay
    failure_overlay = np.where(failure_points, np.nan, stress)
    c4 = ims[3].contourf(X, Y, failure_overlay, cmap='coolwarm')
    ims[3].set_title('Predicted Failure Points Overlay')
    colorbars[3].update_normal(c4)

    # Adding pipe boundary
    pipe_contour = ims[3].contour(X, Y, pipe_mask, colors='black', levels=[0.5], linewidths=2)

    # Redraw contours
    ims[0].contour(X, Y, stress, levels=10, colors='k', linewidths=0.5)
    ims[1].contour(X, Y, temperature, levels=10, colors='k', linewidths=0.5)
    ims[2].contour(X, Y, heat_flux, levels=10, colors='k', linewidths=0.5)
    ims[3].contour(X, Y, failure_overlay, levels=10, colors='k', linewidths=0.5)

    max_stress = np.max(stress)
    max_stress_over_time.append(max_stress)

    max_plastic_strain = np.max(plastic_strain)
    plastic_strain_over_time.append(max_plastic_strain)

    ax_stress.clear()
    ax_stress.plot(max_stress_over_time, color='red')
    ax_stress.set_title('Max Stress Over Time')
    ax_stress.set_xlabel('Time (frames)')
    ax_stress.set_ylabel('Max Stress (MPa)')

    ax_plastic_strain.clear()

    for material, color in zip(['tungsten', 'copper', 'CuCrZr'], ['blue', 'green', 'red']):
        if material == 'tungsten':
            mask = tungsten_mask
        elif material == 'copper':
            mask = copper_mask
        else:
            mask = pipe_mask
        
        current_plastic_strain = compute_plastic_strain(stress[mask], elastic_moduli[material], yield_stresses[material])
        ax_plastic_strain.plot(current_plastic_strain.flatten(), stress[mask].flatten(), marker='o', color=color, linestyle='None', markersize=1, label=material)

    ax_plastic_strain.set_title('Stress vs. Plastic Strain')
    ax_plastic_strain.set_xlabel('Plastic Strain')
    ax_plastic_strain.set_ylabel('Stress (MPa)')
    ax_plastic_strain.legend(loc='best')

    return ims, ax_stress, ax_plastic_strain


def parallel_run_divertor_simulation():
    X, Y, temperature, stress, heat_flux, material_props, pipe_mask = design_monoblock_divertor()

    fig, axs = plt.subplots(3, 2, figsize=(20, 16))

    ims = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    ax_stress = axs[2, 0]
    ax_plastic_strain = axs[2, 1]

    # Create initial contour plots and color bars
    c1 = ims[0].contourf(X, Y, stress, cmap='coolwarm')
    ims[0].set_title('Real-time Stress Distribution')
    colorbars = [fig.colorbar(c1, ax=axs[0, 0], orientation='vertical')]

    c2 = ims[1].contourf(X, Y, temperature, cmap='hot')
    ims[1].set_title('Real-time Temperature Distribution')
    colorbars.append(fig.colorbar(c2, ax=axs[0, 1], orientation='vertical'))

    c3 = ims[2].contourf(X, Y, heat_flux, cmap='inferno')
    ims[2].set_title('Real-time Heat Flux Distribution')
    colorbars.append(fig.colorbar(c3, ax=axs[1, 0], orientation='vertical'))

    failure_overlay = np.where(np.zeros_like(stress), np.nan, stress)
    c4 = ims[3].contourf(X, Y, failure_overlay, cmap='coolwarm')
    ims[3].set_title('Predicted Failure Points Overlay')
    colorbars.append(fig.colorbar(c4, ax=axs[1, 1], orientation='vertical'))

    ani = animation.FuncAnimation(
        fig,
        animate_real_time,
        fargs=(X, Y, temperature, stress, heat_flux, pipe_mask, ims, ax_stress, ax_plastic_strain, material_props, colorbars),
        frames=100,
        interval=1000,
        blit=False
    )

    plt.tight_layout()
    plt.show()

# Run the real-time animation
parallel_run_divertor_simulation()

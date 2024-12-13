import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor
from scipy.interpolate import interp1d

# Ensure that the folders exist for saving frames
os.makedirs("temperature_frames", exist_ok=True)
os.makedirs("heat_flux_frames", exist_ok=True)
os.makedirs("thermal_stress_frames", exist_ok=True)
os.makedirs("predictive_maintenance_frames", exist_ok=True)

# Step 1: Material properties
def get_material_properties(temperature, material="tungsten"):
    if material == "tungsten":
        temp_points = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
        conductivity_values = np.array([173, 170, 165, 160, 156, 151, 147, 143, 140, 136, 133, 130, 127, 125, 122])
        expansion_values = np.array([4.50, 4.50, 4.50, 4.53, 4.53, 4.58, 4.72, 4.76, 4.63, 4.68, 4.68, 4.72, 4.54, 4.40, 4.20])
        modulus_values = np.array([398, 398, 397, 397, 396, 396, 395, 394, 393, 391, 390, 388, 387, 385, 383])
    elif material == "copper":
        temp_points = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
        conductivity_values = np.array([401, 398, 395, 391, 388, 384, 381, 378, 374, 371, 367, 364, 360, 357, 354, 350, 347, 344, 340, 337, 334])
        expansion_values = np.array([16.7, 17.0, 17.2, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4, 18.5, 18.7, 18.8, 19.0, 19.1, 19.3, 19.6, 19.8, 20.1, 20.3, 20.5])
        modulus_values = np.array([117, 116, 114, 112, 110, 108, 105, 102, 98, 95, 92, 90, 87, 84, 82, 80, 78, 76, 74, 72, 70])
    elif material == "CuCrZr":
        temp_points = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
        conductivity_values = np.array([318, 324, 333, 339, 343, 345, 346, 347, 347, 346, 346, 345, 343, 339, 336])
        expansion_values = np.array([16.7, 17.0, 17.3, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4, 18.5, 18.7, 18.8, 19.0, 19.2])
        modulus_values = np.array([128, 127, 127, 125, 123, 121, 118, 116, 113, 110, 106, 102, 98, 95, 90])
    else:
        raise ValueError("Unknown material specified!")

    conductivity = interp1d(temp_points, conductivity_values, kind='linear', fill_value="extrapolate")(temperature)
    expansion = interp1d(temp_points, expansion_values, kind='linear', fill_value="extrapolate")(temperature)
    modulus = interp1d(temp_points, modulus_values, kind='linear', fill_value="extrapolate")(temperature)

    return conductivity.astype(float), expansion.astype(float), modulus.astype(float)

# Step 2: Divertor monoblock design
def design_monoblock_divertor(nx=100, ny=100):
    X, Y = np.linspace(-20, 20, nx), np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(X, Y)

    cooling_pipe_radius = 6
    interlayer_radius = 7.5
    cooling_pipe_center = (0.5, 0.5)

    pipe_mask = (X - cooling_pipe_center[0])**2 + (Y - cooling_pipe_center[1])**2 < cooling_pipe_radius**2
    interlayer_mask = (X - cooling_pipe_center[0])**2 + (Y - cooling_pipe_center[1])**2 < interlayer_radius**2
    armor_mask = ~(pipe_mask | interlayer_mask)

    temperature = np.random.uniform(300, 1200, (ny, nx))
    stress = np.random.uniform(0, 1000, (ny, nx))
    heat_flux = np.random.uniform(10, 20, (ny, nx))

    temperature[pipe_mask] = np.random.uniform(300, 400, size=temperature[pipe_mask].shape)
    interlayer_size = np.sum(interlayer_mask) - np.sum(pipe_mask)
    temperature[interlayer_mask & ~pipe_mask] = np.random.uniform(400, 600, size=interlayer_size)
    temperature[armor_mask] = np.random.uniform(600, 1200, size=temperature[armor_mask].shape)

    material_props = {
        'tungsten': get_material_properties(temperature, material="tungsten"),
        'copper': get_material_properties(temperature, material="copper"),
        'CuCrZr': get_material_properties(temperature, material="CuCrZr")
    }

    return X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask, armor_mask

# Real-time data simulation
def get_real_time_data(temperature, stress, heat_flux):
    temperature += np.random.uniform(-10, 10, temperature.shape)
    stress += np.random.uniform(-50, 50, stress.shape)
    heat_flux += np.random.uniform(-0.5, 0.5, heat_flux.shape)
    return temperature, stress, heat_flux

# Heat conduction simulation---- FDM analysis------
def simulate_heat_conduction(temperature, time_step=0.1, alpha=1e-5):
    ny, nx = temperature.shape
    laplacian_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian = np.zeros_like(temperature)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            laplacian[i, j] = np.sum(temperature[i-1:i+2, j-1:j+2] * laplacian_matrix)
    temperature_new = temperature + alpha * laplacian
    return temperature_new

# Predictive maintenance using regression
def calculate_thermal_stress(temperature, material_props, reference_temp):
    conductivity, expansion, modulus = material_props
    modulus_mpa = modulus * 1000
    thermal_stress = modulus_mpa * expansion * (temperature - reference_temp) * 1e-6
    return thermal_stress

def update_reference_temperature(frame):
    return 20 + (frame % 100) * (500 - 20) / 100

def predictive_maintenance(thermal_stress):
    model = GradientBoostingRegressor()
    X = np.array([thermal_stress.flatten()]).T
    y = np.random.uniform(0, 1, X.shape[0])
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

# Visualization function for saving frames with updated font sizes and temperature gradient
def update_visualization(frame, X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask):
    plt.rcParams.update({'font.size': 26, 'font.family': 'Times New Roman'})  # Set default font to 26pt and Times New Roman
    plt.clf()
    
    temperature, stress, heat_flux = get_real_time_data(temperature, stress, heat_flux)
    temperature = simulate_heat_conduction(temperature)
    reference_temp = update_reference_temperature(frame)

    ### Temperature Visualization ###
    plt.figure(figsize=(12, 8))
    plt.title("Temperature Distribution", fontsize=26)
    contour_temp = plt.contourf(X, Y, temperature, cmap='hot', levels=100)
    cbar = plt.colorbar(contour_temp)
    cbar.set_label('Temperature (K)', fontsize=26)
    cbar.ax.tick_params(labelsize=26)
    plt.xlabel("X-axis (mm)", fontsize=26)
    plt.ylabel("Y-axis (mm)", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.savefig(f"temperature_frames/temperature_frame_{frame:03d}.png", dpi=100)
    plt.close()

    ### Temperature Gradient Plot (1D Cross-Section) ###
    mid_row = temperature.shape[0] // 2  # Select the middle row for cross-section
    temperature_gradient = temperature[mid_row, :]  # Extract temperature along this row

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(-20, 20, temperature_gradient.size), temperature_gradient, color='red', linewidth=2)
    plt.title("Temperature Gradient along Middle Row", fontsize=26)
    plt.xlabel("Position along X-axis (mm)", fontsize=26)
    plt.ylabel("Temperature (K)", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    
    # Add vertical lines to mark material boundaries
    cooling_pipe_radius = 6
    interlayer_radius = 7.5
    plt.axvline(x=-cooling_pipe_radius, color='blue', linestyle='--', label='Cooling Pipe Boundary')
    plt.axvline(x=cooling_pipe_radius, color='blue', linestyle='--')
    plt.axvline(x=-interlayer_radius, color='green', linestyle='--', label='Interlayer Boundary')
    plt.axvline(x=interlayer_radius, color='green', linestyle='--')
    plt.legend(fontsize=20)
    plt.savefig(f"temperature_frames/temperature_gradient_{frame:03d}.png", dpi=100)
    plt.close()

# Example usage
X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask, armor_mask = design_monoblock_divertor()
for frame in range(100):  # Generate frames
    update_visualization(frame, X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask)


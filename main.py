import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import GradientBoostingRegressor
from scipy.interpolate import interp1d
import torch
from joblib import Parallel, delayed


# Step 1: Material properties
def get_material_properties(temperature, material="tungsten"):
    if material == "tungsten":
        temp_points = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
        conductivity_values = np.array([173, 170, 165, 160, 156, 151, 147, 143, 140, 136, 133, 130, 127, 125, 122])
        expansion_values = np.array([4.50, 4.50, 4.50, 4.53, 4.53, 4.58, 4.72, 4.76, 4.63, 4.68, 4.68, 4.72, 4.54, 4.40, 4.20])  # µm/(m·°C)
        modulus_values = np.array([398, 398, 397, 397, 396, 396, 395, 394, 393, 391, 390, 388, 387, 385, 383])  # GPa
    elif material == "copper":
        temp_points = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
        conductivity_values = np.array([401, 398, 395, 391, 388, 384, 381, 378, 374, 371, 367, 364, 360, 357, 354, 350, 347, 344, 340, 337, 334])
        expansion_values = np.array([16.7, 17.0, 17.2, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4, 18.5, 18.7, 18.8, 19.0, 19.1, 19.3, 19.6, 19.8, 20.1, 20.3, 20.5])  # µm/(m·°C)
        modulus_values = np.array([117, 116, 114, 112, 110, 108, 105, 102, 98, 95, 92, 90, 87, 84, 82, 80, 78, 76, 74, 72, 70])  # GPa
    elif material == "CuCrZr":
        temp_points = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
        conductivity_values = np.array([318, 324, 333, 339, 343, 345, 346, 347, 347, 346, 346, 345, 343, 339, 336])
        expansion_values = np.array([16.7, 17.0, 17.3, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4, 18.5, 18.7, 18.8, 19.0, 19.2])  # µm/(m·°C)
        modulus_values = np.array([128, 127, 127, 125, 123, 121, 118, 116, 113, 110, 106, 102, 98, 95, 90])  # GPa
    else:
        raise ValueError("Unknown material specified!")

    conductivity = interp1d(temp_points, conductivity_values, kind='linear', fill_value="extrapolate")(temperature)
    expansion = interp1d(temp_points, expansion_values, kind='linear', fill_value="extrapolate")(temperature)
    modulus = interp1d(temp_points, modulus_values, kind='linear', fill_value="extrapolate")(temperature)

    return conductivity.astype(float), expansion.astype(float), modulus.astype(float)

# Step 2: Divertor monoblock design
def design_monoblock_divertor(nx=100, ny=100):
    # Generate a 2D grid for the divertor monoblock
    X, Y = np.linspace(-20, 20, nx), np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(X, Y)

    # Monoblock Divertor with cooling pipe and interlayer
    cooling_pipe_radius = 6
    interlayer_radius = 7.5  # Adding thickness of interlayer
    cooling_pipe_center = (0.5, 0.5)

    # Masks for cooling pipe and interlayer
    pipe_mask = (X - cooling_pipe_center[0])**2 + (Y - cooling_pipe_center[1])**2 < cooling_pipe_radius**2
    interlayer_mask = (X - cooling_pipe_center[0])**2 + (Y - cooling_pipe_center[1])**2 < interlayer_radius**2
    armor_mask = ~(pipe_mask | interlayer_mask)  # Area outside the pipe and interlayer is the armor

    # Simulated data for temperature, stress, and heat flux
    temperature = np.random.uniform(300, 1200, (ny, nx))  # K
    stress = np.random.uniform(0, 1000, (ny, nx))  # MPa
    heat_flux = np.random.uniform(10, 20, (ny, nx))  # MW/m²

    # Apply lower temperature near cooling pipe
    temperature[pipe_mask] = np.random.uniform(300, 400, size=temperature[pipe_mask].shape)

    # Ensure correct size when assigning temperature for interlayer
    interlayer_size = np.sum(interlayer_mask) - np.sum(pipe_mask)  # Size for interlayer excluding pipe
    temperature[interlayer_mask & ~pipe_mask] = np.random.uniform(400, 600, size=interlayer_size)

    # Update temperature for the armor region
    temperature[armor_mask] = np.random.uniform(600, 1200, size=temperature[armor_mask].shape)

    # Material properties for tungsten armor, copper interlayer, and CuCrZr pipe
    material_props = {}
    material_props['tungsten'] = get_material_properties(temperature, material="tungsten")
    material_props['copper'] = get_material_properties(temperature, material="copper")
    material_props['CuCrZr'] = get_material_properties(temperature, material="CuCrZr")

    return X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask, armor_mask

# Real-time data simulation
def get_real_time_data(temperature, stress, heat_flux):
    temperature += np.random.uniform(-10, 10, temperature.shape)  # K fluctuations
    stress += np.random.uniform(-50, 50, stress.shape)  # MPa fluctuations
    heat_flux += np.random.uniform(-0.5, 0.5, heat_flux.shape)  # MW/m² fluctuations

    return temperature, stress, heat_flux

# Heat conduction simulation
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
    # Convert modulus from GPa to MPa
    modulus_mpa = modulus * 1000  # Convert GPa to MPa
    # Calculate thermal stress (units: MPa)
    thermal_stress = modulus_mpa * expansion * (temperature - reference_temp) * 1e-6  # Convert µm/(m·°C) to (m·°C) for calculation
    return thermal_stress

def update_reference_temperature(frame):
    # Varying reference temperature between 20°C and 500°C over the frames
    return 20 + (frame % 100) * (500 - 20) / 100

def predictive_maintenance(thermal_stress):
    model = GradientBoostingRegressor()
    X = np.array([thermal_stress.flatten()]).T
    y = np.random.uniform(0, 1, X.shape[0])  # Randomly generated labels for illustration

    model.fit(X, y)
    
    predictions = model.predict(X)
    return predictions

# Real-time visualization using animation
# Real-time visualization using animation
def update_visualization(frame, X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask):
    # Clear previous plots
    plt.clf()
    
    # Update temperature, stress, and heat flux
    temperature, stress, heat_flux = get_real_time_data(temperature, stress, heat_flux)
    temperature = simulate_heat_conduction(temperature)

    # Update reference temperature for this frame
    reference_temp = update_reference_temperature(frame)

    # Visualize temperature
    plt.subplot(2, 2, 1)
    plt.title("Temperature Distribution", fontsize=5)  # Reduce font size
    contour_temp = plt.contourf(X, Y, temperature, cmap='hot', levels=100)
    cbar = plt.colorbar(contour_temp)
    cbar.ax.tick_params(labelsize=4)  # Reduce colorbar font size
    cbar.set_label('Temperature (K)', fontsize=4)

    # Calculate and visualize thermal stress for each material
    thermal_stresses = {}
    for material, props in material_props.items():
        thermal_stresses[material] = calculate_thermal_stress(temperature, props, reference_temp)

    # Visualize thermal stress for each material
    plt.subplot(2, 2, 2)
    plt.title(f"Thermal Stress @ Ref Temp: {reference_temp:.2f} °C", fontsize=10)
    
    # Loop through materials and plot
    for material, thermal_stress in thermal_stresses.items():
        contour = plt.contourf(X, Y, thermal_stress, levels=100, alpha=0.5, label=material)
        cbar = plt.colorbar(contour)
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label(f'Thermal Stress ({material}) (MPa)', fontsize=8)

    # Define failure thresholds for each material
    failure_thresholds = {'tungsten': 250, 'copper': 150, 'CuCrZr': 120}  # MPa
    for material, thermal_stress in thermal_stresses.items():
        failure_mask = thermal_stress > failure_thresholds[material]
        plt.contour(X, Y, failure_mask, colors='red', linewidths=1, levels=1, linestyles='solid')

    # Visualize heat flux
    plt.subplot(2, 2, 3)
    plt.title("Heat Flux Distribution", fontsize=5)
    contour_flux = plt.contourf(X, Y, heat_flux, cmap='plasma', levels=100)
    cbar = plt.colorbar(contour_flux)
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('Heat Flux (MW/m²)', fontsize=8)

    # Indicate Cu and CuCrZr regions on heat flux distribution
    plt.contour(X, Y, pipe_mask, colors='blue', linewidths=1, levels=1, linestyles='solid')  # CuCrZr Pipe
    plt.contour(X, Y, interlayer_mask, colors='green', linewidths=1, levels=1, linestyles='solid')  # Cu Interlayer

    # Add a legend for material indications
    plt.scatter([], [], color='blue', label='CuCrZr Pipe', marker='o')
    plt.scatter([], [], color='green', label='Cu Interlayer', marker='o')
    plt.legend(loc='upper right', fontsize=8)  # Reduce font size for legend

    # Predictive maintenance
    predictions = {}
    for material, thermal_stress in thermal_stresses.items():
        predictions[material] = predictive_maintenance(thermal_stress)

    # Visualize predictive maintenance predictions
    plt.subplot(2, 2, 4)
    plt.title("Predictive Maintenance", fontsize=5)
    for material, prediction in predictions.items():
        plt.scatter(range(len(prediction)), prediction, label=f'{material} Predictions', s=5)  # Reduced point size
    plt.xlabel("Sample", fontsize=4)
    plt.ylabel("Prediction Score", fontsize=4)
    plt.legend(loc='upper left', fontsize=4)


def main_simulation():
    X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask, armor_mask = design_monoblock_divertor()

    # Create an animated plot
    fig = plt.figure(figsize=(6, 3))  # Reduced figure size
    ani = animation.FuncAnimation(fig, update_visualization, fargs=(X, Y, temperature, stress, heat_flux, material_props, pipe_mask, interlayer_mask),
                                  frames=100, interval=200)  # Update every 200 ms

    # Save the animation as a GIF
    ani.save('failure.gif', writer='pillow', fps=5)  # Adjust fps as needed
    
    plt.show()


# Start the simulation
if __name__ == "__main__":
    main_simulation()

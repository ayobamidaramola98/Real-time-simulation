Code is written based on python and data used are mainly from literature.
Divertor Monoblock Simulation and Predictive Maintenance
This project simulates the thermal behavior and structural stresses in a divertor monoblock, a key component of nuclear fusion reactors. The simulation covers material properties under different temperatures, real-time heat conduction, and predictive maintenance using machine learning techniques. The project includes real-time visualization with animations to track the temperature distribution, thermal stresses, and heat flux in the system.

Table of Contents:
Installation and Requirements
Material Properties
Monoblock Divertor Design
Heat Conduction Simulation
Predictive Maintenance
Real-time Visualization
Running the Simulation
Installation and Requirements
Prerequisites
This project uses Python and requires the following libraries:

numpy
matplotlib
scikit-learn
scipy
torch
joblib
Make sure all the required packages are installed before running the simulation.

Installation
To install the necessary libraries, you can run:

bash
Copy code
pip install numpy matplotlib scikit-learn scipy torch joblib
Material Properties
The function get_material_properties() retrieves key material properties such as thermal conductivity, thermal expansion, and Young's modulus (stiffness) based on the temperature for materials like:

Tungsten (used as armor)
Copper (used in interlayers)
CuCrZr (used in cooling pipes)
Interpolation (scipy.interpolate) is used to estimate these properties between given temperature data points, ensuring that materials are represented accurately at different temperatures.

Monoblock Divertor Design
The function design_monoblock_divertor() generates a simulated 2D cross-section of a divertor monoblock, which includes:

Cooling Pipe Region (CuCrZr)
Interlayer (Copper)
Armor Region (Tungsten)
Random temperature, stress, and heat flux values are generated for different regions:

Temperature: Higher in the armor, lower in the cooling pipe region.
Stress: A uniform range of stress values across the grid.
Heat Flux: Simulated between 10–20 MW/m².
This function returns the geometry (2D grid), material masks, and initial conditions for temperature, stress, and heat flux.

Heat Conduction Simulation
simulate_heat_conduction() models how heat diffuses through the divertor. The equation is a discrete 2D approximation of the heat conduction equation using a Laplacian operator:

alpha: A thermal diffusivity coefficient.
The temperature at each point in the grid is updated over time to simulate real-time heat transfer.
Predictive Maintenance
The project includes predictive maintenance using machine learning, where the goal is to monitor thermal stresses and predict failure risks:

Thermal Stress Calculation: calculate_thermal_stress() calculates the thermal stress in each material based on temperature, thermal expansion, and modulus.
Gradient Boosting Regressor: The predictive_maintenance() function uses GradientBoostingRegressor from scikit-learn to predict maintenance needs based on thermal stress data.
This predictive model learns from the stress data and assigns scores (0 to 1) representing failure probabilities.

Real-time Visualization
update_visualization() handles real-time plotting of:

Temperature Distribution: A contour plot showing the temperature across the divertor.
Thermal Stress Distribution: Plots thermal stresses for each material (tungsten, copper, CuCrZr) along with failure regions.
Heat Flux: A contour plot of the heat flux distribution.
Predictive Maintenance: Scatter plots showing predicted maintenance scores based on the stress data.
All these components are updated frame-by-frame to animate the evolution of the system over time. The animation is saved as a GIF (failure.gif).

Running the Simulation
To run the simulation, simply execute the script:

bash
Copy code
python main.py
This will generate a 2D animated visualization of the divertor's temperature, stress, heat flux, and predictive maintenance predictions over time. The animation will be displayed on-screen and saved as failure.gif in the working directory.

File Descriptions
get_material_properties(): Retrieves thermal conductivity, expansion, and Young's modulus for tungsten, copper, and CuCrZr at a given temperature.
design_monoblock_divertor(): Creates the geometry and initial conditions (temperature, stress, heat flux) for the monoblock divertor simulation.
simulate_heat_conduction(): Simulates heat diffusion through the divertor over time.
calculate_thermal_stress(): Computes thermal stress for different materials.
predictive_maintenance(): Machine learning model for predictive maintenance based on stress data.
update_visualization(): Real-time plotting and animation function.
main_simulation(): Main simulation loop that initializes and runs the animation.

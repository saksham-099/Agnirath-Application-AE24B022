import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax


time_steps = 144        #5 min intervals in 12 hours

#Importing synthetic solar irradiance values
solar_irradiance = jnp.array(np.loadtxt("SolarI.csv", delimiter=","))


#Creating a dictionary so that the constants are easier to pass to the functions
constants = {
    'Mass': 290.0,  # kg
    'Density_of_air': 1.2,  # kg/m^3
    'Projected_area': 1.0,  # m^2
    'Coeff_of_rolling_resistance': 0.008, 
    'Coeff_of_drag': 0.15, 
    'eff_of_solar_panel': 0.22,
    'eff_of_motor': 0.95,
    'interval_hr': (5 / 60.0),  # 5 minutes in hours
    'g': 9.81,
    'max_battery_wh': 5000.0, # Watt-hours
    'panel_area': 6.0, # m^2
    'Wind_speed' : 3.5, # m/s
    'Max_velocity' : 90*5/18#m/s
}

#Creating simulation function which performs all the physical calculations
@jax.jit
def simulation(velocities, solar_irradiance, constants):
        
    #Calculating acceleration
    acceleration = jnp.diff(velocities)/(constants["interval_hr"]*3600)     #To convert to seconds
    acceleration = jnp.insert(acceleration,0,0)                             #To make the size the same as velocities
    acceleration = acceleration.clip(0)                                     #To remove all negative accelerations as that does not require force from motor
    
    #Calculating forces
    Drag_force = 0.5 * constants["Coeff_of_drag"] *constants["Density_of_air"] * (velocities + constants["Wind_speed"])**2 * constants["Projected_area"]
    Rolling_resistance_force = constants["Coeff_of_rolling_resistance"] * constants["Mass"] * constants["g"]
    Accln_force = constants["Mass"] * acceleration
    
    #calculating net force
    Total_force = Drag_force + Rolling_resistance_force + Accln_force
    
    #calculating power required
    Power_required = Total_force * velocities
    
    #Due to inefficiency of engine
    Total_power = Power_required/constants["eff_of_motor"]
    
    #Energy required
    Energy_consumed = Total_power * constants["interval_hr"]       
    
    #Solar energy gained
    Energy_gained = solar_irradiance * constants["eff_of_solar_panel"] * constants["panel_area"] * constants["interval_hr"]
    
    #Creating a function to find energy profile
    def battery_step(current_battery, step_io):
        Energy_consumed, Energy_gained = step_io
        new_energy = current_battery + Energy_gained - Energy_consumed
        
        #Imposing condition for new_energy to be less than max energy
        new_energy = jax.lax.cond(
        new_energy > constants["max_battery_wh"], 
        lambda ne: constants["max_battery_wh"],   
        lambda ne: ne,                             
        new_energy                                 
        )

        return new_energy,new_energy

    final_energy, energy_profile = jax.lax.scan(battery_step, constants["max_battery_wh"], (Energy_consumed, Energy_gained))
    
    return energy_profile
    
#Creating cost function which the optimizer has to minimize
@jax.jit   
def cost_function(velocities, solar_irradiance, constants):
    
    energy_profile = simulation(velocities, solar_irradiance, constants)
    
    #setting high penalty for violating conditions
    penalty = 1e6
    
    total_distance = jnp.sum(velocities *( 3.6 )* constants['interval_hr'])     #Multiplied 3.6 to convert m/s to kmph
    
    cost = -1*total_distance                                                    # JAX minimizes, so we minimize the negative distance

    #Increasing cost if energy goes below 0
    cost += jnp.sum(jnp.maximum(0, 0 - energy_profile))*penalty
    
    #increasing cost if initial velocity is non zero
    cost += ((velocities[0])**2 ) * penalty

    #Stayinh under the speed limit
    cost += jnp.sum(jnp.maximum(0, velocities - constants["Max_velocity"])) * penalty

    return cost

#Initial guess
velocities = jnp.ones(time_steps) * 15
learning_rate = 0.01

#Defining objective function with only one parameter as required by jax.grad function
obj_function = lambda v: cost_function(v,solar_irradiance, constants)
value_and_grad_fn = jax.jit(jax.value_and_grad(obj_function))

#Using optax because its just better ( jax wasnt finding optimal solution so without chaning code i changed optimizer and it worked )
#Also its insanely faster
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(velocities)

#Running it 5,00,000 times was the best after trying out different values as it stays under 5 mins and still converges
for i in range(500000):
    cost_val, gradient = value_and_grad_fn(velocities)
    updates, opt_state = optimizer.update(gradient, opt_state)
    velocities = optax.apply_updates(velocities, updates)
    
    #Printing progress
    if i % 500 == 0:
        print(f"Step {i}, Cost: {cost_val:.2f}")
    
final_velocities = velocities

#Printing total distance covered
print(jnp.sum(final_velocities*constants["interval_hr"]*3.6))


#Plotting graphs
time_hours = np.arange(time_steps) * constants['interval_hr']
energy_profile = simulation(final_velocities, solar_irradiance, constants)

# Create a figure and a set of subplots (2 rows, 1 column)
# figsize is optional but helps in getting a good layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# --- First Plot (on the top axis, ax1) ---
ax1.plot(jnp.cumsum(final_velocities * constants["interval_hr"] * 3.6), final_velocities * 3.6)
ax1.set_title("Velocity Profile")
ax1.set_xlabel("Distance Travelled (km)")
ax1.set_ylabel("Velocity (km/h)")
ax1.grid(True)

# --- Second Plot (on the bottom axis, ax2) ---
ax2.plot(time_hours, energy_profile)
ax2.set_title("Energy Profile")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Energy Remaining (Wh)")
ax2.grid(True)

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout()

# Show the single figure with both plots
plt.show()

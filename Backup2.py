import os
import sys
import optparse
import numpy as np
import matplotlib.pyplot as plt
from sumolib import checkBinary
import traci
import xml.etree.ElementTree as ET
import random


# PSO parameters
SWARM_SIZE = 20
MAX_ITERATIONS = 50
INERTIA = 0.5
LOCAL_WEIGHT = 2
GLOBAL_WEIGHT = 2
SLOWDOWN_INTERVAL = 200
BURN_IN_PERIOD = 50
SIMULATION_STEPS = 100
PENALTY_PARAMETER = 2


# Function to read route information from the XML file
def read_routes_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    routes = {}
    for route_elem in root.findall('.//route'):
        route_id = route_elem.get('id')
        route_edges = route_elem.get('edges').split()
        routes[route_id] = route_edges

    return routes


# Generate PSO-controlled vehicles in the route file
def generate_pso_vehicles(route_info):
    pso_vehicles = ""
    for i in range(SWARM_SIZE):
        route_id = f"route_{i % 2}"
        valid_edges = route_info.get(route_id, [])
        if valid_edges:
            depart_time = random.uniform(1, 30)
            pso_vehicles += f' <vehicle id="pso_{i}" type="auto" route="{route_id}" depart="{depart_time}" begin="200" end="1000000" from="e1" to="e8" />\n'

    return pso_vehicles



# Initialize simulation
def initialize_simulation():
    options = get_options()
    if options.nogui:
        sumo_binary = checkBinary('sumo')
    else:
        sumo_binary = checkBinary('sumo-gui')

    try:
        traci.start([sumo_binary, "-c", "demo_network.sumocfg","--random-depart-offset", "100", "--tripinfo-output", "tripinfo.xml"])
    except Exception as e:
        print(f"Error starting Sumo: {e}")
        sys.exit()


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options



def set_initial_parameters(route_info):
    traci.simulationStep(BURN_IN_PERIOD)
    traci.simulationStep()  # To continue the simulation

    for i in range(SWARM_SIZE):
        vehicle_id = f"pso_{i}"
        route_id = f"route_{i % 2}"
        route_edges = route_info.get(route_id, [])

        # Check if the route already exists, if not, add it
        if route_id not in traci.route.getIDList():
            try:
                traci.route.add(route_id, route_edges)
            except traci.exceptions.TraCIException as e:
                print(f"Error adding route '{route_id}': {e}")

        try:
            traci.vehicle.add(vehicle_id, route_id, departLane="random", departPos="base")
        except traci.exceptions.TraCIException as e:
            print(f"Error adding vehicle '{vehicle_id}': {e}")

        traci.vehicle.setType(vehicle_id, "car")
        traci.vehicle.setColor(vehicle_id, (0, 0, 255))  # Set color to blue
        traci.vehicle.setMinGap(vehicle_id, np.random.uniform(0.1, 5))
        traci.vehicle.setLaneChangeMode(vehicle_id, 512)  # Disable lane changing for PSO vehicles

        # Set the initial speed separately
        traci.vehicle.setSpeed(vehicle_id, 25)  # Set to the maximum speed

    print("List of vehicles after addition:", traci.vehicle.getIDList())  # Print list of vehicles for debugging
    print("List of routes after addition:", traci.route.getIDList())  # Print list of routes for debugging
    traci.simulationStep(BURN_IN_PERIOD)
    traci.simulationStep()



def evaluate_particle(particle, iteration):
    traci.simulationStep(0)  # Reset simulation steps to 0 for each particle
    for step in range(SIMULATION_STEPS):
        print(f"Iteration: {iteration + 1}/{MAX_ITERATIONS} - Particle: {particle[3:]} - Simulation Step: {step}")
        traci.simulationStep()

        if step % SLOWDOWN_INTERVAL == 0:
            for i in range(SWARM_SIZE):
                vehicle_id = f"pso_{i}"
                if traci.vehicle.getIDList() and vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setSpeedMode(vehicle_id, 0b000001000)
                # else:
                #     print(f"Vehicle {vehicle_id} not found.")
                    # print("List of vehicles:", traci.vehicle.getIDList())

        for i in range(SWARM_SIZE):
            try:
                vehicle_id = f"pso_{i}"
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setSpeed(vehicle_id, particle[0])
                    traci.vehicle.setMinGap(vehicle_id, particle[1])
                    traci.vehicle.setParameter(vehicle_id, "slowDown", str(particle[2]))
                # else:
                #     print(f"Vehicle {vehicle_id} not found.")
                    # print("List of vehicles:", traci.vehicle.getIDList())
            except traci.exceptions.TraCIException as e:
                print(f"Error setting parameters for {vehicle_id}: {e}")

        # Check if the list of vehicles is empty and end the simulation
        if not traci.vehicle.getIDList():
            print("List of vehicles is empty. Ending simulation.")
            break

        # Ensure that the simulation stops when SIMULATION_STEPS is reached
        if step == SIMULATION_STEPS - 1:
            break

    fitness = get_fitness()
    print(f"Final Fitness: {fitness}")
    return fitness



def get_fitness():
    merges = traci.simulation.getCollidingVehiclesNumber()
    collisions = traci.simulation.getArrivedNumber()
    fitness = merges - PENALTY_PARAMETER * collisions
    return fitness



def update_velocity(particle, personal_best, global_best, velocity_memory):
    inertia_term = INERTIA * np.array(particle[3:])
    personal_best_term = LOCAL_WEIGHT * np.random.random() * (np.array(personal_best[3:]) - np.array(particle[3:]))
    global_best_term = GLOBAL_WEIGHT * np.random.random() * (np.array(global_best[3:]) - np.array(particle[3:]))
    memory_term = np.random.random() * (velocity_memory - np.array(particle[3:]))
    particle[3:] = inertia_term + personal_best_term + global_best_term + memory_term


def update_position(particle):
    particle[0] = max(0, min(25, particle[0]))  # Ensure speed is within [0, 25]
    particle[1] = max(0.1, min(5, particle[1]))  # Ensure minGap is within [0.1, 5]
    particle[2] = max(5, min(25, particle[2]))  # Ensure slowDown is within [5, 25]


def run_pso_experiment():
    global_best_particle = np.random.uniform(low=[0, 0.1, 5], high=[25, 5, 25], size=3).tolist()
    global_best_particle.extend(np.random.rand(3).tolist())  # Append velocity terms

    velocity_memory = np.zeros(3)  # Initialize memory for velocity

    # Lists to store data for plotting
    fitness_history = []
    global_best_history = []

    # Initialize an empty NumPy array for particle fitness history
    particle_fitness_history = np.zeros((MAX_ITERATIONS, SWARM_SIZE))

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration: {iteration + 1}/{MAX_ITERATIONS}")
        for i in range(SWARM_SIZE):
            particle = np.random.uniform(low=[0, 0.1, 5], high=[25, 5, 25], size=3).tolist()
            particle.extend(np.random.rand(3).tolist())  # Append velocity terms
            update_velocity(particle, particle, global_best_particle, velocity_memory)
            update_position(particle)

            fitness = evaluate_particle(particle, iteration)

            if global_best_particle is None or fitness > get_fitness():
                global_best_particle = np.copy(particle)

            update_velocity(particle, particle, global_best_particle, velocity_memory)
            update_position(particle)

            # Update memory with current velocity
            velocity_memory = np.array(particle[3:])

            # Save fitness value to particle fitness history
            particle_fitness_history[iteration, i] = fitness

            # Save data for plotting
            fitness_history.append(fitness)
            global_best_history.append(get_fitness())

        traci.simulationStep()  # Run a simulation step

        # Add a condition to break out of the loop when reaching the desired simulation steps
        if traci.simulation.getTime() >= SIMULATION_STEPS:
            break

    # Ensure that the simulation stops at the specified number of steps
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()
    sys.stdout.flush()

    # Print or save the collected results
    print_results(particle_fitness_history, global_best_history)



    # Print or save the collected results
    print_results(fitness_history, global_best_history)



def print_results(particle_fitness_history, global_best_fitness_history):
    num_particles = len(particle_fitness_history[0])

    # Plot individual particle fitness over iterations
    plt.figure(figsize=(12, 8))
    for i in range(num_particles):
        plt.subplot(num_particles + 1, 1, i + 1)
        plt.plot(particle_fitness_history[:, i], label=f'Particle {i + 1} Fitness')
        plt.title(f'Particle {i + 1} Fitness Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.legend()

    # Plot global best fitness over iterations
    plt.subplot(num_particles + 1, 1, num_particles + 1)
    plt.plot(global_best_fitness_history, label='Global Best Fitness', color='black', linestyle='dashed')
    plt.title('Global Best Fitness Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Read route information from the demo_network.rou.xml file
    route_info = read_routes_from_xml("C:/Sumo/Thesis2/Maps/SwarmOptimizationinPSO1/demo_network.rou.xml")
    print(route_info)

    # Generate PSO-controlled vehicles in the route file
    pso_vehicles = generate_pso_vehicles(route_info)
    print(pso_vehicles)

    # Modify the network.rou.xml file with the PSO-controlled vehicle type and route
    modified_routes = """
    <routes>
        <vType id="car" 
                vClass="passenger" length="5" maxSpeed="15" carFollowModel="Krauss" color="255,204,0" accel="2.6" decel="4.5" maxSpeed="70" begin="0" end="4000" safety="none" 
                sigma="1.0"/>

         <route id="route_0" edges="e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1"/>
         <route id="route_1" edges="e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 e6 e7 e8 e1"/>


        %s

    </routes>
    """ % pso_vehicles

    with open("C:/Sumo/Thesis2/Maps/SwarmOptimizationinPSO1/new_network.rou.xml", "w") as file:
        file.write(modified_routes)

    initialize_simulation()
    set_initial_parameters(route_info)
    run_pso_experiment()
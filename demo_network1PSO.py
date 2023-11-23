#!/usr/bin/env python

import os
import sys
import optparse
import random
import csv
import pandas as pd
import xml.etree.ElementTree as ET
from sumolib import checkBinary
import traci

# We need to import some Python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Checks for the binary in environment variables
import traci



def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


def load_network_data():
    # Parse XML file and set route_id
    xml_file = ET.parse("demo_network.rou.xml")
    route_id = "route_0"

    # Count the number of "auto" vehicles in the XML file
    auto_vehicle_elements = xml_file.findall(".//vehicle[@type='auto']")
    num_auto_vehicles = len(auto_vehicle_elements)

    # Count the number of all vehicles in the XML file
    all_vehicle_elements = xml_file.findall(".//vehicle")
    num_all_vehicles = len(all_vehicle_elements)

    return xml_file, route_id, num_auto_vehicles, num_all_vehicles, all_vehicle_elements



# Define a file for logging the results
results_file = "pso_results.csv"

# Initialize variables for tracking PSO results
pso_results = []

# Initialize DataFrame columns
columns = ['Step', 'Normal Vehicles', 'Autonomous Vehicles(Particles)', 'Vehicle Type', 'Fitness', 'Travel Time', 'Distance(m)', 'Speed(km/h)','Interaction Score','Interaction Details', 'Interaction Information', 'Global Best Fitness', 'Global Best Position']
df = pd.DataFrame(columns=columns)



def load_vehicle_parameters_from_xml(xml_file):
    vehicle_params = {}
    vType_elements = xml_file.findall(".//vType")
    for vType in vType_elements:
        v_id = vType.get("id")
        accel = float(vType.get("accel"))
        decel = float(vType.get("decel"))
        max_speed = float(vType.get("maxSpeed"))
        vehicle_params[v_id] = {'accel': accel, 'decel': decel, 'maxSpeed': max_speed}
    return vehicle_params



def load_route_from_xml(xml_file, route_id):
    route_elements = xml_file.findall(f".//route[@id='{route_id}']")
    if len(route_elements) > 0:
        route_edges = route_elements[0].get("edges")
        return route_edges.split()
    else:
        return []



# Define PSO Parameters
# Set num_particles to the count of "auto" vehicles
xml_file, route_id, num_auto_vehicles, num_all_vehicles, all_vehicle_elements = load_network_data()

# Set num_particles to the count of "auto" vehicles
num_particles = num_auto_vehicles

num_iterations = 100
inertia_weight = 0.7
c1 = 1.5  # Cognitive coefficient
c2 = 2.0  # Social coefficient

# Set the initial values of w1 and w2
w1 = 1.0
w2 = 1.0

# Define the increment step for w1 and w2
w1_increment = 0.01
w2_increment = 0.01

# Initialize Particle Positions and Velocities
particles = []
particle_velocities = []


for _ in range(num_particles):
    # Initialize particle with random values
    particle = {
        'accel': random.uniform(0.1, 5.0),
        'decel': random.uniform(0.1, 5.0),
        'maxSpeed': random.uniform(10, 30)
    }
    particles.append(particle)

    # Initialize particle velocities
    velocity = {
        'accel': random.uniform(-1.0, 1.0),
        'decel': random.uniform(-1.0, 1.0),
        'maxSpeed': random.uniform(-5, 5)
    }
    particle_velocities.append(velocity)

# Define the Fitness Function
def fitness_function(particle):
    # Set vehicle parameters for all vehicles
    for veh_id in traci.vehicle.getIDList():
        # traci.vehicle.setAccel(veh_id, particle['accel'])
        # traci.vehicle.setDecel(veh_id, particle['decel'])
        # traci.vehicle.setMaxSpeed(veh_id, particle['maxSpeed'])

        # Check and clamp acceleration, deceleration, and maxSpeed based on the vehicle type
        vehicle_type = traci.vehicle.getTypeID(veh_id)
        if vehicle_type == 'car':
            accel_min = 0.1
            accel_max = 5.0
            decel_min = 0.1
            decel_max = 5.0
            max_speed_min = 1.0
            max_speed_max = 30.0
        elif vehicle_type == 'ev':
            accel_min = 0.1
            accel_max = 5.0
            decel_min = 0.1
            decel_max = 5.0
            max_speed_min = 1.0
            max_speed_max = 60.0
        else:
            # Define ranges for other vehicle types
            accel_min = 0.1
            accel_max = 5.0
            decel_min = 0.1
            decel_max = 5.0
            max_speed_min = 1.0
            max_speed_max = 30.0

        # Clamp acceleration, deceleration, and maxSpeed
        accel_clamped = min(max(particle['accel'], accel_min), accel_max)
        decel_clamped = min(max(particle['decel'], decel_min), decel_max)
        max_speed_clamped = min(max(particle['maxSpeed'], max_speed_min), max_speed_max)

        traci.vehicle.setAccel(veh_id, accel_clamped)
        traci.vehicle.setDecel(veh_id, decel_clamped)
        traci.vehicle.setMaxSpeed(veh_id, max_speed_clamped)



    # Run the simulation
    traci.simulationStep()

    # Calculate travel time for all vehicles
    travel_times = []
    for veh_id in traci.vehicle.getIDList():
        time = traci.simulation.getTime()  # Get the current simulation time
        edge_id = traci.vehicle.getRoadID(veh_id)  # Get the current edge ID for the vehicle
        adapted_travel_time = traci.vehicle.getAdaptedTraveltime(veh_id, time, edge_id)
        travel_times.append(adapted_travel_time)

    # Calculate the fitness value as the sum of travel times for all vehicles
    fitness_value = sum(travel_times)

    return fitness_value


def run():
    step = 0
    global particles
    global global_best_fitness
    global xml_file

    # Load vehicle parameters and routes from the XML file
    vehicle_params = load_vehicle_parameters_from_xml(xml_file)
    route_edges = load_route_from_xml(xml_file, route_id)  

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        print(step)

        # Update particle positions and velocities using PSO algorithm
        for i in range(num_particles):
            particle = particles[i]
            velocity = particle_velocities[i]

            # Calculate fitness for the current particle
            current_fitness = fitness_function(particle)

            # Update particle's best-known position and fitness
            if current_fitness < best_known_fitness[i]:
                best_known_fitness[i] = current_fitness
                best_known_position[i] = particle.copy()

            # Find the global best position and fitness
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle.copy()

            # Update particle velocities and positions
            for param in particle:
                r1, r2 = random.random(), random.random()
                velocity[param] = (
                    inertia_weight * velocity[param] +
                    c1 * r1 * (best_known_position[i][param] - particle[param]) +
                    c2 * r2 * (global_best_position[param] - particle[param])
                )
                particle[param] += velocity[param]
            
            # Set vehicle parameters for all vehicles
            for veh_id in traci.vehicle.getIDList():
                # traci.vehicle.setAccel(veh_id, particle['accel'])
                # traci.vehicle.setDecel(veh_id, particle['decel'])
                # traci.vehicle.setMaxSpeed(veh_id, particle['maxSpeed'])


                # Check and clamp acceleration, deceleration, and maxSpeed based on the vehicle type
                vehicle_type = traci.vehicle.getTypeID(veh_id)
                if vehicle_type == 'car':
                    accel_min = 0.1
                    accel_max = 5.0
                    decel_min = 0.1
                    decel_max = 5.0
                    max_speed_min = 1.0
                    max_speed_max = 30.0
                elif vehicle_type == 'ev':
                    accel_min = 0.1
                    accel_max = 5.0
                    decel_min = 0.1
                    decel_max = 5.0
                    max_speed_min = 1.0
                    max_speed_max = 60.0
                else:
                    # Define ranges for other vehicle types
                    accel_min = 0.1
                    accel_max = 5.0
                    decel_min = 0.1
                    decel_max = 5.0
                    max_speed_min = 1.0
                    max_speed_max = 30.0

            # Clamp parameter values within valid ranges
            particle['accel'] = min(max(particle['accel'], accel_min), accel_max)
            particle['decel'] = min(max(particle['decel'], decel_min), decel_max)
            particle['maxSpeed'] = min(max(particle['maxSpeed'], max_speed_min), max_speed_max)
            
            # Update vehicle parameters based on the particle
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.setAccel(veh_id, particle['accel'])
                traci.vehicle.setDecel(veh_id, particle['decel'])
                traci.vehicle.setMaxSpeed(veh_id, particle['maxSpeed'])

            # Log the results for this iteration
            pso_results.append([step, current_fitness, global_best_fitness, global_best_position.copy()])

        step += 1

    traci.close()
    sys.stdout.flush()


    # Save the PSO results to a CSV file
    with open(results_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Step', 'Current Fitness', 'Global Best Fitness', 'Global Best Position'])
        csvwriter.writerows(pso_results)

# Initialize variables for best-known positions and fitness
best_known_position = [particle.copy() for particle in particles]
best_known_fitness = [float('inf')] * num_particles
global_best_position = particles[0].copy()
global_best_fitness = float('inf')

# main entry point
if __name__ == "__main__":
    options = get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    xml_file, route_id, num_auto_vehicles, num_all_vehicles, all_vehicle_elements = load_network_data()

    traci.start([sumoBinary, "-c", "demo_network.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    run(num_steps=500, xml_file=xml_file)



















#     #!/usr/bin/env python

# import os
# import sys
# import optparse
# import random

# import csv

# # we need to import some python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")


# from sumolib import checkBinary  # Checks for the binary in environ vars
# import traci


# def get_options():
#     opt_parser = optparse.OptionParser()
#     opt_parser.add_option("--nogui", action="store_true",
#                          default=False, help="run the commandline version of sumo")
#     options, args = opt_parser.parse_args()
#     return options


# # Define a file for logging the results
# results_file = "pso_results.csv"

# # Initialize variables for tracking PSO results
# pso_results = []



# def load_vehicle_parameters_from_xml(xml_file):
#     vehicle_params = {}
#     vType_elements = xml_file.findall(".//vType")
#     for vType in vType_elements:
#         v_id = vType.get("id")
#         accel = float(vType.get("accel"))
#         decel = float(vType.get("decel"))
#         max_speed = float(vType.get("maxSpeed"))
#         vehicle_params[v_id] = {'accel': accel, 'decel': decel, 'maxSpeed': max_speed}
#     return vehicle_params



# def load_route_from_xml(xml_file, route_id):
#     route_elements = xml_file.findall(f".//route[@id='{route_id}']")
#     if len(route_elements) > 0:
#         route_edges = route_elements[0].get("edges")
#         return route_edges.split()
#     else:
#         return []



# # Define PSO Parameters
# num_particles = 10
# num_iterations = 100
# inertia_weight = 0.7
# c1 = 1.5  # Cognitive coefficient
# c2 = 2.0  # Social coefficient

# # Set the initial values of w1 and w2
# w1 = 1.0
# w2 = 1.0

# # Define the increment step for w1 and w2
# w1_increment = 0.01
# w2_increment = 0.01



# # Initialize Particle Positions and Velocities
# particles = []
# particle_velocities = []


# # Main loop for the PSO iterations
# for iteration in range(num_iterations):
#     # Update the weights w1 and w2
#     w1 = min(w1 + w1_increment, 2.0)
#     w2 = min(w2 + w2_increment, 2.0)

# for _ in range(num_particles):
#     # Initialize particle with random values
#     particle = {
#         'accel': random.uniform(0.1, 5.0),
#         'decel': random.uniform(0.1, 5.0),
#         'maxSpeed': random.uniform(10, 30)
#     }
#     particles.append(particle)

#     # Initialize particle velocities
#     velocity = {
#         'accel': random.uniform(-1.0, 1.0),
#         'decel': random.uniform(-1.0, 1.0),
#         'maxSpeed': random.uniform(-5, 5)
#     }
#     particle_velocities.append(velocity)



# def calculate_interaction_scores():
#     # Initialize an empty list to store interaction scores
#     interaction_scores = []

#     # Get a list of vehicle IDs
#     vehicle_ids = traci.vehicle.getIDList()

#     # Iterate through all pairs of vehicles (excluding self-interaction)
#     for i in range(len(vehicle_ids)):
#         vehicle_id_i = vehicle_ids[i]

#         for j in range(i + 1, len(vehicle_ids)):
#             vehicle_id_j = vehicle_ids[j]

#             # Get vehicle attributes
#             accel_i = traci.vehicle.getAccel(vehicle_id_i)
#             decel_i = traci.vehicle.getDecel(vehicle_id_i)
#             max_speed_i = traci.vehicle.getMaxSpeed(vehicle_id_i)
#             accel_j = traci.vehicle.getAccel(vehicle_id_j)
#             decel_j = traci.vehicle.getDecel(vehicle_id_j)
#             max_speed_j = traci.vehicle.getMaxSpeed(vehicle_id_j)

#             # Calculate interaction score between vehicle i and vehicle j
#             interaction_score = 0

#             # Example scoring logic (you can customize this)
#             if accel_i < accel_j:
#                 interaction_score += 1  # Vehicle i has lower acceleration
#             elif accel_i > accel_j:
#                 interaction_score -= 1  # Vehicle i has higher acceleration

#             if decel_i < decel_j:
#                 interaction_score += 1  # Vehicle i has lower deceleration
#             elif decel_i > decel_j:
#                 interaction_score -= 1  # Vehicle i has higher deceleration

#             if max_speed_i < max_speed_j:
#                 interaction_score += 1  # Vehicle i has lower max speed
#             elif max_speed_i > max_speed_j:
#                 interaction_score -= 1  # Vehicle i has higher max speed

#             # Append the interaction score to the list
#             interaction_scores.append(interaction_score)

#     return interaction_scores

# # def travel_times(): 2
#       # Calculate travel time for all vehicles
#     travel_times = []
#     for veh_id in traci.vehicle.getIDList():
#         time = traci.simulation.getTime()
#         edge_id = traci.vehicle.getRoadID(veh_id)
#         adapted_travel_time = traci.vehicle.getAdaptedTraveltime(veh_id, time, edge_id)
#         travel_times.append(adapted_travel_time)

#     return travel_times


# # def travel_times(): 3
#     # Calculate travel time for all vehicles and store them in a dictionary
#     vehicle_travel_times = {}
#     for veh_id in traci.vehicle.getIDList():
#         time = traci.simulation.getTime()
#         edge_id = traci.vehicle.getRoadID(veh_id)
#         adapted_travel_time = traci.vehicle.getAdaptedTraveltime(veh_id, time, edge_id)
#         vehicle_travel_times[veh_id] = adapted_travel_time

#     # Calculate the average travel time
#     if len(vehicle_travel_times) > 0:
#         average_travel_time = sum(vehicle_travel_times.values()) / len(vehicle_travel_times)
#     else:
#         average_travel_time = 0.0  # Handle the case when no vehicles are present

#     return vehicle_travel_times, average_travel_time

# def travel_times():
#     # Calculate travel time for all vehicles and store them in a dictionary
#     vehicle_travel_times = {}
#     for veh_id in traci.vehicle.getIDList():
#         # Get the current simulation time
#         time = traci.simulation.getTime()

#         # Get the departure time for the vehicle
#         depart_time = traci.vehicle.getDeparture(veh_id)

#         # Calculate the travel time for the vehicle
#         if depart_time >= 0:
#             travel_time = time - depart_time
#             vehicle_travel_times[veh_id] = travel_time

#     # Calculate the average travel time
#     if vehicle_travel_times:
#         average_travel_time = sum(vehicle_travel_times.values()) / len(vehicle_travel_times)
#     else:
#         average_travel_time = 0.0  # Handle the case when no vehicles are present

#     return vehicle_travel_times, average_travel_time




# # Define the Fitness Function
# def fitness_function(particle, vehicle_travel_times, average_travel_time):
#     # Set vehicle parameters for all vehicles
#     for veh_id in traci.vehicle.getIDList():
#         # traci.vehicle.setAccel(veh_id, particle['accel'])
#         # traci.vehicle.setDecel(veh_id, particle['decel'])
#         # traci.vehicle.setMaxSpeed(veh_id, particle['maxSpeed'])

#         # Check and clamp acceleration, deceleration, and maxSpeed based on the vehicle type
#         vehicle_type = traci.vehicle.getTypeID(veh_id)
#         if vehicle_type == 'car':
#             accel_min = 0.1
#             accel_max = 5.0
#             decel_min = 0.1
#             decel_max = 5.0
#             max_speed_min = 1.0
#             max_speed_max = 30.0
#         elif vehicle_type == 'ev':
#             accel_min = 0.1
#             accel_max = 5.0
#             decel_min = 0.1
#             decel_max = 5.0
#             max_speed_min = 1.0
#             max_speed_max = 60.0
#         else:
#             # Define ranges for other vehicle types
#             accel_min = 0.1
#             accel_max = 5.0
#             decel_min = 0.1
#             decel_max = 5.0
#             max_speed_min = 1.0
#             max_speed_max = 30.0

#         # Clamp acceleration, deceleration, and maxSpeed
#         accel_clamped = min(max(particle['accel'], accel_min), accel_max)
#         decel_clamped = min(max(particle['decel'], decel_min), decel_max)
#         max_speed_clamped = min(max(particle['maxSpeed'], max_speed_min), max_speed_max)

#         traci.vehicle.setAccel(veh_id, accel_clamped)
#         traci.vehicle.setDecel(veh_id, decel_clamped)
#         traci.vehicle.setMaxSpeed(veh_id, max_speed_clamped)



#     # Run the simulation
#     traci.simulationStep()


#     #calculate the interaction score between vehicles
#     # Initialize an empty list to store interaction scores
#     interaction_scores = []

#     # Get a list of vehicle IDs
#     vehicle_ids = traci.vehicle.getIDList()

#     # Iterate through all pairs of vehicles (excluding self-interaction)
#     for i in range(len(vehicle_ids)):
#         vehicle_id_i = vehicle_ids[i]

#         for j in range(i + 1, len(vehicle_ids)):
#             vehicle_id_j = vehicle_ids[j]

#             # Get vehicle attributes
#             accel_i = traci.vehicle.getAccel(vehicle_id_i)
#             decel_i = traci.vehicle.getDecel(vehicle_id_i)
#             max_speed_i = traci.vehicle.getMaxSpeed(vehicle_id_i)
#             accel_j = traci.vehicle.getAccel(vehicle_id_j)
#             decel_j = traci.vehicle.getDecel(vehicle_id_j)
#             max_speed_j = traci.vehicle.getMaxSpeed(vehicle_id_j)

#             # Calculate interaction score between vehicle i and vehicle j
#             interaction_score = 0

#             # Example scoring logic (you can customize this)
#             if accel_i < accel_j:
#                 interaction_score += 1  # Vehicle i has lower acceleration
#             elif accel_i > accel_j:
#                 interaction_score -= 1  # Vehicle i has higher acceleration

#             if decel_i < decel_j:
#                 interaction_score += 1  # Vehicle i has lower deceleration
#             elif decel_i > decel_j:
#                 interaction_score -= 1  # Vehicle i has higher deceleration

#             if max_speed_i < max_speed_j:
#                 interaction_score += 1  # Vehicle i has lower max speed
#             elif max_speed_i > max_speed_j:
#                 interaction_score -= 1  # Vehicle i has higher max speed

#             # Append the interaction score to the list
#             interaction_scores.append(interaction_score)

#     # Calculate the fitness value using the weighted sum
#     fitness_value = w1 * sum([1 / travel_time for travel_time in vehicle_travel_times.values()]) + w2 * sum(interaction_scores)

#     # Append interaction scores and travel times to pso_results
#     pso_results.append([interaction_scores, list(vehicle_travel_times.values())]) 

#     return fitness_value






# def run():
#     step = 0
#     global particles
#     global global_best_fitness
#     global xml_file

#     # Load vehicle parameters and routes from the XML file
#     vehicle_params = load_vehicle_parameters_from_xml(xml_file)
#     route_edges = load_route_from_xml(xml_file, route_id)

#     while traci.simulation.getMinExpectedNumber() > 0:
#         traci.simulationStep()
#         print("Step:", step)

#         # Update particle positions and velocities using PSO algorithm
#         for i in range(num_particles):
#             particle = particles[i]
#             velocity = particle_velocities[i]


#             # Inside your run function
#             vehicle_travel_times, average_travel_time = travel_times()
#             print("Travel Times (Individual):", vehicle_travel_times)
#             print("Average Travel Time:", average_travel_time)

#             # Calculate fitness for the current particle
#             current_fitness = fitness_function(particle, vehicle_travel_times, average_travel_time)

#             # Update particle's best-known position and fitness
#             if current_fitness < best_known_fitness[i]:
#                 best_known_fitness[i] = current_fitness
#                 best_known_position[i] = particle.copy()

#             # Find the global best position and fitness
#             if current_fitness < global_best_fitness:
#                 global_best_fitness = current_fitness
#                 global_best_position = particle.copy()

#             # Update particle velocities and positions
#             for param in particle:
#                 r1, r2 = random.random(), random.random()
#                 velocity[param] = (
#                     inertia_weight * velocity[param] +
#                     c1 * r1 * (best_known_position[i][param] - particle[param]) +
#                     c2 * r2 * (global_best_position[param] - particle[param])
#                 )
#                 particle[param] += velocity[param]

#         # Calculate interaction scores and the fitness value
#             interaction_scores = calculate_interaction_scores()  # You should implement this function
#             fitness_value = w1 * sum([1 / travel_time for travel_time in vehicle_travel_times.values()]) + w2 * sum(interaction_scores)

#             # Print the values
#             print("Particle", i, "Fitness:", current_fitness)
#             print("Travel Times:", vehicle_travel_times)
#             print("Interaction Scores:", interaction_scores)
#             print("Cumulative Fitness:", fitness_value)

#         # Append interaction scores and travel times to pso_results
#         pso_results.append([interaction_scores, vehicle_travel_times])

#         step += 1

#     traci.close()
#     sys.stdout.flush()

#     # Save the PSO results to a CSV file
#     with open(results_file, 'w', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(['Step', 'Current Fitness', 'Global Best Fitness', 'Global Best Position'])
#         csvwriter.writerows(pso_results)


# # Initialize variables for best-known positions and fitness
# best_known_position = [particle.copy() for particle in particles]
# best_known_fitness = [float('inf')] * num_particles
# global_best_position = particles[0].copy()
# global_best_fitness = float('inf')


# # main entry point
# if __name__ == "__main__":
#     options = get_options()

#     # check binary
#     if options.nogui:
#         sumoBinary = checkBinary('sumo')
#     else:
#         sumoBinary = checkBinary('sumo-gui')

#     # Load the XML file with vehicle and route definitions
#     import xml.etree.ElementTree as ET
#     xml_file = ET.parse("demo_network.rou.xml")  # Replace with the path to your XML file
#     route_id = "route_0"  # Specify the route ID you want to use

#     # traci starts sumo as a subprocess and then this script connects and runs
#     traci.start([sumoBinary, "-c", "demo_network.sumocfg",
#                              "--tripinfo-output", "tripinfo.xml"])
#     run()



















# def fitness_function(step, particle, particle_best_fitness, particle_best_position, global_best_fitness, global_best_position, vehicle_travel_times, interaction_scores, interaction_details):
#     fitness_values = {}

#     for i, veh_id in enumerate(traci.vehicle.getIDList()):
#         try:
#             vehicle_type = traci.vehicle.getTypeID(veh_id)
            
#             if vehicle_type == 'car':
#                 accel_min = 0.1
#                 accel_max = 5.0
#                 decel_min = 0.1
#                 decel_max = 5.0
#                 max_speed_min = 1.0
#                 max_speed_max = 15.0
#                 pass

#             elif vehicle_type == 'auto':
#                 # PSO Update Formula
#                 for key in particle.keys():
#                     particle_velocities[i][key] = (
#                         inertia_weight * particle_velocities[i][key] +
#                         c1 * random.random() * (particle_best_position[i][key] - particles[i][key]) +
#                         c2 * random.random() * (global_best_position[key] - particles[i][key])
#                     )
#                     particles[i][key] += particle_velocities[i][key]

#                 # Apply vehicle-specific parameters
#                 accel_min = 0.1
#                 accel_max = 5.0
#                 decel_min = 0.1
#                 decel_max = 5.0
#                 max_speed_min = 1.0
#                 max_speed_max = 18.6 if vehicle_type == 'auto' else 15.0

#                 accel_clamped = min(max(particle['accel'], accel_min), accel_max)
#                 decel_clamped = min(max(particle['decel'], decel_min), decel_max)
#                 max_speed_clamped = min(max(particle['maxSpeed'], max_speed_min), max_speed_max)

#                 traci.vehicle.setAccel(veh_id, accel_clamped)
#                 traci.vehicle.setDecel(veh_id, decel_clamped)
#                 traci.vehicle.setMaxSpeed(veh_id, max_speed_clamped)

#                 traci.simulationStep()

#                 distances = distance()
#                 speeds = speed()
#                 vehicle_interaction_score = interaction_scores.get(veh_id, 0.0)
#                 vehicle_travel_time = vehicle_travel_times.get(veh_id, 0.0)
#                 fitness_value = w1 * (1 / vehicle_travel_time) + w2 * vehicle_interaction_score

#                 interaction_info = get_interaction_info(veh_id)

#                 fitness_values[veh_id] = fitness_value

#                 pso_results.append([
#                     step,
#                     veh_id,
#                     fitness_value,  # Corrected variable name
#                     vehicle_travel_times[veh_id],
#                     interaction_scores[veh_id],
#                     interaction_info,
#                     None,  # Global Best Fitness
#                     None,   # Global Best Position
#                     particle_velocities[i][key],
#                     particle_best_position[i][key],
#                     particles[i][key],
#                     global_best_position[key]
#                 ])

#                 df = df.append({
#                     'Step': step,
#                     'Normal Vehicles': veh_id if vehicle_type == 'car' else None,
#                     'Autonomous Vehicles(Particles)': veh_id if vehicle_type == 'auto' else None,
#                     'Fitness': fitness_value,  # Corrected variable name
#                     'Travel Time': vehicle_travel_times.get(veh_id, None),
#                     'Distance(m)': distances.get(veh_id, None),
#                     'Speed(km/h)': speeds.get(veh_id, None),
#                     'Interaction Score': interaction_scores.get(veh_id, None),
#                     'Interaction Information': interaction_info,
#                     'Interaction Details': interaction_details.get(veh_id, None),
#                     'Global Best Fitness': None,
#                     'Global Best Position': None,
#                     'Particle Velocity': particle_velocities[i][key],
#                     'Particle Best Position': particle_best_position[i][key],
#                     'Particle': particles[i][key],
#                     'Global Best Position': global_best_position[key]
#                 }, ignore_index=True)

#                 if veh_id not in particle_best_fitness[i] or fitness_value < particle_best_fitness[i][veh_id]:
#                     particle_best_fitness[i][veh_id] = fitness_value
#                     particle_best_position[i] = particle.copy()

#                 if fitness_value < global_best_fitness:
#                     global_best_fitness = fitness_value
#                     global_best_position = particle.copy()

#         except Exception as e:
#             continue

#     return fitness_values
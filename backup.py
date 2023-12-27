#!/usr/bin/env python

import os
import sys
import optparse
import numpy as np
import matplotlib.pyplot as plt
from sumolib import checkBinary
import traci
import xml.etree.ElementTree as ET
import random


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
columns = ['Step', 'Normal Vehicles', 'Autonomous Vehicles(Particles)', 'Vehicle Type', 'Fitness', 'Travel Time', 'Distance(m)', 'Speed(km/h)','Interaction Score','Interaction Details', 'Interaction Information', 'Global Best Fitness', 'Global Best Position', 'Particle Velocity', 'Particle Best Position', 'Particle', 'Global Best Position']
df = pd.DataFrame(columns=columns)


# Define PSO Parameters
# Set num_particles to the count of "auto" vehicles
xml_file, route_id, num_auto_vehicles, num_all_vehicles, all_vehicle_elements = load_network_data()

# PSO parameters
SWARM_SIZE = 20
MAX_ITERATIONS = 50
INERTIA = 0.5
LOCAL_WEIGHT = 2
GLOBAL_WEIGHT = 2
SLOWDOWN_INTERVAL = 200
BURN_IN_PERIOD = 20
SIMULATION_STEPS = 80
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


def collect_id_list():
    # Collect all vehicle IDs
    all_vehicle_ids = traci.vehicle.getIDList()

    return all_vehicle_ids


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


def calculate_individual_interaction_score(vehicle_id, all_vehicle_ids):
    collision_follower_avoided = 0
    collision_leader_avoided = 0
    collision_neighbor_avoided = 0
    change_lane_due_to_block = 0
    interaction_details = []
    MIN_FOLLOW_DISTANCE = 2
    MIN_LEADER_DISTANCE = 2
    MIN_LANE_CHANGE_DISTANCE = 2
    MAX_SPEED_DIFF = 15

    if vehicle_id not in traci.vehicle.getIDList():
        return collision_follower_avoided, collision_leader_avoided, collision_neighbor_avoided, change_lane_due_to_block, interaction_details

    x1, y1 = traci.vehicle.getPosition(vehicle_id)
    vehicle_type = traci.vehicle.getTypeID(vehicle_id).split('@')[0]


    # Check for too close following normal vehicle
    follower_id, _ = traci.vehicle.getFollower(vehicle_id)
    if follower_id and follower_id in traci.vehicle.getIDList():
        x2, y2 = traci.vehicle.getPosition(follower_id)
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        follower_type = traci.vehicle.getTypeID(follower_id).split('@')[0]

        # if vehicle_type == 'auto' and follower_type == 'car':
        if vehicle_type == 'auto':

            # Get the Distance of the autonomous vehicle
            vehicle_distance = traci.vehicle.getDistance(vehicle_id)
            follower_distance = traci.vehicle.getDistance(follower_id)

            # Get the distance difference of the autonomous vehicle
            distance_difference = vehicle_distance - follower_distance
            distance_difference = abs(distance_difference)
            print(f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Follower Vehicle*****")
            print(f"AV{vehicle_id} Distance Traveled is {vehicle_distance} m.\n")
            print(f"V{follower_id} Distance Traveled is {follower_distance} m.\n")
            print(f"DD b/w AV{vehicle_id} and follower NV{follower_id}: {distance_difference} m.\n") 

            # Get the speed of the autonomous vehicle
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            follower_speed = traci.vehicle.getSpeed(follower_id)

            # Get the distance difference of the autonomous vehicle
            speed_difference = vehicle_speed - follower_speed
            speed_difference = abs(speed_difference)
            print(f"AV{vehicle_id} Speed is {vehicle_speed} km/h.\n")
            print(f"V{follower_id} Speed is {follower_speed} km/h.\n")
            print(f"SD b/w AV{vehicle_id} and follower NV{follower_id}: {speed_difference} km/h.")
            print(f"****************************************************************************************\n \n \n")

            # if vehicle_type == 'auto' and follower_type == 'car':
            if vehicle_type == 'auto':
                if distance_difference <= MIN_FOLLOW_DISTANCE:
                    print(f"---------------------------If Distance between AV and Following Vehicle too close-------------------------")
                    print(f"Autonomous Vehicle {vehicle_id} is too close to Following Vehicle({follower_id}),\n distance: {distance_difference} m.\n") 
                    # If autonomous vehicle speed is greater than or equal to 5, decrease speed by 1
                    if vehicle_speed <= 15:
                        print(f"-------------------------If Speed between AV is Not Standard---------------------------")
                        traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5.0)
                        print(f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 5} km/h.\n")
                        
                        # Calculate new distance
                        new_distance = distance_difference + 10
                        print( f"Autonomous Vehicle{vehicle_id} increased distance from Following Normal Vehicle, new distance from Following Vehicle: {new_distance} m.\n")
                        collision_follower_avoided +=1
                        interaction_details.append("AV successfully increased speed and distance from FV which was too close.\n")
                        print(f"collision_follower_avoided Incremented.")
                        print(f"--------------------------------------------------------------------------------\n \n \n")
                    else:
                        print(f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n")
                        collision_follower_avoided +=1
                        interaction_details.append("AV speed is acceptable, maintaining distance from FV which was too close.\n")
                        print(f"collision_follower_avoided Incremented.")
                        print(f"--------------------------------------------------------------------------------\n \n \n") 
                        # new_distance = traci.vehicle.getDistance(vehicle_id)                 
                else:
                     print(f"----------------------------Distance Adequate b/w AV and Following Vehicle----------------------------") 
                     print(f"Autonomous Vehicle{vehicle_id} maintaining distance from Following Vehicle distance: {distance_difference} m. \n \n \n")
                     collision_follower_avoided +=1
                     interaction_details.append("AV maintaining distance from FV.\n")
                     print(f"collision_follower_avoided Incremented.")
                     print(f"--------------------------------------------------------------------------------------------------\n \n \n") 
            else:
                pass
        else:
            pass
    else:
        pass

    # Check for too close leading autonomous vehicle
    leader_id_info = traci.vehicle.getLeader(vehicle_id)
    if leader_id_info is not None:
        leader_id, leader_distance = leader_id_info
        if leader_id in traci.vehicle.getIDList():
            leader_type = traci.vehicle.getTypeID(leader_id).split('@')[0]

            # if vehicle_type == 'auto' and leader_type == 'auto':
            if vehicle_type == 'auto':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                leader_distance = traci.vehicle.getDistance(leader_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - leader_distance
                distance_difference = abs(distance_difference)
                print(f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Leader Vehicle*****")
                print(f"AV{vehicle_id} Distance Traveled is {vehicle_distance} m.\n")
                print(f"V{leader_id} Distance Traveled is {leader_distance} m.\n")
                print(f"DD b/w AV{vehicle_id} and Leading AV{leader_id}: {distance_difference} m.\n")

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - leader_speed
                speed_difference = abs(speed_difference)
                print(f"AV{vehicle_id} Speed is {vehicle_speed} km/h.\n")
                print(f"V{leader_id} Speed is {leader_speed} km/h.\n")
                print(f"SD b/w AV{vehicle_id} and Leader AV{leader_id}: {speed_difference} km/h.")
                print(f"********************************************************************************************\n \n \n") 


                # if vehicle_type == 'auto' and leader_type == 'car':
                if vehicle_type == 'auto':

                    if distance_difference <= MIN_LEADER_DISTANCE:
                        print(f"---------------------------If Distance between AV and Leading Vehicle too close-------------------------")
                        print(f"Autonomous Vehicle {vehicle_id} is too close to the Leading Vehicle({leader_id}),\n distance: {distance_difference} m.\n") 
                        # If autonomous vehicle speed is greater than or equal to 18, decrease speed to 18
                        if vehicle_speed >= 10:
                            print(f"-------------------------If Speed between AV is Not Standard-------------------------------")
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5.0)
                            print(f"Autonomous Vehicle {vehicle_id} decreased speed to {vehicle_speed - 2} km/h.\n") 
                            
                            # Calculate new distance
                            new_leader_distance = distance_difference + 10
                            print(f"Autonomous Vehicle {vehicle_id} increased distance from Leading Autonomous Vehicle{leader_id}, new distance: {new_leader_distance} m.\n")
                            collision_leader_avoided +=1
                            interaction_details.append("AV successfully decreased speed and distance from LV which was too close.\n")
                            print(f"collision_leader_avoided Incremented.") 
                            print(f"----------------------------------------------------------------------------------------\n\n\n")
                        else:
                            print(f"Autonomous Vehicle {vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n") 
                            # new_leader_distance = traci.vehicle.getDistance(vehicle_id)
                            collision_leader_avoided +=1
                            interaction_details.append("AV speed is acceptable, maintaining distance from LV which was too close.\n")
                            print(f"collision_leader_avoided Incremented.")
                            print(f"---------------------------------------------------------------------------------------------\n\n\n")
                    else:
                        print(f"-------------------------Distance Adequate b/w AV and Neighboring Vehicle-------------------------------") 
                        print(f"Autonomous Vehicle {vehicle_id} maintaining distance From Leading Autonomous Vehicle{leader_id}: {distance_difference} km/h.\n \n \n") 
                        collision_leader_avoided +=1
                        interaction_details.append("AV maintaining distance from LV.\n")
                        print(f"collision_leader_avoided Incremented.")
                        print(f"----------------------------------------------------------------------------------------------------------\n\n\n")
                else:
                    pass

                # Check for significantly faster leading autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)
                speed_difference = vehicle_speed - leader_speed
                if vehicle_type == 'auto':
                    if vehicle_speed > MAX_SPEED_DIFF:
                        leader_type = traci.vehicle.getTypeID(leader_id).split('@')[0]
                        if vehicle_type == 'auto':
                            print(f"Autonomous Vehicle {vehicle_id} is significantly faster than the Leading Vehicle({leader_id}),\n speed difference: {speed_difference} km/h.\n") 
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed - 2.0)  # Decrease speed to increase distance
                            new_speed_difference = vehicle_speed - leader_speed
                            print(f"Autonomous Vehicle {vehicle_id} is Speed Decreased,\n new speed difference: {new_speed_difference} km/h.\n")
                            collision_leader_avoided +=1
                            interaction_details.append("AV Successfully adjsuted its speed from being too fast.\n")
                            print(f"collision_leader_avoided Incremented.") 
                            # new_speed_difference = vehicle_speed - traci.vehicle.getSpeed(leader_id)
                            # info += f"Autonomous Vehicle {vehicle_id} increased distance from significantly faster than Leading Autonomous Vehicle, new speed difference: {new_speed_difference}\n \n \n"
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
        else:
            pass
    else:
        pass

    # Check for too close neighbor normal vehicle in a different lane
    neighbors = traci.vehicle.getNeighbors(vehicle_id, 5)
    for neighbor_id, _ in neighbors:
        if neighbor_id in traci.vehicle.getIDList():
            x2, y2 = traci.vehicle.getPosition(neighbor_id)
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            neighbor_type = traci.vehicle.getTypeID(neighbor_id).split('@')[0]

            # if vehicle_type == 'auto' and neighbor_type == 'car':
            if vehicle_type == 'auto':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                neighbor_distance = traci.vehicle.getDistance(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - neighbor_distance
                distance_difference = abs(distance_difference)
                print(f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Neighbor Vehicle*****")
                print(f"AV{vehicle_id} Distance Traveled is {vehicle_distance} m.\n")
                print(f"V{neighbor_id} Distance Traveled is {neighbor_distance} m.\n")
                print(f"DD b/w AV{vehicle_id} and neighboring NV{neighbor_id}: {distance_difference} m.\n") 

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                neighbor_speed = traci.vehicle.getSpeed(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - neighbor_speed
                speed_difference = abs(speed_difference)
                print(f"AV{vehicle_id} Speed is {vehicle_speed} km/h.\n")
                print(f"V{neighbor_id} Speed is {neighbor_speed} km/h.\n")
                print(f"SD b/w AV{vehicle_id} and neighboring NV{neighbor_id}: {speed_difference} km/h.") 
                print(f"**********************************************************************************************\n \n \n")

                if vehicle_type == 'auto':
                    if distance_difference < MIN_LANE_CHANGE_DISTANCE:
                        print(f"---------------------------If Distance between AV and Neighboring Vehicle too close-------------------------")
                        print(f"Autonomous Vehicle{vehicle_id} is too close to a Neighbor Normal Vehicle({neighbor_id}) in a different lane,\n distance: {distance_difference} m.\n") 

                        # If autonomous vehicle speed is greater than or equal to 25, decrease speed by 1
                        if vehicle_speed >= 15:
                            print(f"-------------------------If Speed between AV is Not Standard-------------------------------")
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5)
                            print(f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 5} km/h.\n") 
                            # Calculate new distance
                            new_distance = speed_difference + 10
                            print(f"Autonomous Vehicle{vehicle_id} increased distance from Neighbor Normal Vehicle in a different lane, new distance: {new_distance} m.\n \n \n")
                            collision_neighbor_avoided +=1
                            interaction_details.append("AV successfully increased it distance and speed from NV being too close to it.\n")
                            print(f"collision_neighbor_avoided Incremented.")
                            print(f"-------------------------------------------------------------------------------------------") 
                        else:
                            # If autonomous vehicle speed is less than 25, print actual speed and maintain distance
                            print(f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} m/s, maintaining distance.\n")
                            collision_neighbor_avoided +=1
                            interaction_details.append("AV speed is acceptable, maintaining distance from NV being too close to it.\n")
                            print(f"collision_neighbor_avoided Incremented.") 
                            # new_distance = traci.vehicle.getDistance(vehicle_id)
                            print(f"-------------------------------------------------------------------------------------------") 
                    else:
                        print(f"----------------------------Distance Adequate b/w AV and Neighboring Vehicle----------------------------") 
                        print(f"Autonomous Vehicle{vehicle_id} maintaining distance from Neighbor Normal vehicle, Distance: {distance_difference} m.\n \n \n")
                        collision_neighbor_avoided +=1
                        interaction_details.append("AV maintaining distance from NV.\n")
                        print(f"collision_neighbor_avoided Incremented.")
                        print(f"------------------------------------------------------------------------------------------------------------\n \n \n")

    # Check if the autonomous vehicle is blocked and change lane
    if not traci.vehicle.getNextTLS(vehicle_id) and not traci.vehicle.getLaneChangeState(vehicle_id, 0):  # 0 corresponds to 'left'
        # Get the best lane information
        if vehicle_type == 'auto':
            best_lanes = traci.vehicle.getBestLanes(vehicle_id)
            if best_lanes:
                best_lane_id = best_lanes[0][0]  # Choose the first best lane
                best_lane_direction = best_lanes[0][2]  # Direction information
                current_lane_id = traci.vehicle.getLaneID(vehicle_id)
                if best_lane_id != current_lane_id:
                    print(f"Autonomous Vehicle {vehicle_id} is blocked. Changing lane to {best_lane_id}.\n") 
                    traci.vehicle.changeLane(vehicle_id, best_lane_id, 10.0)  # Change to the best lane
                    traci.simulationStep()  # Allow the simulation to process the lane change
                    print(f"{vehicle_id} changed lane to {best_lane_id} in direction {best_lane_direction} to increase distance from Neighbor Normal Vehicle in a different lane\n \n \n")
                    change_lane_due_to_block += 1
                    interaction_details.append("AV changed lane in appropriate direction to increase distance from NV in a different lane.\n") 

    blockers = traci.vehicle.getNeighbors(vehicle_id, 1)
    for blocker_id, _ in blockers:
        if blocker_id != vehicle_id and blocker_id in traci.vehicle.getIDList():
            x2, y2 = traci.vehicle.getPosition(blocker_id)
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            blocker_type = traci.vehicle.getTypeID(blocker_id).split('@')[0]
            if vehicle_type == 'auto':
                change_lane_due_to_block += 1
                print(f"************************************Incase AV is Blocked************************************************")
                current_lane = traci.vehicle.getLaneID(vehicle_id)
                print(f"Autonomous Vehicle{vehicle_id} is blocked by Normal Vehicle{blocker_id} in lane {current_lane}, distance: {distance} m.\n")
                
                # # Get the index of the current lane in the lane list
                traci.vehicle.changeLane(vehicle_id, 0, duration=0000)
                next_lane = traci.vehicle.getLaneID(vehicle_id)
                print(f"Autonomous Vehicle{vehicle_id} Lane changed to {next_lane}.\n")
                change_lane_due_to_block +=1
                interaction_details.append("Autonomous Vehicle Lane changed to appropriate lane.\n")
                print(f"change_lane_due_to_block Incremented.\n")
                print(f"******************************************************************************************************\n\n\n")
                


    return collision_follower_avoided, collision_leader_avoided, collision_neighbor_avoided, change_lane_due_to_block, interaction_details


def calculate_interaction_scores():
    interaction_scores = {}
    interaction_details = {}

    all_vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in all_vehicle_ids:
        score, details = calculate_individual_interaction_score(vehicle_id, all_vehicle_ids)
        interaction_scores[vehicle_id] = score
        interaction_details[vehicle_id] = details

    return interaction_scores, interaction_details


def distance():
    distances = {}
    all_vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in all_vehicle_ids:
        distance = traci.vehicle.getDistance(vehicle_id)
        distances[vehicle_id] = distance

    return distances

def speed():
    speeds = {}
    all_vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in all_vehicle_ids:
        speed = traci.vehicle.getSpeed(vehicle_id)
        speeds[vehicle_id] = speed

    return speeds


def travel_times():
    vehicle_travel_times = {}
    for veh_id in traci.vehicle.getIDList():
        time = traci.simulation.getTime()
        depart_time = traci.vehicle.getDeparture(veh_id)
        if depart_time >= 0:
            travel_time = time - depart_time
            vehicle_travel_times[veh_id] = travel_time

    return vehicle_travel_times



def fitness_function(step, particle, particle_best_fitness, particle_best_position, global_best_fitness, global_best_position, vehicle_travel_times, interaction_scores, interaction_details):
    fitness_values = {}

    for i, veh_id in enumerate(traci.vehicle.getIDList()):
        try:
            vehicle_type = traci.vehicle.getTypeID(veh_id)
            if vehicle_type == 'car':
                # Your existing logic for 'car' type vehicles
                pass

            for i, veh_id in enumerate(traci.vehicle.getIDList()):
                try:
                    vehicle_type = traci.vehicle.getTypeID(veh_id)
                    if vehicle_type == 'car':
                        accel_min = 0.1
                        accel_max = 5.0
                        decel_min = 0.1
                        decel_max = 5.0
                        max_speed_min = 1.0
                        max_speed_max = 15.0
                        pass
                    elif vehicle_type == 'auto':
                        accel_min = 0.1
                        accel_max = 5.0
                        decel_min = 0.1
                        decel_max = 5.0
                        max_speed_min = 1.0
                        max_speed_max = 18.6
                    else:
                        accel_min = 0.1
                        accel_max = 5.0
                        decel_min = 0.1
                        decel_max = 5.0
                        max_speed_min = 1.0
                        max_speed_max = 15.0

                    accel_clamped = min(max(particle['accel'], accel_min), accel_max)
                    decel_clamped = min(max(particle['decel'], decel_min), decel_max)
                    max_speed_clamped = min(max(particle['maxSpeed'], max_speed_min), max_speed_max)

                    traci.vehicle.setAccel(veh_id, accel_clamped)
                    traci.vehicle.setDecel(veh_id, decel_clamped)
                    traci.vehicle.setMaxSpeed(veh_id, max_speed_clamped)

                    traci.simulationStep()

                    distances = distance() 
                    speeds = speed()  
                    vehicle_interaction_score = interaction_scores.get(veh_id, 0.0)
                    vehicle_travel_time = vehicle_travel_times.get(veh_id, 0.0)
                    fitness_value = w1 * (1 / vehicle_travel_time) + w2 * vehicle_interaction_score

                    interaction_info = get_interaction_info(veh_id)

                    fitness_values[veh_id] = fitness_value

                    if veh_id not in particle_best_fitness[i] or fitness_value < particle_best_fitness[i][veh_id]:  
                        particle_best_fitness[i][veh_id] = fitness_value  
                        particle_best_position[i] = particle.copy()

                    if fitness_value < global_best_fitness:  
                        global_best_fitness = fitness_value
                        global_best_position = particle.copy()                    

                    pso_results.append([
                        step,
                        veh_id,
                        fitness_value,  # Corrected variable name
                        vehicle_travel_times[veh_id],
                        interaction_scores[veh_id],
                        interaction_info,
                        None,  # Global Best Fitness
                        None,   # Global Best Position
                        particle_velocities[i][key],
                        particle_best_position[i][key],
                        particles[i][key],
                        global_best_position[key]
                    ])

                    df = df.append({
                        'Step': step,
                        'Normal Vehicles': veh_id if vehicle_type == 'car' else None,
                        'Autonomous Vehicles(Particles)': veh_id if vehicle_type == 'auto' else None,
                        'Fitness': fitness_value,  # Corrected variable name
                        'Travel Time': vehicle_travel_times.get(veh_id, None),
                        'Distance(m)' : distances.get(veh_id, None),
                        'Speed(km/h)' : speeds.get(veh_id, None),
                        'Interaction Score': interaction_scores.get(veh_id, None),
                        'Interaction Information': interaction_info,
                        'Interaction Details': interaction_details.get(veh_id, None),
                        'Global Best Fitness': None,
                        'Global Best Position': None,
                        'Particle Velocity': particle_velocities[i][key],
                        'Particle Best Position': particle_best_position[i][key],
                        'Particle': particles[i][key],
                        'Global Best Position': global_best_position[key]
                    }, ignore_index=True)



                except Exception as e:
                    continue
        except Exception as e:
            continue

    return fitness_values


def evaluate_particle(particle):
    for step in range(SIMULATION_STEPS):
        # print(f"Simulation Step {step}")
        traci.simulationStep()

        if step % SLOWDOWN_INTERVAL == 0:
            for i in range(SWARM_SIZE):
                vehicle_id = f"pso_{i}"
                if traci.vehicle.getIDList() and vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setSpeedMode(vehicle_id, 0b000001000)
                else:
                #     print(f"Vehicle {vehicle_id} not found.")
                  print("List of vehicles:", traci.vehicle.getIDList())

        for i in range(SWARM_SIZE):
            try:
                vehicle_id = f"pso_{i}"
                if vehicle_id in traci.vehicle.getIDList():
                    traci.vehicle.setSpeed(vehicle_id, particle[0])
                    traci.vehicle.setMinGap(vehicle_id, particle[1])
                    traci.vehicle.setParameter(vehicle_id, "slowDown", str(particle[2]))
                # else:
                #     print(f"Vehicle {vehicle_id} not found.")
                    print("List of vehicles:", traci.vehicle.getIDList())
            except traci.exceptions.TraCIException as e:
                print(f"Error setting parameters for {vehicle_id}: {e}")

        # Check if the list of vehicles is empty and end the simulation
        if not traci.vehicle.getIDList():
            print("List of vehicles is empty. Ending simulation.")
            break

        # Ensure that the simulation stops when SIMULATION_STEPS is reached
        if step == SIMULATION_STEPS - 1:
            break

    # Ensure that the simulation stops
    for _ in range(10):
        traci.simulationStep()

    fitness = get_fitness()
    print(f"Final Fitness: {fitness}")
    return fitness




def get_interaction_info(vehicle_id):
    MIN_FOLLOW_DISTANCE = 100
    MIN_LEADER_DISTANCE = 100
    MIN_LANE_CHANGE_DISTANCE = 100
    MAX_SPEED_DIFF = 5

    info = ""

    if vehicle_id not in traci.vehicle.getIDList():
        return info  # Return an empty string if the vehicle is not known

    x1, y1 = traci.vehicle.getPosition(vehicle_id)
    vehicle_type = traci.vehicle.getTypeID(vehicle_id).split('@')[0]

    # Check for too close following normal vehicle
    follower_id, _ = traci.vehicle.getFollower(vehicle_id)
    if follower_id and follower_id in traci.vehicle.getIDList():
        x2, y2 = traci.vehicle.getPosition(follower_id)
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        follower_type = traci.vehicle.getTypeID(follower_id).split('@')[0]

        # if vehicle_type == 'auto' and follower_type == 'car':
        if vehicle_type == 'auto':

            # Get the Distance of the autonomous vehicle
            vehicle_distance = traci.vehicle.getDistance(vehicle_id)
            follower_distance = traci.vehicle.getDistance(follower_id)

            # Get the distance difference of the autonomous vehicle
            distance_difference = vehicle_distance - follower_distance
            distance_difference = abs(distance_difference)
            info+=f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Follower Vehicle*****\n"
            info+=f"AV{vehicle_id} Distance Traveled is {vehicle_distance} m.\n"
            info+=f"V{follower_id} Distance Traveled is {follower_distance} m.\n"
            info+=f"DD b/w AV{vehicle_id} and follower NV{follower_id}: {distance_difference} m.\n" 

            # Get the speed of the autonomous vehicle
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            follower_speed = traci.vehicle.getSpeed(follower_id)

            # Get the distance difference of the autonomous vehicle
            speed_difference = vehicle_speed - follower_speed
            speed_difference = abs(speed_difference)
            info+=f"AV{vehicle_id} Speed is {vehicle_speed} km/h.\n"
            info+=f"V{follower_id} Speed is {follower_speed} km/h.\n"
            info+=f"SD b/w AV{vehicle_id} and follower NV{follower_id}: {speed_difference} km/h.\n"
            info+=f"****************************************************************************************\n \n \n"

            # if vehicle_type == 'auto' and follower_type == 'car':
            if vehicle_type == 'auto':
                if distance_difference <= MIN_FOLLOW_DISTANCE:
                    info+=f"---------------------------If Distance between AV and Following Vehicle too close-------------------------\n"
                    info+=f"Autonomous Vehicle {vehicle_id} is too close to Following Normal Vehicle({follower_id}),\n distance: {distance_difference} m.\n" 
                    # If autonomous vehicle speed is greater than or equal to 5, decrease speed by 1
                    if vehicle_speed <= 15:
                        info+=f"-------------------------If Speed between AV is Not Standard---------------------------\n"
                        traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5.0)
                        info+=f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 5} km/h.\n"
                        
                        # Calculate new distance
                        new_distance = distance_difference + 10
                        info+= f"Autonomous Vehicle{vehicle_id} increased distance from Following Normal Vehicle,\n new distance from Following Normal Vehicle: {new_distance} m.\n"
                
                        info+=f"****Interaction Incremented*****\n"
                        info+=f"--------------------------------------------------------------------------------\n \n \n"
                    else:
                        info+=f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n"
                        info+=f"--------------------------------------------------------------------------------\n \n \n" 
                        # new_distance = traci.vehicle.getDistance(vehicle_id)                 
                else:
                     info+=f"----------------------------Distance Adequate b/w AV and Following Vehicle----------------------------\n" 
                     info+=f"Autonomous Vehicle{vehicle_id} maintaining distance from Following Normal Vehicle distance: {distance_difference} m. \n"
                     info+=f"--------------------------------------------------------------------------------------------------\n \n \n"
            else:
                pass
        else:
            pass
    else:
        pass
    # Check for too close leading autonomous vehicle
    leader_id_info = traci.vehicle.getLeader(vehicle_id)
    if leader_id_info is not None:
        leader_id, leader_distance = leader_id_info
        if leader_id in traci.vehicle.getIDList():
            leader_type = traci.vehicle.getTypeID(leader_id).split('@')[0]

            # if vehicle_type == 'auto' and leader_type == 'auto':
            if vehicle_type == 'auto':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                leader_distance = traci.vehicle.getDistance(leader_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - leader_distance
                distance_difference = abs(distance_difference)
                info+=f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Leader Vehicle*****\n"
                info+=f"AV{vehicle_id} Distance Traveled is {vehicle_distance} m.\n"
                info+=f"V{leader_id} Distance Traveled is {leader_distance} m.\n"
                info+=f"DD b/w AV{vehicle_id} and Leading AV{leader_id}: {distance_difference} m.\n"

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - leader_speed
                speed_difference = abs(speed_difference)
                info+=f"AV{vehicle_id} Speed is {vehicle_speed} km/h.\n"
                info+=f"V{leader_id} Speed is {leader_speed} km/h.\n"
                info+=f"SD b/w AV{vehicle_id} and Leader AV{leader_id}: {speed_difference} km/h.\n"
                info+=f"********************************************************************************************\n \n \n" 


                # if vehicle_type == 'auto' and leader_type == 'car':
                if vehicle_type == 'auto':

                    if distance_difference <= MIN_LEADER_DISTANCE:
                        info+=f"---------------------------If Distance between AV and Leading Vehicle too close-------------------------\n"
                        info+=f"Autonomous Vehicle {vehicle_id} is too close to the Leading Vehicle({leader_id}),\n distance: {distance_difference} m.\n"
                        # If autonomous vehicle speed is greater than or equal to 18, decrease speed to 18
                        if vehicle_speed >= 10:
                            info+=f"-------------------------If Speed between AV is Not Standard-------------------------------\n"
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5.0)
                            info+=f"Autonomous Vehicle {vehicle_id} decreased speed to {vehicle_speed - 2} km/h.\n"
                            
                            # Calculate new distance
                            new_leader_distance = distance_difference + 10
                            info+=f"Autonomous Vehicle {vehicle_id} increased distance from Leading Autonomous Vehicle{leader_id},\n new distance: {new_leader_distance} m.\n"
                    
                            info+=f"****Interaction Incremented*****\n"
                            info+=f"---------------------------------------------------------------------------------------------\n\n\n"
                        else:
                            info+=f"Autonomous Vehicle {vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n"
                            # new_leader_distance = traci.vehicle.getDistance(vehicle_id)
                            info+=f"---------------------------------------------------------------------------------------------\n\n\n"
                    else:
                        info+=f"-------------------------Distance Adequate b/w AV and Neighboring Vehicle-------------------------------\n"
                        info+=f"Autonomous Vehicle {vehicle_id} maintaining distance From Leading Autonomous Vehicle{leader_id}: {distance_difference} km/h.\n"
                        info+=f"----------------------------------------------------------------------------------------------------------\n\n\n"
                else:
                    pass

                # Check for significantly faster leading autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)
                speed_difference = vehicle_speed - leader_speed
                if vehicle_type == 'auto' and leader_type == 'auto':
                    if speed_difference > MAX_SPEED_DIFF:
                        leader_type = traci.vehicle.getTypeID(leader_id).split('@')[0]
                        if vehicle_type == 'auto' and leader_type == 'auto':
                            info+=f"Autonomous Vehicle {vehicle_id} is significantly faster than the Leading Autonomous Vehicle({leader_id}),\n speed difference: {speed_difference} km/h.\n" 
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed - 2.0)  # Decrease speed to increase distance
                            new_speed_difference = vehicle_speed - leader_speed
                            info+=f"Autonomous Vehicle {vehicle_id} is Speed Decreased,\n new speed difference: {new_speed_difference} km/h.\n"
                    
                            info+=f"****Interaction Incremented*****\n\n"
                            # new_speed_difference = vehicle_speed - traci.vehicle.getSpeed(leader_id)
                            # info += f"Autonomous Vehicle {vehicle_id} increased distance from significantly faster than Leading Autonomous Vehicle, new speed difference: {new_speed_difference}\n \n \n"
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
        else:
            pass
    else:
        pass
    # Check for too close neighbor normal vehicle in a different lane
    neighbors = traci.vehicle.getNeighbors(vehicle_id, 5)
    for neighbor_id, _ in neighbors:
        if neighbor_id in traci.vehicle.getIDList():
            x2, y2 = traci.vehicle.getPosition(neighbor_id)
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            neighbor_type = traci.vehicle.getTypeID(neighbor_id).split('@')[0]

            # if vehicle_type == 'auto' and neighbor_type == 'car':
            if vehicle_type == 'auto':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                neighbor_distance = traci.vehicle.getDistance(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - neighbor_distance
                distance_difference = abs(distance_difference)
                info+=f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Neighbor Vehicle*****\n"
                info+=f"AV{vehicle_id} Distance Traveled is {vehicle_distance} m.\n"
                info+=f"V{neighbor_id} Distance Traveled is {neighbor_distance} m.\n"
                info+=f"DD b/w AV{vehicle_id} and neighboring NV{neighbor_id}: {distance_difference} m.\n" 

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                neighbor_speed = traci.vehicle.getSpeed(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - neighbor_speed
                speed_difference = abs(speed_difference)
                info+=f"AV{vehicle_id} Speed is {vehicle_speed} km/h.\n"
                info+=f"V{neighbor_id} Speed is {neighbor_speed} km/h.\n"
                info+=f"SD b/w AV{vehicle_id} and neighboring NV{neighbor_id}: {speed_difference} km/h." 
                info+=f"**********************************************************************************************\n \n \n"

                if vehicle_type == 'auto':
                    if distance_difference < MIN_LANE_CHANGE_DISTANCE:
                        info+=f"---------------------------If Distance between AV and Neighboring Vehicle too close-------------------------\n"
                        info+=f"Autonomous Vehicle{vehicle_id} is too close to a Neighbor Normal Vehicle({neighbor_id}) in a different lane,\n distance: {distance_difference} m.\n" 

                        # If autonomous vehicle speed is greater than or equal to 25, decrease speed by 1
                        if vehicle_speed >= 15:
                            info+=f"-------------------------If Speed between AV is Not Standard-------------------------------\n"
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5)
                            info+=f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 5} km/h.\n"
                            # Calculate new distance
                            new_distance = speed_difference + 10
                            info+=f"Autonomous Vehicle{vehicle_id} increased distance from Neighbor Normal Vehicle in a different lane,\n new distance: {new_distance} m.\n"
                    
                            info+=f"****Interaction Incremented*****\n"
                            info+=f"------------------------------------------------------------------------------------------- \n \n \n" 
                        else:
                            # If autonomous vehicle speed is less than 25, print actual speed and maintain distance
                            info+=f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} m/s, maintaining distance.\n"
                            # new_distance = traci.vehicle.getDistance(vehicle_id)
                            info+=f"-------------------------------------------------------------------------------------------\n \n \n" 
                    else:
                        info+=f"----------------------------Distance Adequate b/w AV and Neighboring Vehicle---------------------------- \n" 
                        info+=f"Autonomous Vehicle{vehicle_id} maintaining distance from Neighbor Normal vehicle, Distance: {distance_difference} m.\n"
                        info+=f"---------------------------------------------------------------------------------------------------------\n \n \n"

    # Check if the autonomous vehicle is blocked and change lane
    if not traci.vehicle.getLaneChangeState(vehicle_id, 0):  # 0 corresponds to 'left'
        # Get the best lane information
        if vehicle_type == 'auto':
            best_lanes = traci.vehicle.getBestLanes(vehicle_id)
            if best_lanes:
                best_lane_id = best_lanes[0][0]  # Choose the first best lane
                best_lane_direction = best_lanes[0][2]  # Direction information
                current_lane_id = traci.vehicle.getLaneID(vehicle_id)
                if best_lane_id != current_lane_id:
                    info+=f"Autonomous Vehicle {vehicle_id} is blocked. Changing lane to {best_lane_id}.\n" 
                    traci.vehicle.changeLane(vehicle_id, best_lane_id, 10.0)  # Change to the best lane
                    traci.simulationStep()  # Allow the simulation to process the lane change
                    info+=f"{vehicle_id} changed lane to {best_lane_id} in direction {best_lane_direction} to increase distance from Neighbor Normal Vehicle in a different lane\n"
    
    blockers = traci.vehicle.getNeighbors(vehicle_id, 1)
    for blocker_id, _ in blockers:
        if blocker_id != vehicle_id and blocker_id in traci.vehicle.getIDList():
            x2, y2 = traci.vehicle.getPosition(blocker_id)
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            blocker_type = traci.vehicle.getTypeID(blocker_id).split('@')[0]
            if vehicle_type == 'auto' and blocker_type == "car":
                
                info+=f"************************************Incase AV is Blocked************************************************\n"
                current_lane = traci.vehicle.getLaneID(vehicle_id)
                info+=f"Autonomous Vehicle{vehicle_id} is blocked by Vehicle{blocker_id} in lane {current_lane}, distance: {distance} m.\n"
                
                # # Get the index of the current lane in the lane list
                traci.vehicle.changeLane(vehicle_id, 0, duration=0000)
                next_lane = traci.vehicle.getLaneID(vehicle_id)
                info+=f"Autonomous Vehicle{vehicle_id} Lane changed to {next_lane}.\n"
        
                info+=f"****Interaction Incremented*****\n"
                info+=f"******************************************************************************************************\n"

    return info


def run(num_steps, xml_file):
    global particles
    global global_best_fitness
    global global_best_position
    global df

    # Initialize global best fitness and position
    global_best_fitness = float('inf')
    global_best_position = None

    # Initialize particle best fitness and position
    particle_best_fitness = [{}] * num_particles
    particle_best_position = [None] * num_particles


    step = 0
    while step < num_steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        print("Step:", step)

        for i in range(num_particles):
            particle = particles[i]

            distances = distance() 
            speeds = speed()
            vehicle_ids = traci.vehicle.getIDList()
            vehicle_travel_times = travel_times()
            interaction_scores, interaction_details = calculate_interaction_scores()

            fitness_values = fitness_function(step, particle, particle_best_fitness[i], particle_best_position[i], global_best_fitness, global_best_position, vehicle_travel_times, interaction_scores, interaction_details)

            # Update particle and global best fitness and position
            for veh_id, fitness in fitness_values.items():
                print("Step:", step)

                interaction_details_str = '\n'.join(f"{key}: {', '.join(details)}" for key, details in interaction_details.items())

                # Check if the vehicle is known before getting its information
                if veh_id in traci.vehicle.getIDList():
                    # Vehicle is known, get its information
                    vehicle_type_with_prefix = traci.vehicle.getTypeID(veh_id)
                    vehicle_type_parts = vehicle_type_with_prefix.split('@')
                    if len(vehicle_type_parts) == 2:
                        vehicle_type = vehicle_type_parts[0]
                    else:
                        vehicle_type = vehicle_type_with_prefix

                    print("Normal Vehicle:" if vehicle_type == 'car' else "Autonomous Vehicle (Particle):", veh_id)
                    print("Fitness:", fitness if vehicle_type == 'auto' else "")  # Print Fitness only for 'auto' type
                    print("Travel Time:", vehicle_travel_times.get(veh_id, None))
                    print("Interaction Score:", interaction_scores.get(veh_id, None))
                    print()  # Separate the vehicle output

                    interaction_info = get_interaction_info(veh_id)
                    if interaction_info:
                        # print("\nInteraction Information:\n", interaction_info, "\n")

                        # Append only if interaction_info is not empty
                        particle_results = [
                            step,
                            None if vehicle_type != 'car' else veh_id,
                            veh_id,
                            fitness if vehicle_type == 'auto' else "",  # Fill in only for 'auto' type
                            vehicle_travel_times.get(veh_id, None),
                            interaction_scores.get(veh_id, None),
                            '\n'.join(interaction_details.get(veh_id, [])),
                            interaction_details_str,
                            None,
                            None
                        ]
                        pso_results.append(particle_results)

                    df = df.append({
                        'Step': step,
                        'Normal Vehicles': None if vehicle_type != 'car' else veh_id,
                        'Autonomous Vehicles(Particles)': None if vehicle_type != 'auto' else veh_id,
                        'Vehicle Type': vehicle_type,
                        'Fitness': fitness if vehicle_type == 'auto' else "",  # Fill in only for 'auto' type
                        'Travel Time': vehicle_travel_times.get(veh_id, None),
                        'Distance(m)' : distances.get(veh_id, None),
                        'Speed(km/h)' : speeds.get(veh_id, None),
                        'Interaction Score': interaction_scores.get(veh_id, None),
                        'Interaction Details': interaction_details_str,
                        'Interaction Information': interaction_info, 
                        'Global Best Fitness': None,
                        'Global Best Position': None
                    }, ignore_index=True)

                    if veh_id not in particle_best_fitness[i] or fitness < particle_best_fitness[i][veh_id]:
                        particle_best_fitness[i][veh_id] = fitness
                        particle_best_position[i] = particle.copy()

                    if fitness < global_best_fitness:
                        vehicle_position = traci.vehicle.getPosition(veh_id)
                        global_best_position = vehicle_position if vehicle_position else global_best_position
                else:
                    print(f"Warning: Vehicle {veh_id} is not known.")

            step += 1

    if global_best_position:
        # Find the minimum fitness value among "auto" vehicles
        min_auto_fitness = min(df[df['Vehicle Type'].str.startswith('auto')]['Fitness'])

        # Update the row with global best fitness and position for the "auto" vehicle with the minimum fitness
        global_best_row = df[(df['Fitness'] == min_auto_fitness) & df['Vehicle Type'].str.startswith('auto')].iloc[0]
        df.at[global_best_row.name, 'Global Best Fitness'] = min_auto_fitness
        df.at[global_best_row.name, 'Global Best Position'] = f"{global_best_position[0]}, {global_best_position[1]}"

        # Print Global Best Information
        print("\nGlobal Best Information:")
        print(f"Step: {global_best_row['Step']}")
        print(f"Normal Vehicles: {global_best_row['Normal Vehicles']}")
        print(f"Autonomous Vehicles(Particles): {global_best_row['Autonomous Vehicles(Particles)']}")
        print(f"Global Best Fitness: {min_auto_fitness}")
        print(f"Global Best Position: {global_best_position}\n")

    # Save the pso_results to the CSV file
    df.to_csv(results_file, index=False)

    traci.close()
    sys.stdout.flush()

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








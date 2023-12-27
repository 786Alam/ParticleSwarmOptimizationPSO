import os
import sys
import optparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sumolib import checkBinary
import traci
import xml.etree.ElementTree as ET
import math
import random

# PSO parameters
SWARM_SIZE = 30
MAX_ITERATIONS = 10
INERTIA = 0.5
LOCAL_WEIGHT = 2
GLOBAL_WEIGHT = 2
SLOWDOWN_INTERVAL = 200
BURN_IN_PERIOD = 50
ITERATIONS = 300
PENALTY_PARAMETER = 2




# Define a file for logging the results
results_file = "pso_results.csv"

# Initialize variables for tracking PSO results
pso_results = []


# Initialize DataFrame columns
columns = ['Iteration', 'Normal Vehicles', 'Autonomous Vehicles(Particles)', 'Vehicle Type', 'Fitness', 'Travel Time', 'Distance(m)', 'Speed(km/h)','Collision with Follower avoided','Collision with Leader avoided','Collision with Neighbor avoided','Lane Change Due to Block','Interaction Details', 'Interaction Information', 'Global Best Fitness', 'Global Best Position', 'Particle Velocity', 'Particle Best Position', 'Particle', 'Global Best Position']
df = pd.DataFrame(columns=columns)


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
            pso_vehicles += f' <vehicle id="pso_{i}" type="auto" route="{route_id}" depart="{depart_time}" begin="100" end="1500" from="e1" to="e8" />\n'

    return pso_vehicles

# Initialize simulation
def initialize_simulation():
    options = get_options()
    if options.nogui:
        sumo_binary = checkBinary('sumo')
    else:
        sumo_binary = checkBinary('sumo-gui')

    try:
        traci.start([sumo_binary, "-c", "demo_network.sumocfg", "--random-depart-offset", "100", "--tripinfo-output", "tripinfo.xml"])
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

    # Wait for a few simulation steps to allow dynamically added vehicles to appear
    for _ in range(100):
        traci.simulationStep()

    print("List of vehicles after addition:", traci.vehicle.getIDList())  # Print list of vehicles for debugging
    print("List of routes after addition:", traci.route.getIDList())  # Print list of routes for debugging

    traci.simulationStep()

def display_entered_and_left_vehicles(iteration):
    all_vehicles = traci.vehicle.getIDList()
    entered_vehicles = traci.simulation.getArrivedIDList()
    left_vehicles = traci.simulation.getDepartedIDList()

    # print(f"Iteration {iteration}:")
    # print("All vehicles:", all_vehicles)
    # print("")



# Calculate Collisions and ways to deal with them
def calculate_individual_interaction_score(vehicle_id):
    collision_follower_avoided = 0
    collision_leader_avoided = 0
    collision_neighbor_avoided = 0
    change_lane_due_to_block = 0
    interaction_details = []
    MIN_FOLLOW_DISTANCE = 5
    MIN_LEADER_DISTANCE = 10
    MIN_LANE_CHANGE_DISTANCE = 10
    MAX_SPEED_DIFF = 5 

    MIN_DISTANCE_PENALTY = 2
    SPEED_DIFF_PENALTY = 1.5
    COLLISION_PENALTY = 3
    fitness_penalty_follow=0
    fitness_penalty_lead=0
    fitness_penalty_neighbor=0
    fitness_penalty_collision=0

    # normal_vehicles = [vehicle_id for vehicle_id in traci.vehicle.getIDList() if vehicle_id.startswith("n_")]
    # autonomous_vehicles = [vehicle_id for vehicle_id in traci.vehicle.getIDList() if vehicle_id.startswith("pso_")]

    if vehicle_id not in traci.vehicle.getIDList():
        return collision_follower_avoided, collision_leader_avoided, collision_neighbor_avoided, change_lane_due_to_block, interaction_details

    x1, y1 = traci.vehicle.getPosition(vehicle_id)
    vehicle_type = traci.vehicle.getTypeID(vehicle_id)


    # Check for too close following normal vehicle
    follower_id, _ = traci.vehicle.getFollower(vehicle_id)
    if follower_id and follower_id in traci.vehicle.getIDList():
        x2, y2 = traci.vehicle.getPosition(follower_id)
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # follower_type = traci.vehicle.getTypeID(follower_id).split('@')[0]

        if vehicle_id.startswith('pso_') and (follower_id.startswith('pso_') or follower_id.startswith('n_')):
            # Your existing code here

            # Get the Distance of the autonomous vehicle
            vehicle_distance = traci.vehicle.getDistance(vehicle_id)
            follower_distance = traci.vehicle.getDistance(follower_id)

            # Get the distance difference of the autonomous vehicle
            distance_difference = vehicle_distance - follower_distance
            distance_difference = abs(distance_difference)
            # print(f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Follower Vehicle*****")
            # print(f"AV {vehicle_id} Distance Traveled is {vehicle_distance} m.\n")
            # print(f"NV {follower_id} Distance Traveled is {follower_distance} m.\n")
            # print(f"DD b/w AV {vehicle_id} and follower NV {follower_id}: {distance_difference} m.\n") 

            # Get the speed of the autonomous vehicle
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            follower_speed = traci.vehicle.getSpeed(follower_id)

            # Get the distance difference of the autonomous vehicle
            speed_difference = vehicle_speed - follower_speed
            speed_difference = abs(speed_difference)
            # print(f"AV {vehicle_id} Speed is {vehicle_speed} km/h.\n")
            # print(f"V {follower_id} Speed is {follower_speed} km/h.\n")
            # print(f"SD b/w AV {vehicle_id} and follower NV {follower_id}: {speed_difference} km/h.")
            # print(f"****************************************************************************************\n \n \n")

            # if vehicle_type == 'auto' and follower_type == 'car':
            if vehicle_id.startswith('pso_') and (follower_id.startswith('pso_') or follower_id.startswith('n_')):
                if distance_difference > MIN_FOLLOW_DISTANCE:
                    # print(f"---------------------------If Distance between AV and Following Vehicle too close-------------------------")
                    # print(f"Autonomous Vehicle {vehicle_id} is too close to Following Vehicle {follower_id},\n distance: {distance_difference} m.\n") 
                    penalty = MIN_DISTANCE_PENALTY / (distance_difference - MIN_FOLLOW_DISTANCE)
                    fitness_penalty_follow += penalty
                    # print(f"Penalty: {penalty} for violating minimum follow distance.\n")
                    # If autonomous vehicle speed is greater than or equal to 5, decrease speed by 1
                    if speed_difference > MAX_SPEED_DIFF:
                        # print(f"-------------------------If Speed between AV is Not Standard---------------------------")
                        penalty = SPEED_DIFF_PENALTY / (speed_difference - MAX_SPEED_DIFF)
                        fitness_penalty_follow += penalty
                        # print(f"Penalty: {penalty} for high speed difference.\n")
                        traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5.0)
                        # print(f"Autonomous Vehicle {vehicle_id} increased speed to {vehicle_speed + 5} km/h.\n")
                        
                        # Calculate new distance
                        new_distance = distance_difference + 10
                        # print( f"Autonomous Vehicle {vehicle_id} increased distance from Following Normal Vehicle, new distance from Following Vehicle: {new_distance} m.\n")
                        interaction_details.append("AV successfully increased speed and distance from FV which was too close.\n")
                        # print(f"collision_follower_avoided Incremented.")
                        # print(f"--------------------------------------------------------------------------------\n \n \n")
                    else:
                        # print(f"Autonomous Vehicle {vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n")
                        collision_follower_avoided +=1
                        interaction_details.append("AV speed is acceptable, maintaining distance from FV which was too close.\n")
                        # print(f"collision_follower_avoided Incremented.")
                        # print(f"--------------------------------------------------------------------------------\n \n \n") 
                        # new_distance = traci.vehicle.getDistance(vehicle_id)                 
                else:
                    #  print(f"----------------------------Distance Adequate b/w AV and Following Vehicle----------------------------") 
                    #  print(f"Autonomous Vehicle {vehicle_id} maintaining distance from Following Vehicle distance: {distance_difference} m. \n \n \n")
                     collision_follower_avoided +=1
                     interaction_details.append("AV maintaining distance from FV.\n")
                    #  print(f"collision_follower_avoided Incremented.")
                    #  print(f"--------------------------------------------------------------------------------------------------\n \n \n") 
            else:
                pass
        else:
            pass
    else:
        pass

    leader_id_info = traci.vehicle.getLeader(vehicle_id)
    if leader_id_info is not None:
        leader_id, leader_distance = leader_id_info
        if leader_id in traci.vehicle.getIDList():
            leader_type = traci.vehicle.getTypeID(leader_id).split('@')[0]

            if vehicle_id.startswith('pso_') and (leader_id.startswith('pso_') or leader_id.startswith('n_')):


                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                leader_distance = traci.vehicle.getDistance(leader_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - leader_distance
                distance_difference = abs(distance_difference)
                # print(f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Leader Vehicle*****")
                # print(f"AV {vehicle_id} Distance Traveled is {vehicle_distance} m.\n")
                # print(f"V {leader_id} Distance Traveled is {leader_distance} m.\n")
                # print(f"DD b/w AV {vehicle_id} and Leading V {leader_id}: {distance_difference} m.\n")

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - leader_speed
                speed_difference = abs(speed_difference)
                # print(f"AV {vehicle_id} Speed is {vehicle_speed} km/h.\n")
                # print(f"V {leader_id} Speed is {leader_speed} km/h.\n")
                # print(f"SD b/w AV {vehicle_id} and Leader AV {leader_id}: {speed_difference} km/h.")
                # print(f"********************************************************************************************\n \n \n") 


                # if vehicle_type == 'auto' and leader_type == 'car' or 'auto':
                if vehicle_id.startswith('pso') and (leader_id.startswith('pso_') or leader_id.startswith('n_')):
                    if distance_difference > MIN_LEADER_DISTANCE:
                        # print(f"---------------------------If Distance between AV and Leading Vehicle too close-------------------------")
                        # print(f"Autonomous Vehicle {vehicle_id} is too close to the Leading Vehicle {leader_id},\n distance: {distance_difference} m.\n") 
                        penalty = MIN_DISTANCE_PENALTY / (distance_difference - MIN_LEADER_DISTANCE)
                        fitness_penalty_lead += penalty
                        # print(f"Penalty: {penalty} for violating minimum follow distance.\n")
            
   
                        # If autonomous vehicle speed is greater than or equal to 18, decrease speed to 18
                        if speed_difference > MAX_SPEED_DIFF:
                            # print(f"-------------------------If Speed between AV is Not Standard-------------------------------")
                            penalty = SPEED_DIFF_PENALTY / (speed_difference - MAX_SPEED_DIFF)
                            fitness_penalty_lead += penalty
                            # print(f"Penalty: {penalty} for high speed difference.\n")
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5.0)
                            # print(f"Autonomous Vehicle {vehicle_id} decreased speed to {vehicle_speed - 2} km/h.\n") 
                            
                            # Calculate new distance
                            new_leader_distance = distance_difference + 10
                            # print(f"Autonomous Vehicle {vehicle_id} increased distance from Leading Autonomous Vehicle {leader_id}, new distance: {new_leader_distance} m.\n")
                            interaction_details.append("AV successfully decreased speed and distance from LV which was too close.\n")
                            # print(f"collision_leader_avoided Incremented.") 
                            # print(f"----------------------------------------------------------------------------------------\n\n\n")
                        else:
                            # print(f"Autonomous Vehicle {vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n") 
                            # new_leader_distance = traci.vehicle.getDistance(vehicle_id)
                            collision_leader_avoided +=1
                            interaction_details.append("AV speed is acceptable, maintaining distance from LV which was too close.\n")
                            # print(f"collision_leader_avoided Incremented.")
                            # print(f"---------------------------------------------------------------------------------------------\n\n\n")
                    else:
                        # print(f"-------------------------Distance Adequate b/w AV and Neighboring Vehicle-------------------------------") 
                        # print(f"Autonomous Vehicle {vehicle_id} maintaining distance From Leading Autonomous Vehicle {leader_id}: {distance_difference} km/h.\n \n \n") 
                        collision_leader_avoided +=1
                        interaction_details.append("AV maintaining distance from LV.\n")
                        # print(f"collision_leader_avoided Incremented.")
                        # print(f"----------------------------------------------------------------------------------------------------------\n\n\n")
                else:
                    pass

                # Check for significantly faster leading autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)
                speed_difference = vehicle_speed - leader_speed
                if vehicle_id.startswith('pso_'):
                    if speed_difference >= MAX_SPEED_DIFF:
                        leader_type = traci.vehicle.getTypeID(leader_id)
                        if vehicle_type == 'auto':
                            print(f"Autonomous Vehicle {vehicle_id} is significantly faster than the Leading Vehicle {leader_id},\n speed difference: {speed_difference} km/h.\n") 
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

    neighbors = traci.vehicle.getNeighbors(vehicle_id, 5)
    for neighbor_id, _ in neighbors:
        if neighbor_id in traci.vehicle.getIDList():
            x2, y2 = traci.vehicle.getPosition(neighbor_id)
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            neighbor_type = traci.vehicle.getTypeID(neighbor_id).split('@')[0]

            if vehicle_id.startswith('pso_') and (neighbor_id.startswith('pso_') or neighbor_id.startswith('n_')):

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                neighbor_distance = traci.vehicle.getDistance(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - neighbor_distance
                distance_difference = abs(distance_difference)
                # print(f"*****Distance & Speed Traveled difference b/w Autonomous Vehicle and Neighbor Vehicle*****")
                # print(f"AV {vehicle_id} Distance Traveled is {vehicle_distance} m.\n")
                # print(f"V {neighbor_id} Distance Traveled is {neighbor_distance} m.\n")
                # print(f"DD b/w AV {vehicle_id} and neighboring NV {neighbor_id}: {distance_difference} m.\n") 

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                neighbor_speed = traci.vehicle.getSpeed(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - neighbor_speed
                speed_difference = abs(speed_difference)
                # print(f"AV{vehicle_id} Speed is {vehicle_speed} km/h.\n")
                # print(f"V{neighbor_id} Speed is {neighbor_speed} km/h.\n")
                # print(f"SD b/w AV {vehicle_id} and neighboring V {neighbor_id}: {speed_difference} km/h.") 
                # print(f"**********************************************************************************************\n \n \n")

                if vehicle_id.startswith('pso_') and (neighbor_id.startswith('pso_') or neighbor_id.startswith('n_')):
                    if distance_difference > MIN_LANE_CHANGE_DISTANCE:
                        # print(f"---------------------------If Distance between AV and Neighboring Vehicle too close-------------------------")
                        penalty = MIN_DISTANCE_PENALTY / (distance_difference - MIN_FOLLOW_DISTANCE)
                        fitness_penalty_neighbor += penalty
                        # print(f"Penalty: {penalty} for violating minimum follow distance.\n")
                        # print(f"Autonomous Vehicle {vehicle_id} is too close to a Neighbor Normal Vehicle {neighbor_id} in a different lane,\n distance: {distance_difference} m.\n") 

                        # If autonomous vehicle speed is greater than or equal to 25, decrease speed by 1
                        if speed_difference > MAX_SPEED_DIFF:
                            penalty = SPEED_DIFF_PENALTY / (speed_difference - MAX_SPEED_DIFF)
                            fitness_penalty_neighbor += penalty
                            # print(f"Penalty: {penalty} for high speed difference.\n")
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 5)
                            # print(f"Autonomous Vehicle {vehicle_id} increased speed to {vehicle_speed + 5} km/h.\n") 
                            # Calculate new distance
                            new_distance = speed_difference + 10
                            # print(f"Autonomous Vehicle {vehicle_id} increased distance from Neighbor Normal Vehicle in a different lane, new distance: {new_distance} m.\n \n \n")
                            interaction_details.append("AV successfully increased it distance and speed from NV being too close to it.\n")
                            # print(f"collision_neighbor_avoided Incremented.")
                            # print(f"-------------------------------------------------------------------------------------------") 
                        else:
                            # If autonomous vehicle speed is less than 25, print actual speed and maintain distance
                            # print(f"Autonomous Vehicle {vehicle_id} actual speed is {vehicle_speed} m/s, maintaining distance.\n")
                            collision_neighbor_avoided +=1
                            interaction_details.append("AV speed is acceptable, maintaining distance from NV being too close to it.\n")
                            # print(f"collision_neighbor_avoided Incremented.") 
                            # new_distance = traci.vehicle.getDistance(vehicle_id)
                            # print(f"-------------------------------------------------------------------------------------------") 
                    else:
                        # print(f"----------------------------Distance Adequate b/w AV and Neighboring Vehicle----------------------------") 
                        # print(f"Autonomous Vehicle {vehicle_id} maintaining distance from Neighbor Normal vehicle, Distance: {distance_difference} m.\n \n \n")
                        collision_neighbor_avoided +=1
                        interaction_details.append("AV maintaining distance from NV.\n")
                        # print(f"collision_neighbor_avoided Incremented.")
                        # print(f"------------------------------------------------------------------------------------------------------------\n \n \n")


    blockers = traci.vehicle.getNeighbors(vehicle_id, 1)
    for blocker_id, _ in blockers:
        if blocker_id != vehicle_id and blocker_id in traci.vehicle.getIDList():
            x2, y2 = traci.vehicle.getPosition(blocker_id)
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            blocker_type = traci.vehicle.getTypeID(blocker_id).split('@')[0]

            # print(f"************************************Incase AV is Blocked************************************************")
            if vehicle_id.startswith('pso_') and (blocker_id.startswith('pso_') or blocker_id.startswith('n_')):

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                blocker_distance = traci.vehicle.getDistance(blocker_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - blocker_distance
                distance_difference = distance_difference

                if distance_difference < 0:
                    current_lane = traci.vehicle.getLaneID(vehicle_id)
                    # print(f"Autonomous Vehicle {vehicle_id} is blocked by Normal Vehicle {blocker_id} in lane {current_lane}, distance: {distance} m.\n")
                    penalty = COLLISION_PENALTY / (distance_difference - MIN_LANE_CHANGE_DISTANCE)
                    fitness_penalty_collision += penalty
                    # print(f"Penalty: {penalty} for violating minimum follow distance.\n")
                    # # Get the index of the current lane in the lane list
                    traci.vehicle.changeLane(vehicle_id, 0, duration=0000)
                    next_lane = traci.vehicle.getLaneID(vehicle_id)
                    # print(f"Autonomous Vehicle {vehicle_id} Lane changed to {next_lane}.\n")
                    change_lane_due_to_block +=1
                    interaction_details.append("Autonomous Vehicle Lane changed to the appropriate lane.\n")
                    # print(f"change_lane_due_to_block Incremented.\n")
                    # print(f"******************************************************************************************************\n\n\n")
                else:
                    current_lane = traci.vehicle.getLaneID(vehicle_id)
                    # print(f"Autonomous Vehicle {vehicle_id} is blocked by Normal Vehicle {blocker_id} in lane {current_lane}, distance: {distance} m.\n")
                    # # Get the index of the current lane in the lane list
                    traci.vehicle.changeLane(vehicle_id, 0, duration=0000)
                    next_lane = traci.vehicle.getLaneID(vehicle_id)
                    # print(f"Autonomous Vehicle {vehicle_id} Lane changed to {next_lane}.\n")
                    change_lane_due_to_block +=1
                    interaction_details.append("Autonomous Vehicle Lane changed to the appropriate lane.\n")
                    # print(f"change_lane_due_to_block Incremented.\n")
                    # print(f"******************************************************************************************************\n\n\n")


    return collision_follower_avoided, fitness_penalty_follow, collision_leader_avoided, fitness_penalty_lead, collision_neighbor_avoided, fitness_penalty_neighbor, change_lane_due_to_block, fitness_penalty_collision, interaction_details



# Send Interaction details for calcuation of PSO Logic
def calculate_interaction_scores():
    collision_follower_avoided = {}
    collision_leader_avoided = {}
    collision_neighbor_avoided = {}
    change_lane_due_to_block = {}
    fitness_penalty_follow= {}
    fitness_penalty_lead= {}
    fitness_penalty_neighbor= {}
    fitness_penalty_collision = {}
    interaction_details = {}

    all_vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in all_vehicle_ids:
        (
            follower_avoided,
            penalty_follow,
            leader_avoided,
            penalty_lead,
            neighbor_avoided,
            penalty_neighbor,
            lane_change_due_to_block,
            penalty_collision,
            details,
        ) = calculate_individual_interaction_score(vehicle_id)

        collision_follower_avoided[vehicle_id] = follower_avoided
        fitness_penalty_follow[vehicle_id] = penalty_follow
        collision_leader_avoided[vehicle_id] = leader_avoided
        fitness_penalty_lead[vehicle_id] = penalty_lead
        collision_neighbor_avoided[vehicle_id] = neighbor_avoided
        fitness_penalty_neighbor[vehicle_id] = penalty_neighbor
        change_lane_due_to_block[vehicle_id] = lane_change_due_to_block
        fitness_penalty_collision[vehicle_id] = penalty_collision
        interaction_details[vehicle_id] = details

    return (
        collision_follower_avoided,
        fitness_penalty_follow,
        collision_leader_avoided,
        fitness_penalty_lead,
        collision_neighbor_avoided,
        fitness_penalty_neighbor,
        change_lane_due_to_block,
        fitness_penalty_collision,
        interaction_details,
    )


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





# def get_fitness(vehicle_ids):
#     # Retrieve interaction scores and travel time for the swarm of vehicles
#     collision_follower_avoided, fitness_penalty_follow, collision_leader_avoided, fitness_penalty_lead, collision_neighbor_avoided, fitness_penalty_neighbor, change_lane_due_to_block, fitness_penalty_collision, interaction_details = calculate_interaction_scores()

#     # Use the first vehicle in the swarm as the reference for travel time
#     reference_vehicle_id = vehicle_ids[0]

#     # Pass the swarm IDs to the travel_times function
#     travel_time = travel_times().get(reference_vehicle_id, 0.001)

#     # Ensure travel_time is not 0 to avoid division by zero
#     if travel_time == 0:
#         return float('inf')

#     # Extract values from dictionaries
#     collision_follower_avoided_val = collision_follower_avoided.get(reference_vehicle_id, 0)
#     fitness_penalty_follow_val = fitness_penalty_follow.get(reference_vehicle_id, 0)
#     collision_leader_avoided_val = collision_leader_avoided.get(reference_vehicle_id, 0)
#     fitness_penalty_lead_val = fitness_penalty_lead.get(reference_vehicle_id, 0)
#     collision_neighbor_avoided_val = collision_neighbor_avoided.get(reference_vehicle_id, 0)
#     fitness_penalty_neighbor_val = fitness_penalty_neighbor.get(reference_vehicle_id, 0)
#     change_lane_due_to_block_val = change_lane_due_to_block.get(reference_vehicle_id, 0)
#     fitness_penalty_collision_val = fitness_penalty_collision.get(reference_vehicle_id, 0)

#     # Calculate the fitness value using the formula
#     fitness = 1 / travel_time + (
#         (collision_follower_avoided_val - fitness_penalty_follow_val) +
#         (collision_leader_avoided_val - fitness_penalty_lead_val) +
#         (collision_neighbor_avoided_val - fitness_penalty_neighbor_val) +
#         (change_lane_due_to_block_val - fitness_penalty_collision_val)
#     )

#     return fitness


def get_fitness(vehicle_id):
    # Retrieve interaction scores and travel time for the swarm of vehicles
    collision_follower_avoided, fitness_penalty_follow, collision_leader_avoided, fitness_penalty_lead, collision_neighbor_avoided, fitness_penalty_neighbor, change_lane_due_to_block, fitness_penalty_collision, interaction_details = calculate_interaction_scores()

    # Use the vehicle_id as the reference for travel time
    reference_vehicle_id = vehicle_id

    # Pass the swarm IDs to the travel_times function
    travel_time = travel_times().get(reference_vehicle_id, 0.001)

    # Ensure travel_time is not 0 to avoid division by zero
    if travel_time == 0:
        return float('inf')

    # Extract values from dictionaries
    collision_follower_avoided_val = collision_follower_avoided.get(reference_vehicle_id, 0)
    fitness_penalty_follow_val = fitness_penalty_follow.get(reference_vehicle_id, 0)
    collision_leader_avoided_val = collision_leader_avoided.get(reference_vehicle_id, 0)
    fitness_penalty_lead_val = fitness_penalty_lead.get(reference_vehicle_id, 0)
    collision_neighbor_avoided_val = collision_neighbor_avoided.get(reference_vehicle_id, 0)
    fitness_penalty_neighbor_val = fitness_penalty_neighbor.get(reference_vehicle_id, 0)
    change_lane_due_to_block_val = change_lane_due_to_block.get(reference_vehicle_id, 0)
    fitness_penalty_collision_val = fitness_penalty_collision.get(reference_vehicle_id, 0)

    # Calculate the fitness value using the formula
    fitness = 1 / travel_time + (
        (collision_follower_avoided_val - fitness_penalty_follow_val) +
        (collision_leader_avoided_val - fitness_penalty_lead_val) +
        (collision_neighbor_avoided_val - fitness_penalty_neighbor_val) +
        (change_lane_due_to_block_val - fitness_penalty_collision_val)
    )


    return fitness



def update_position(particle):
    particle[0] = max(0, min(25, particle[0]))  # Ensure speed is within [0, 25]
    particle[1] = max(0.1, min(5, particle[1]))  # Ensure minGap is within [0.1, 5]
    particle[2] = max(5, min(25, particle[2]))  # Ensure slowDown is within [5, 25]


def update_velocity(particle_velocity, personal_best_position, global_best_position, velocity_memory):
    # Update the particle's velocity based on inertia, personal best, and global best
    inertia_term = INERTIA * np.array(velocity_memory)
    personal_best_term = LOCAL_WEIGHT * np.random.random() * (np.array(personal_best_position) - np.array(particle_velocity))
    global_best_term = GLOBAL_WEIGHT * np.random.random() * (np.array(global_best_position) - np.array(particle_velocity))
    new_velocity = inertia_term + personal_best_term + global_best_term
    return new_velocity.tolist()


def update_vehicle_attributes(vehicle_id, new_position):
    # Update the vehicle's attributes in the simulation according to new_position
    try:
        if vehicle_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(vehicle_id, new_position[0])  # Example: updating speed
            # Add other attribute updates here based on new_position
    except Exception as e:  # Replace with specific simulation exceptions
        print(f"Error updating attributes for {vehicle_id}: {e}")


def evaluate_particle(vehicle_id):
    # Evaluate the fitness of the vehicle with its current attributes in the simulation
    fitness = get_fitness([vehicle_id])  # Implement this function based on your fitness evaluation criteria
    return fitness


def initialize_positions_and_velocities(vehicle_ids):
    positions = {}
    velocities = {}
    personal_bests = {}

    for vehicle_id in vehicle_ids:
        x, y = random.uniform(0, 100), random.uniform(0, 100)
        velocity = random.uniform(0, 10)

        positions[vehicle_id] = (x, y)
        velocities[vehicle_id] = velocity
        personal_bests[vehicle_id] = (x, y)

    return positions, velocities, personal_bests



def pso_algorithm():

    vehicle_ids = [f"pso_{i}" for i in range(SWARM_SIZE)]
    positions, velocities, personal_bests = initialize_positions_and_velocities(vehicle_ids)
    global_best_position = min(positions, key=lambda k: get_fitness(k))
    global_best_fitness = get_fitness(global_best_position)
    w = INERTIA
    c1 = LOCAL_WEIGHT
    c2 = GLOBAL_WEIGHT

    fitness_values = {}

    for vehicle_id in vehicle_ids:
        # Update velocity
        velocities[vehicle_id] = w * velocities[vehicle_id] + \
                                c1 * random.random() * (personal_bests[vehicle_id][0] - positions[vehicle_id][0]) + \
                                c2 * random.random() * (positions[global_best_position][0] - positions[vehicle_id][0])

        # Update position
        x = positions[vehicle_id][0] + velocities[vehicle_id]
        y = positions[vehicle_id][1]

        # Evaluate fitness in the new position
        fitness = get_fitness(vehicle_id)

        # Update personal best if the new position is better
        if fitness < get_fitness(vehicle_id):  # Pass the vehicle_id, not the positions dictionary
            personal_bests[vehicle_id] = (x, y)

        # Update global best if the new position is better
        if fitness < global_best_fitness:
            global_best_position = vehicle_id
            global_best_fitness = fitness

        # Update velocity memory
        positions[vehicle_id] = (x, y)

        # Store fitness value
        fitness_values[vehicle_id] = fitness

    # Return relevant information (modify as needed)
    return positions, velocities, personal_bests, global_best_position, global_best_fitness, fitness_values




def run_pso_experiment():
    vehicle_ids = [f"pso_{i}" for i in range(SWARM_SIZE)]

    # Get the list of edges in the simulation
    edges = traci.edge.getIDList()
    # Randomly select an edge from the list
    sumo_edge_id = random.choice(edges)



    # Initialize an empty list to store data
    result_data = []

    # Main simulation loop
    for iteration in range(ITERATIONS):
        traci.simulationStep()
        display_entered_and_left_vehicles(iteration)
        positions, velocities, personal_bests, global_best_position, global_best_fitness, fitness_values = pso_algorithm()

        collision_follower_avoided, fitness_penalty_follow, collision_leader_avoided, fitness_penalty_lead, collision_neighbor_avoided, fitness_penalty_neighbor, change_lane_due_to_block, fitness_penalty_collision, interaction_details = calculate_interaction_scores()


        # Calculate distances, speeds, and travel times
        distances = distance()
        speeds = speed()
        travel_times_data = travel_times()

        # Print or process the values
        print("\n Iteration:", iteration)
        print("\n Positions:", positions)
        print(" \nVelocities:", velocities)
        print("\n Personal Bests:", personal_bests)
        print("\n Global Best Position:", global_best_position)
        print("\n Fitness Values:", fitness_values)
        print("\n Global Best Fitness:", global_best_fitness)


        # Call calculate_interaction_scores for all vehicles
        collision_follower_avoided, fitness_penalty_follow, collision_leader_avoided, fitness_penalty_lead, collision_neighbor_avoided, fitness_penalty_neighbor, change_lane_due_to_block, fitness_penalty_collision, interaction_details = calculate_interaction_scores()

        # Append data to results
        append_data_to_results(
            iteration,
            collision_follower_avoided,
            fitness_penalty_follow,
            collision_leader_avoided,
            fitness_penalty_lead,
            collision_neighbor_avoided,
            fitness_penalty_neighbor,
            change_lane_due_to_block,
            fitness_penalty_collision,
            interaction_details,
            result_data,
            distances,
            speeds,
            travel_times_data
        )
        # Print or process the interaction details if needed


    # Create the final DataFrame
    global df
    df = pd.DataFrame(result_data)

    # Save the DataFrame to a CSV file
    df.to_csv('pso_results.csv', index=False)

    traci.close()
    sys.stdout.flush()




def append_data_to_results(iteration, collision_follower_avoided, fitness_penalty_follow, collision_leader_avoided, fitness_penalty_lead, collision_neighbor_avoided, fitness_penalty_neighbor,
                            change_lane_due_to_block, fitness_penalty_collision, interaction_details, result_data, distances, speeds, travel_times_data):
    global df  # Declare df as a global variable

    normal_vehicles = [vehicle_id for vehicle_id in traci.vehicle.getIDList() if vehicle_id.startswith("n_")]
    autonomous_vehicles = [vehicle_id for vehicle_id in traci.vehicle.getIDList() if vehicle_id.startswith("pso_")]

    normal_vehicles_count = len(normal_vehicles)
    autonomous_vehicles_count = len(autonomous_vehicles)

    print(f"Iteration {iteration} - Normal Vehicles Count: {normal_vehicles_count}, Autonomous Vehicles Count: {autonomous_vehicles_count}")

    # Choose the minimum length among normal and autonomous vehicles
    min_length = min(normal_vehicles_count, autonomous_vehicles_count)

    # Inside append_data_to_results function
    iteration_data = {
        'Iteration': iteration,
        'Normal Vehicle': normal_vehicles[:min_length],
        'Autonomous Vehicles (Particles)': autonomous_vehicles[:min_length],
        'Vehicle Type': ['Normal'] * min_length + ['Autonomous'] * min_length,
        'Travel Time (Seconds)': {f'{vehicle_id}': travel_times_data.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Distance (Km)': {f'{vehicle_id}': distances.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Speed (m/s)': {f'{vehicle_id}': speeds.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Collision with Follower avoided': {f'{vehicle_id}': collision_follower_avoided.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Collision with Leader avoided': {f'{vehicle_id}': collision_leader_avoided.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Collision with Neighbor avoided': {f'{vehicle_id}': collision_neighbor_avoided.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Lane Change Due to Block': {f'{vehicle_id}': change_lane_due_to_block.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Follower Vehicle Close Promximity Penalty Invoked': {f'{vehicle_id}': fitness_penalty_follow.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Leading Vehicle Close Promximity Penalty Invoked': {f'{vehicle_id}': fitness_penalty_lead.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Neighboring Vehicle Close Promximity Penalty Invoked': {f'{vehicle_id}': fitness_penalty_neighbor.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Collision Penalty Invoked': {f'{vehicle_id}': fitness_penalty_collision.get(vehicle_id, 0) for vehicle_id in autonomous_vehicles},
        'Interaction Details': {f'{vehicle_id}': interaction_details.get(vehicle_id, []) for vehicle_id in autonomous_vehicles}

    }

    # Append the iteration data lists to the result_data list
    result_data.append(iteration_data)










if __name__ == "__main__":
        # Read route information from the demo_network.rou.xml file
    route_info = read_routes_from_xml("C:/Sumo/Thesis2/Maps/SwarmOptimizationinPSO4/demo_network.rou.xml")

    # Generate PSO-controlled vehicles in the route file
    pso_vehicles = generate_pso_vehicles(route_info)
    print(pso_vehicles)

    # Modify the network.rou.xml file with the PSO-controlled vehicle type and route
    modified_routes = """
    <routes>
        <vType id="auto" 
                vClass="passenger" length="5" carFollowModel="Krauss" color="255,204,0" accel="2.6" decel="4.5" maxSpeed="30" begin="0" end="1500" safety="none" 
                sigma="1.0"/>

          #  <route id="route_0" edges="e1 e2 e3 e4 e5 m1 m2 e3 e4 e5 e6 e7 m4 m3 m2 e3 e4 e5 e6 e7 m4 m5 e2 e3 e4 e5 e6 e7 e8"/>
        #  <route id="route_1" edges="e1 e2 e3 e4 e5 m1 m3 e2 e3 e4 e5 m1 m3 e2 e3 e4 e5 e6 e7 e8 e1 e2 e3 e4 e5 m1 m3 e2 e3 e4 e5 m1 m3 e2 e3 e4 e5 e6 e7 e8"/>

        %s

    </routes>
    """ % pso_vehicles

    with open("C:/Sumo/Thesis2/Maps/SwarmOptimizationinPSO4/new_network.rou.xml", "w") as file:
        file.write(modified_routes)

    # Initialize simulation
    initialize_simulation()

    # Set initial parameters
    set_initial_parameters(route_info)
    run_pso_experiment()



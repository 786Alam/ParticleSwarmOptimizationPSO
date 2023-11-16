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
columns = ['Step', 'Normal Vehicles', 'Autonomous Vehicles(Particles)', 'Vehicle Type', 'Fitness', 'Travel Time', 'Interaction Score', 'Interaction Information', 'Global Best Fitness', 'Global Best Position']
df = pd.DataFrame(columns=columns)




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

for vehicle_element in all_vehicle_elements:
    vehicle_type = vehicle_element.get("type")
    if vehicle_type == 'auto':
        particle = {
            'accel': random.uniform(0.1, 5.0),
            'decel': random.uniform(0.1, 5.0),
            'maxSpeed': random.uniform(10, 30)
        }
        particles.append(particle)

        velocity = {
            'accel': random.uniform(-1.0, 1.0),
            'decel': random.uniform(-1.0, 1.0),
            'maxSpeed': random.uniform(-5, 5)
        }
        particle_velocities.append(velocity)

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
    interaction_score = 0
    MIN_FOLLOW_DISTANCE = 100
    MIN_LEADER_DISTANCE = 100
    MIN_LANE_CHANGE_DISTANCE = 100
    MAX_SPEED_DIFF = 5

    if vehicle_id not in traci.vehicle.getIDList():
        return interaction_score

    x1, y1 = traci.vehicle.getPosition(vehicle_id)
    vehicle_type = traci.vehicle.getTypeID(vehicle_id).split('@')[0]


    # Check for too close following normal vehicle
    follower_id, _ = traci.vehicle.getFollower(vehicle_id)
    if follower_id and follower_id in traci.vehicle.getIDList():
        x2, y2 = traci.vehicle.getPosition(follower_id)
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        follower_type = traci.vehicle.getTypeID(follower_id).split('@')[0]

        if vehicle_type == 'auto' and follower_type == 'car':

            # Get the Distance of the autonomous vehicle
            vehicle_distance = traci.vehicle.getDistance(vehicle_id)
            follower_distance = traci.vehicle.getDistance(follower_id)

            # Get the distance difference of the autonomous vehicle
            distance_difference = vehicle_distance - follower_distance
            distance_difference = abs(distance_difference)
            print(f"Distance Difference between Autonomous vehicle{vehicle_id} and follower Normal vehicle{follower_id}: {distance_difference} m.\n") 

            # Get the speed of the autonomous vehicle
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            follower_speed = traci.vehicle.getSpeed(follower_id)

            # Get the distance difference of the autonomous vehicle
            speed_difference = vehicle_speed - follower_speed
            speed_difference = abs(speed_difference)
            print(f"Speed Difference between Autonomous vehicle{vehicle_id} and follower Normal vehicle{follower_id}: {speed_difference} km/h.\n")

            if vehicle_type == 'auto' and follower_type == 'car':
                if distance_difference <= MIN_FOLLOW_DISTANCE:
                    print(f"Autonomous Vehicle {vehicle_id} is too close to Following Normal Vehicle({follower_id}),\n distance: {distance_difference} m.\n") 
                    # If autonomous vehicle speed is greater than or equal to 5, decrease speed by 1
                    if vehicle_speed <= 2:
                        print(f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 2} km/h.\n") 
                        # Calculate new distance
                        new_distance = distance_difference + 10
                        print( f"Autonomous Vehicle{vehicle_id} increased distance from Following Normal Vehicle, new distance from Following Normal Vehicle: {new_distance} m.\n")
                    else:
                        print(f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n") 
                        # new_distance = traci.vehicle.getDistance(vehicle_id)
                        print(f"Autonomous Vehicle{vehicle_id} maintaining distance from Following Normal Vehicle distance: {distance_difference} m. \n \n \n") 
                else:
                    pass
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

            if vehicle_type == 'auto' and leader_type == 'auto':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                leader_distance = traci.vehicle.getDistance(leader_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - leader_distance
                distance_difference = abs(distance_difference)
                print(f"Distance Difference between Autonomous vehicle{vehicle_id} and Leading Autonomous vehicle{leader_id}: {distance_difference} m.\n")

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - leader_speed
                speed_difference = abs(speed_difference)
                print(f"Speed Difference between Autonomous vehicle {vehicle_id} and Leader Autonomous Vehicle{leader_id}: {speed_difference} km/h.\n") 


                if vehicle_type == 'auto' and leader_type == 'car':
                    if distance_difference <= MIN_LEADER_DISTANCE:
                        print(f"Autonomous Vehicle {vehicle_id} is too close to the Leading Vehicle({leader_id}),\n distance: {distance_difference} m.\n") 
                        # If autonomous vehicle speed is greater than or equal to 18, decrease speed to 18
                        if vehicle_speed >= 2:
                            print(f"Autonomous Vehicle {vehicle_id} decreased speed to {vehicle_speed - 1} km/h.\n") 
                            # Calculate new distance
                            new_leader_distance = distance_difference + 10
                            print(f"Autonomous Vehicle {vehicle_id} increased distance from Leading Autonomous Vehicle{leader_id}, new distance: {new_leader_distance} m.\n") 
                        else:
                            print(f"Autonomous Vehicle {vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n") 
                            # new_leader_distance = traci.vehicle.getDistance(vehicle_id)
                            print(f"Autonomous Vehicle {vehicle_id} maintaining distance From Leading Autonomous Vehicle{leader_id}: {distance_difference} km/h.\n \n \n") 
                    else:
                        pass
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
                            print(f"Autonomous Vehicle {vehicle_id} is significantly faster than the Leading Autonomous Vehicle({leader_id}),\n speed difference: {speed_difference} km/h.\n") 
                            # traci.vehicle.setSpeed(vehicle_id, 5.0)  # Set speed to increase distance
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

            if vehicle_type == 'auto' and neighbor_type == 'car':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                neighbor_distance = traci.vehicle.getDistance(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - neighbor_distance
                distance_difference = abs(distance_difference)
                print(f"Distance Difference between Autonomous vehicle{vehicle_id} and Neighboring Normal vehicle{neighbor_id}: {distance_difference} m.\n") 

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                neighbor_speed = traci.vehicle.getSpeed(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - neighbor_speed
                speed_difference = abs(speed_difference)
                print(f"Speed Difference between Autonomous vehicle{vehicle_id} and Neighboring Normal vehicle{neighbor_id}: {speed_difference} km/h.\n") 

                if vehicle_type == 'auto' and neighbor_type == 'car':
                    if distance_difference < MIN_LANE_CHANGE_DISTANCE:
                        print(f"Autonomous Vehicle{vehicle_id} is too close to a Neighbor Normal Vehicle({neighbor_id}) in a different lane,\n distance: {distance_difference} m.\n") 

                        # If autonomous vehicle speed is greater than or equal to 25, decrease speed by 1
                        if vehicle_speed >= 2:
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 1)
                            print(f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 1} km/h.\n") 
                            # Calculate new distance
                            new_distance = speed_difference + 10
                            print(f"Autonomous Vehicle{vehicle_id} increased distance from Neighbor Normal Vehicle in a different lane, new distance: {new_distance} m.\n \n \n") 
                        else:
                            # If autonomous vehicle speed is less than 25, print actual speed and maintain distance
                            print(f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} m/s, maintaining distance.\n") 
                            # new_distance = traci.vehicle.getDistance(vehicle_id)
                            print(f"Autonomous Vehicle{vehicle_id} maintaining distance from Neighbor Normal vehicle, Distance: {distance_difference} m.\n \n \n") 

                        # Check if the autonomous vehicle is blocked and change lane
                        if not traci.vehicle.getNextTLS(vehicle_id) and not traci.vehicle.getLaneChangeState(vehicle_id, 0):  # 0 corresponds to 'left'
                            # Get the best lane information
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

    blockers = traci.vehicle.getNeighbors(vehicle_id, 1)
    for blocker_id, _ in blockers:
        if blocker_id != vehicle_id and blocker_id in traci.vehicle.getIDList():
            x2, y2 = traci.vehicle.getPosition(blocker_id)
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            blocker_type = traci.vehicle.getTypeID(blocker_id).split('@')[0]
            if vehicle_type == 'auto' and blocker_type == 'car':
                interaction_score += 1
                print(f"Autonomous Vehicle{vehicle_id} is blocked by Normal Vehicle{blocker_id},\n distance: {distance} m.\n")


    return interaction_score




def calculate_interaction_scores():
    interaction_scores = {}
    all_vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in all_vehicle_ids:
        interaction_score = calculate_individual_interaction_score(vehicle_id, all_vehicle_ids)
        interaction_scores[vehicle_id] = interaction_score

    return interaction_scores



def travel_times():
    vehicle_travel_times = {}
    for veh_id in traci.vehicle.getIDList():
        time = traci.simulation.getTime()
        depart_time = traci.vehicle.getDeparture(veh_id)
        if depart_time >= 0:
            travel_time = time - depart_time
            vehicle_travel_times[veh_id] = travel_time

    return vehicle_travel_times

def fitness_function(step, particle, particle_best_fitness, particle_best_position, global_best_fitness, global_best_position, vehicle_travel_times, interaction_scores):
    fitness_values = {}

    for veh_id in traci.vehicle.getIDList():
        try:
            vehicle_type = traci.vehicle.getTypeID(veh_id)
            if vehicle_type == 'car':
                accel_min = 0.1
                accel_max = 5.0
                decel_min = 0.1
                decel_max = 5.0
                max_speed_min = 1.0
                max_speed_max = 30.0
            elif vehicle_type == 'auto':
                accel_min = 0.1
                accel_max = 5.0
                decel_min = 0.1
                decel_max = 5.0
                max_speed_min = 1.0
                max_speed_max = 60.0
            else:
                accel_min = 0.1
                accel_max = 5.0
                decel_min = 0.1
                decel_max = 5.0
                max_speed_min = 1.0
                max_speed_max = 30.0

            accel_clamped = min(max(particle['accel'], accel_min), accel_max)
            decel_clamped = min(max(particle['decel'], decel_min), decel_max)
            max_speed_clamped = min(max(particle['maxSpeed'], max_speed_min), max_speed_max)

            traci.vehicle.setAccel(veh_id, accel_clamped)
            traci.vehicle.setDecel(veh_id, decel_clamped)
            traci.vehicle.setMaxSpeed(veh_id, max_speed_clamped)

            traci.simulationStep()

            vehicle_interaction_score = interaction_scores.get(veh_id, 0.0)
            vehicle_travel_time = vehicle_travel_times.get(veh_id, 0.0)
            fitness_value = w1 * (1 / vehicle_travel_time) + w2 * vehicle_interaction_score

            interaction_info = get_interaction_info(veh_id)

            fitness_values[veh_id] = fitness_value

            pso_results.append([
                step,
                veh_id,
                fitness_value,  # Corrected variable name
                vehicle_travel_times[veh_id],
                interaction_scores[veh_id],
                interaction_info,
                None,  # Global Best Fitness
                None   # Global Best Position
            ])

            df = df.append({
                'Step': step,
                'Normal Vehicles': veh_id if vehicle_type == 'car' else None,
                'Autonomous Vehicles(Particles)': veh_id if vehicle_type == 'auto' else None,
                'Fitness': fitness_value,  # Corrected variable name
                'Travel Time': vehicle_travel_times.get(veh_id, None),
                'Interaction Score': interaction_scores.get(veh_id, None),
                'Interaction Information': interaction_info,
                'Global Best Fitness': None,
                'Global Best Position': None
            }, ignore_index=True)

            if veh_id not in particle_best_fitness[i] or fitness_value < particle_best_fitness[i][veh_id]:  # Corrected variable name
                particle_best_fitness[i][veh_id] = fitness_value  # Corrected variable name
                particle_best_position[i] = particle.copy()

            if fitness_value < global_best_fitness:  # Corrected variable name
                global_best_fitness = fitness_value  # Corrected variable name
                global_best_position = particle.copy()

        except Exception as e:
            continue

    return fitness_values


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

        if vehicle_type == 'auto' and follower_type == 'car':

            # Get the Distance of the autonomous vehicle
            vehicle_distance = traci.vehicle.getDistance(vehicle_id)
            follower_distance = traci.vehicle.getDistance(follower_id)

            # Get the distance difference of the autonomous vehicle
            distance_difference = vehicle_distance - follower_distance
            distance_difference = abs(distance_difference)
            info += f"Distance Difference between vehicle{vehicle_id} and follower Normal vehicle{follower_id}: {distance_difference} m.\n"

            # Get the speed of the autonomous vehicle
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            follower_speed = traci.vehicle.getSpeed(follower_id)

            # Get the distance difference of the autonomous vehicle
            speed_difference = vehicle_speed - follower_speed
            speed_difference = abs(speed_difference)
            info += f"Speed Difference between vehicle{vehicle_id} and follower Normal Vehicle{follower_id}: {speed_difference} km/h.\n"

            if vehicle_type == 'auto' and follower_type == 'car':
                if distance_difference <= MIN_FOLLOW_DISTANCE:
                    info += f"Autonomous Vehicle{vehicle_id} is too close to Following Normal Vehicle({follower_id}),\n distance: {distance_difference} m.\n"
                    # If autonomous vehicle speed is greater than or equal to 5, decrease speed by 1
                    if vehicle_speed <= 2:
                        info += f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 2} km/h.\n"
                        # Calculate new distance
                        new_distance = distance_difference + 10
                        info += f"Autonomous Vehicle{vehicle_id} increased distance from Following Normal Vehicle, new distance from Following Normal Vehicle{follower_id}: {new_distance} m.\n"
                    else:
                        info += f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n"
                        # new_distance = traci.vehicle.getDistance(vehicle_id)
                        info += f"Autonomous Vehicle{vehicle_id} maintaining distance from Following Normal Vehicle distance: {distance_difference} m.\n \n \n"
                else:
                    pass
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

            if vehicle_type == 'auto' and leader_type == 'auto':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                leader_distance = traci.vehicle.getDistance(leader_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - leader_distance
                distance_difference = abs(distance_difference)
                info += f"Distance Difference between vehicle{vehicle_id} and Leading Autonomous vehicle{leader_id}:{distance_difference} m.\n"

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                leader_speed = traci.vehicle.getSpeed(leader_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - leader_speed
                speed_difference = abs(speed_difference)
                info += f"Speed Difference between vehicle{vehicle_id} and Leading Autonomous Vehicle{leader_id}:{speed_difference} km/h.\n"


                if vehicle_type == 'auto' and leader_type == 'car':
                    if distance_difference <= MIN_LEADER_DISTANCE:
                        info += f"Autonomous Vehicle{vehicle_id} is too close to the Leading Autonomous Vehicle({leader_id}),\n distance: {leader_distance} m.\n"
                        # If autonomous vehicle speed is greater than or equal to 18, decrease speed to 18
                        if vehicle_speed >= 2:
                            info += f"Autonomous Vehicle{vehicle_id} decreased speed to {vehicle_speed - 1} km/h.\n"
                            # Calculate new distance
                            new_leader_distance = distance_difference + 10
                            info += f"Autonomous Vehicle{vehicle_id} increased distance from Leading Autonomous Vehicle, new distance: {new_leader_distance} m.\n"
                        else:
                            info += f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} m/s, maintaining distance.\n"
                            # new_leader_distance = traci.vehicle.getDistance(vehicle_id)
                            info += f"Autonomous Vehicle{vehicle_id} maintaining distance From Leading Autonomous Vehicle: {leader_distance} m.\n \n \n"
                    else:
                        pass
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
                            info += f"Autonomous Vehicle{vehicle_id} is significantly faster than the Leading Autonomous Vehicle({leader_id}),\n speed difference: {speed_difference} km/h.\n"
                            # traci.vehicle.setSpeed(vehicle_id, 5.0)  # Set speed to increase distance
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

            if vehicle_type == 'auto' and neighbor_type == 'car':

                # Get the Distance of the autonomous vehicle
                vehicle_distance = traci.vehicle.getDistance(vehicle_id)
                neighbor_distance = traci.vehicle.getDistance(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                distance_difference = vehicle_distance - neighbor_distance
                distance_difference = abs(distance_difference)
                info += f"Distance Difference between vehicle{vehicle_id} and Neighboring Normal vehicle{neighbor_id}: {distance_difference} m.\n"

                # Get the speed of the autonomous vehicle
                vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                neighbor_speed = traci.vehicle.getSpeed(neighbor_id)

                # Get the distance difference of the autonomous vehicle
                speed_difference = vehicle_speed - neighbor_speed
                speed_difference = abs(speed_difference)
                info += f"Speed Difference between vehicle{vehicle_id} and Neighboring Normal vehicle{neighbor_id}: {speed_difference} km/h.\n"

                if vehicle_type == 'auto' and neighbor_type == 'car':
                    if speed_difference < MIN_LANE_CHANGE_DISTANCE:
                        info += f"Autonomous Vehicle{vehicle_id} is too close to a Neighbor Normal Vehicle({neighbor_id}) in a different lane,\n distance: {distance_difference} m.\n"

                        # If autonomous vehicle speed is greater than or equal to 25, decrease speed by 1
                        if vehicle_speed >= 2:
                            traci.vehicle.setSpeed(vehicle_id, vehicle_speed + 1)
                            info += f"Autonomous Vehicle{vehicle_id} increased speed to {vehicle_speed + 1} km/h.\n"
                            # Calculate new distance
                            new_distance = speed_difference + 10
                            info += f"Autonomous Vehicle{vehicle_id} increased distance from Neighbor Normal Vehicle in a different lane, new distance: {new_distance} m.\n \n \n"
                        else:
                            # If autonomous vehicle speed is less than 25, print actual speed and maintain distance
                            info += f"Autonomous Vehicle{vehicle_id} actual speed is {vehicle_speed} km/h, maintaining distance.\n"
                            # new_distance = traci.vehicle.getDistance(vehicle_id)
                            info += f"Autonomous Vehicle{vehicle_id} maintaining distance from Neighbor Normal vehicle, Distance: {distance_difference} m.\n \n \n"

                        # Check if the autonomous vehicle is blocked and change lane
                        if not traci.vehicle.getNextTLS(vehicle_id) and not traci.vehicle.getLaneChangeState(vehicle_id, 0):  # 0 corresponds to 'left'
                            # Get the best lane information
                            best_lanes = traci.vehicle.getBestLanes(vehicle_id)
                            if best_lanes:
                                best_lane_id = best_lanes[0][0]  # Choose the first best lane
                                best_lane_direction = best_lanes[0][2]  # Direction information
                                current_lane_id = traci.vehicle.getLaneID(vehicle_id)
                                if best_lane_id != current_lane_id:
                                    info += f"Autonomous Vehicle {vehicle_id} is blocked. Changing lane to {best_lane_id}.\n"
                                    traci.vehicle.changeLane(vehicle_id, best_lane_id, 10.0)  # Change to the best lane
                                    traci.simulationStep()  # Allow the simulation to process the lane change
                                    info += f"{vehicle_id} changed lane to {best_lane_id} in direction {best_lane_direction} to increase distance from Neighbor Normal Vehicle in a different lane\n \n \n"

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

            vehicle_ids = traci.vehicle.getIDList()
            vehicle_travel_times = travel_times()
            interaction_scores = calculate_interaction_scores()

            fitness_values = fitness_function(step, particle, particle_best_fitness[i], particle_best_position[i], global_best_fitness, global_best_position, vehicle_travel_times, interaction_scores)

            # Update particle and global best fitness and position
            for veh_id, fitness in fitness_values.items():
                print("Step:", step)

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
                        print("\nInteraction Information:\n", interaction_info, "\n")

                        # Append only if interaction_info is not empty
                        particle_results = [
                            step,
                            None if vehicle_type != 'car' else veh_id,
                            veh_id,
                            fitness if vehicle_type == 'auto' else "",  # Fill in only for 'auto' type
                            vehicle_travel_times.get(veh_id, None),
                            interaction_scores.get(veh_id, None),
                            interaction_info,
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
                        'Interaction Score': interaction_scores.get(veh_id, None),
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
    run(num_steps=200, xml_file=xml_file)








import utils
import environment
import argparse
import os
import numpy as np
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--row', type = int)
parser.add_argument('-c', '--col', type = int)
parser.add_argument('--update_type', type = str, choices=['partial', 'all'], default = 'partial')
parser.add_argument('--update_epoch', type = int, default = 25)
parser.add_argument('--comm', default=False, action = "store_true")
parser.add_argument('--mem', default=False, action = "store_true")
args = parser.parse_args()

def main():
    # test env
    duration = 500
    lanes = ['F', 'L', 'R']
    directions = ['N', 'E', 'S', 'W']
    A = [(['E', 'W'], ['F']), (['E', 'W'], ['L']), 
        (['N', 'S'], ['F']), (['N', 'S'], ['L']), 
        (['E'], ['F', 'L']), (['W'], ['F', 'L']), 
        (['N'], ['F', 'L']), (['S'], ['F', 'L'])]
    vehicle_parameters = {'duration': duration, 'min_speed': 2.22, 'max_speed': 13.33, 
                        'lanes': lanes, 'arrival_rates': {'E': 5/60, 'N': 5/60, 'W': 10/60, 'S': 7/60}}
    intersection_parameters = {'duration': duration, 'action_duration': 10, 
                            'Lanes': lanes, 'Directions': directions, 'A': A, 
                            'gamma': 0.95, 'alpha': 0.1, 'espilon': 0.1, 'is_mem_based':False,  
                            'is_dynamic_action_duration': False, 'dynamic_action_duration': 4, 
                            'reward_function': utils.calculate_reward_default, 'n_vehicle_leaving_per_lane': 1}
    graph_structure_params = {'rows': args.row, 'cols': args.col, 'length': 60}
    communcation_parameters = {'A': A, 'mu': 0.0, 'sigma': 1.0, 'tau': 0.0}

    folder_path = f'metrics/run_{args.row}_{args.col}'
    if os.path.isdir(folder_path) == False:
        os.mkdir(folder_path)

    tot_waiting_time = []
    node_waiting_time = {}
    all_clearance = []
    node_clearance = {}
    all_arriving = []

    for seed in np.arange(0, 30, 1):
        utils.seed_all(seed = int(seed))
        env_SARSA = environment.Env(duration=duration, update_type= args.update_type, comm_based = args.comm, graph_structure_parameters=graph_structure_params, 
                                    vehicle_parameters=vehicle_parameters, intersection_parameters=intersection_parameters,
                                    communication_parameters=communcation_parameters)

        env_SARSA.SARSA_run(n_episodes=1)

        test_graph, test_vehicles = env_SARSA.generate_test_structures(graph_structure_parameters=graph_structure_params, 
                                                                vehicle_parameters=vehicle_parameters, 
                                                                intersection_parameters=intersection_parameters)
        test_graph.set_memory_based(is_mem_based=args.mem)
        env_SARSA.test(test_graph=test_graph, test_vehicles=test_vehicles, update_epoch = args.update_epoch)
        waiting_time_mean, waiting_time_node, departing_metrics, agg_departing_metrics, arriving_vehicles = env_SARSA.get_congestion_metric()

        tot_waiting_time.append(waiting_time_mean)
        all_clearance.append(agg_departing_metrics)
        all_arriving.append(arriving_vehicles)


        if len(node_waiting_time) == 0:
            node_waiting_time = {key: [] for key in waiting_time_node.keys()}
        if len(node_clearance) == 0:
            node_clearance = {key: [] for key in departing_metrics.keys()}
        
        for key in node_waiting_time.keys():
            node_waiting_time[key].append(waiting_time_node[key])

        for key in node_clearance.keys():
            node_clearance[key].append(departing_metrics[key])
    
    np.save(f'{folder_path}/total_waiting_time_{args.row}_{args.col}_mem_{args.mem}_comm_{args.comm}', tot_waiting_time)
    np.save(f'{folder_path}/total_clearance_{args.row}_{args.col}_mem_{args.mem}_comm_{args.comm}', all_clearance)
    np.save(f'{folder_path}/total_arriving_{args.row}_{args.col}_mem_{args.mem}_comm_{args.comm}', all_arriving)

    with open(f"{folder_path}/node_waiting_time_{args.row}_{args.col}_mem_{args.mem}_comm_{args.comm}.pickle", 'wb') as f:
        pickle.dump(node_waiting_time, f)

    with open(f"{folder_path}/node_clearance_{args.row}_{args.col}_mem_{args.mem}_comm_{args.comm}.pickle", 'wb') as f:
        pickle.dump(node_clearance, f)    

if __name__ == "__main__":
    main()
import utils
import environment
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seed')
parser.add_argument('-r', '--row', type = int)
parser.add_argument('-c', '--col', type = int)
parser.add_argument('--comm', default=False, action = "store_true")
args = parser.parse_args()

utils.seed_all(seed = int(args.seed))

def main():
    # test env
    duration = 100
    lanes = ['F', 'L', 'R']
    directions = ['N', 'E', 'S', 'W']
    A = [(['E', 'W'], ['F']), (['E', 'W'], ['L']), 
        (['N', 'S'], ['F']), (['N', 'S'], ['L']), 
        (['E'], ['F', 'L']), (['W'], ['F', 'L']), 
        (['N'], ['F', 'L']), (['S'], ['F', 'L'])]
    vehicle_parameters = {'duration': duration, 'min_speed': 2.22, 'max_speed': 13.33, 
                        'lanes': lanes, 'arrival_rates': {'E': 25, 'N': 25, 'W': 22, 'S': 27}}
    intersection_parameters = {'duration': duration, 'action_duration': 5, 
                            'Lanes': lanes, 'Directions': directions, 'A': A, 
                            'gamma': 0.95, 'alpha': 0.1, 'espilon': 0.1, 'is_mem_based':False,
                            'is_dynamic_action_duration': False, 'dynamic_action_duration': 4, 
                            'reward_function': utils.calculate_reward_default, 'n_vehicle_leaving_per_lane': 1, 'verbose': False}
    graph_structure_params = {'rows': args.row, 'cols': args.col, 'length': 60}
    communcation_parameters = {'A': A, 'mu': 0.0, 'sigma': 1.0, 'tau': 0.0, 'agg_func': np.max}

    env_SARSA = environment.Env(duration=duration, comm_based = args.comm, graph_structure_parameters=graph_structure_params, 
            vehicle_parameters=vehicle_parameters, intersection_parameters=intersection_parameters,
            communication_parameters=communcation_parameters, verbose = False)

    env_SARSA.SARSA_run(n_episodes=1, update_epoch = 1)
    w = env_SARSA.display_congestion_metric()

    if os.path.isfile(f'waiting_time_{args.row}_{args.col}_{args.comm}.npy') == False:
        np.save(f'waiting_time_{args.row}_{args.col}_{args.comm}', np.array(w))
    else:
        res = np.load(f'waiting_time_{args.row}_{args.col}_{args.comm}.npy')
        res = np.append(res, np.array(w))
        np.save(f'waiting_time_{args.row}_{args.col}_{args.comm}', res)

if __name__ == "__main__":
    main()
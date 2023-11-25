from typing import Dict, Any, Tuple, List
import random
import numpy as np

# the default reward function
def calculate_reward_default(current_state, next_state, action, debug=False):
    """
        Only Considers the Direction/Lane on which the action was taken
        and find the # of vehicles displaced.
    """
    difference = next_state - current_state
    if(debug):
        print(action)
        print(difference)
    reward = 0
    for d in action[0]:
        for i in action[1]:
            if(difference.state[d][i]<=0):
                reward+= -difference.state[d][i] + 1
            elif(difference.state[d][i]>=6):
                reward-= difference.state[d][i]
    return reward

def calculate_reward_diffusion(current_state, next_state, action, debug=False):
  """Calculates the diffusion coefficients of the particles (vehicles).
  Returns:
    The diffusion coefficient. 
    >0 represents array2 is a better state than array 1
  """
  # Calculate the mean signed squared displacement of the particles
  d = current_state - next_state

  diffusion_coefficient = np.mean(np.sign(d.to_numpy())*(d.to_numpy())**2)
  return diffusion_coefficient


def calculate_reward_entropy(current_state, next_state, action, debug=False):
  """Calculates the entropy difference between two arrays.

  Returns:
    The entropy difference between the two arrays.
    > 0 values represents that array2 is a better state than array 1
  """

  # add 1 to both states to make sure log(0) doesnt happen
  array1 = np.add(current_state.to_numpy(),1)
  array2 = np.add(next_state.to_numpy(),1)

  weighted_entropy1 = -np.sum(array1 * np.log(array1))

  weighted_entropy2 = -np.sum(array2 * np.log(array2))

  entropy_difference = weighted_entropy2 - weighted_entropy1
  return entropy_difference


def generate_graph(rows: int, cols: int, length=None) -> Dict[Any, Tuple[int, List[Tuple[Any, int, str]]]]:
    directions = ['N', 'E', 'S', 'W']
    opposite_d = {'N': 'S', 'E':'W', 'S':'N', 'W':'E'}
    graph_structure = {}

    def get_random_length():
        return random.randint(1, 10) if length is None else length
    
    fill_k = 0
    for i in range(rows):
        for j in range(cols):
            node_name = str(i * cols + j)
            neighbors = []
            if node_name in graph_structure.keys():
                neighbors = neighbors
            for direction in directions:
                neighbor = None
                if direction == 'N' and i > 0:
                    neighbor = (str((i - 1) * cols + j), get_random_length(), direction)
                elif direction == 'E' and j < cols - 1:
                    neighbor = (str(i * cols + (j + 1)), get_random_length(), direction)
                elif direction == 'S' and i < rows - 1:
                    neighbor = (str((i + 1) * cols + j), get_random_length(), direction)
                elif direction == 'W' and j > 0:
                    neighbor = (str(i * cols + (j - 1)), get_random_length(), direction)
                else:
                    # Add filler nodes for edges without neighbors
                    neighbor = ('in' + str(fill_k), 0, direction)
                    graph_structure[neighbor[0]] = (1, [(node_name, neighbor[1], opposite_d[direction])])
                    fill_k+=1
                neighbors.append(neighbor)
                

            graph_structure[node_name] = (len(neighbors), neighbors)
    
    return graph_structure


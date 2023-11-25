from graph import Graph
import environment
import memory
import copy
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import deque
from PIL import Image

class X_state:
    def __init__(self, Lanes: List[str] = ['F', 'L', 'R'], Directions: List[str] = ['E', 'N', 'W', 'S']) -> None:
        self.state = {}
        for direction in Directions:
            self.state[direction] = {}
            for lane in Lanes:
                self.state[direction][lane] = 0
    
    def __eq__(self, other):
        if isinstance(other, X_state):
            return self.state == other.state
        return False
    
    def self_sum(self):
        result = 0
        for direction in self.state:
            for lane in self.state[direction]:
                # Calculate the sum of values within each direction and square the result
                result += self.state[direction][lane]
        return result

    def distance_metric(self, other):
        if isinstance(other, X_state):
            difference = self - other
            return difference.sum_and_square()
        
        else:
            raise ValueError("Distance metric is only defined for x_state objects.")

    def sum_and_square(self):
        result = 0
        for direction in self.state:
            for lane in self.state[direction]:
                # Calculate the sum of values within each direction and square the result
                result += self.state[direction][lane]

        return result**2
    
    def __sub__(self, other):
        if isinstance(other, X_state):
            result_state = {}
            for direction in self.state:
                result_state[direction] = {}
                for lane in self.state[direction]:
                    result_state[direction][lane] = self.state[direction][lane] - other.state[direction][lane]
            new_x_state = X_state()
            new_x_state.state = result_state
            return new_x_state
        else:
            raise ValueError("Subtraction is only defined for x_state objects.")
        
    def to_numpy(self):
        return np.array(self.__get_values_tuple())
    
    @classmethod
    def numpy_to_x_state(cls, np_state):
        state = X_state()
        i = 0
        for direction in state.state:
            for lane in state.state[direction]:
                # Calculate the sum of values within each direction and square the result
                state.state[direction][lane] = np_state[i]
                i+=1
        return state
        
    def __get_values_tuple(self):
        # Use a nested list comprehension to extract values from inner dictionaries
        values_list = [value for inner_dict in self.state.values() for value in inner_dict.values()]
        # Convert the list of values into a tuple
        values_tuple = tuple(values_list)
        return values_tuple

    def __hash__(self):
        # To make the object hashable, convert the nested dictionaries to a frozenset
        return hash(self.__get_values_tuple())
    
    def __str__(self):
        result = ""
        for direction in self.state:
            for lane in self.state[direction]:
                result += f"{direction}{lane}: {self.state[direction][lane]}\n"
        return result

class Intersection:
    def __init__(self, name: str, reward_function, duration: int=200, action_duration: int=10, gamma=0.95, alpha=0.1, espilon=0.1,
                 Lanes: List[str] = ['F', 'L', 'R'], is_mem_based: bool = False, 
                 is_dynamic_action_duration: bool = False, dynamic_action_duration: int = 4,
                 Directions: List[str] = ['E', 'N', 'W', 'S'], 
                 A = [(['E', 'W'], ['F']), (['E', 'W'], ['L']), 
                        (['N', 'S'], ['F']), (['N', 'S'], ['L']), 
                        (['E'], ['F', 'L']), (['W'], ['F', 'L']), 
                        (['N'], ['F', 'L']), (['S'], ['F', 'L'])],
                n_vehicle_leaving_per_lane=1) -> None:
        
        # to do implement dynamic action duration
        self.name = name
        self.Lanes = Lanes
        self.Directions = Directions
        self.duration = duration
        self.isdynamic_action_duration = is_dynamic_action_duration
        self.dynamic_action_duration = dynamic_action_duration
        self.action_duration = action_duration
        self.states = [X_state() for _ in range(self.duration)]
        self.i = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = espilon
        self.n_vehicle_leaving_per_lane = n_vehicle_leaving_per_lane
        self.calculate_reward = reward_function
        self.departing_metrics = []
       
        # list of vehicles for this intersection
        self.vehicles: Dict[Tuple[str, str], List[Any]] = {}
        for direction in Directions:
            for lane in Lanes:
                self.vehicles[(direction, lane)] = []
        self.actions = A
        self.action_strings = [self.format_action(action) for action in self.actions]
        self.q_table = self.get_empty_q_table()
        self.is_mem_based = is_mem_based
        if is_mem_based:
            self.mem = memory.Memory(q_table=self.q_table, gamma=gamma, alpha=alpha, duration=duration)

        self.current_action = self.get_next_action(next_state=self.get_current_state(), first=True)

    def reset(self):
        # reset all states and vehicles
        self.i=0        
        self.states = [X_state() for _ in range(self.duration)]
        self.vehicles: Dict[Tuple[str, str], List[Any]] = {}
        self.current_action = self.get_next_action(next_state=self.get_current_state(), first=True)
        for direction in self.Directions:
            for lane in self.Lanes:
                self.vehicles[(direction, lane)] = []
        if self.is_mem_based:
            self.mem = memory.Memory(q_table=self.q_table, gamma=self.gamma, alpha=self.alpha, duration=self.duration)

    def get_empty_q_table(self):
        q_table = pd.DataFrame(columns=self.action_strings, dtype=np.float64)
        return q_table

    def insert_vehicle(self, vehicle, lane, direction):
        self.vehicles[(direction, lane)].append(vehicle)

    def remove_vehicle(self, vehicle, lane, direction):
        self.vehicles[(direction, lane)].remove(vehicle)

    def update_q_table(self, reward, state_next, action_next, done):
        state = self.get_current_state()
        action = self.current_action

        action_string = self.format_action(action)
        action_next_string = self.format_action(action_next)

        if self.i==0:
            self.q_table_check_if_state_exist(state)
        self.q_table_check_if_state_exist(state_next)

        q_value_predict = self.q_table.loc[state, action_string]
        if not done:
            q_value_real = reward + self.gamma * self.q_table.loc[state_next, action_next_string]
        else:
            q_value_real = reward
        self.q_table.loc[state, action_string] += self.alpha * (q_value_real - q_value_predict)
    
    def safe_right_turn(self, direction):
        right_safety_lane_check = {'E': [('S', 'F'), ('W', 'L')], 
                                   'W': [('N', 'F'), ('E', 'L')], 
                                   'N': [('E', 'F'), ('S', 'L')], 
                                   'S': [('W', 'F'), ('N', 'L')]}
        for d, l in right_safety_lane_check[direction]:
            if self.get_current_state().state[d][l] > 0:
                return False
        return True


    def apply_state(self, state, action):
        self.i+=1
        self.current_action = action
        self.states[self.i] = state

    def q_table_check_if_state_exist(self, state):
        if state not in self.q_table.index:
            new_row = pd.Series([0] * len(self.actions), 
                                index=self.q_table.columns, 
                                name=state)
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])

    def move_vehicle(self, next_state, current_action, i):
        total_depart = 0
        for direction in self.Directions:
            for lane in self.Lanes:
                tmp = {}
                depart_count = 0

                safe_right_turn = self.safe_right_turn(direction) if lane == 'R' else False              
                for vehicle in self.vehicles[(direction, lane)]:
                    # print(vehicle.node_times)
                    departed = vehicle.step(current_time = i, next_state=next_state,
                                             action=current_action, safe_right_turn=safe_right_turn)
                    if departed:
                        depart_count += 1
                        total_depart += 1
                    if depart_count == self.n_vehicle_leaving_per_lane:
                        break

        self.departing_metrics.append(total_depart)
                
    def step(self, debug=False):
        next_state = X_state()
        i = self.i
        action = self.current_action
        ## carry over the previous number of cars at the junction
        current_state = self.get_current_state()
        next_state = copy.deepcopy(current_state)


        self.move_vehicle(next_state=next_state, current_action=action, i=i)
        
        done = False
        reward = self.calculate_reward(current_state=self.get_current_state(), next_state=next_state,action=action,debug=debug)
        if(self.i+1 >= self.duration):
            done = True
        return next_state, reward, done
    
    def has_traffic_changed(self, last_states):
        # check if the traffic has changed much in the current action duration
        # for direction, lane in self.current_action:
        for direction in self.current_action[0]:
            for lane in self.current_action[1]:
                for i in range(len(last_states)-1):
                    # if the traffic has changed much then return true
                    if abs(last_states[i].state[direction][lane] - last_states[i+1].state[direction][lane]) > 1:
                        return True
        return False
    
    def get_next_action(self, next_state, first=False):        
        # i=0 and is the very first action
        if self.i==0 and first:
            target_action = self.actions[np.random.choice(len(self.actions))]
        # only change the action if more than actiton duration time has passed
        elif self.isdynamic_action_duration:
            # if it then we need to change the action duration when the traffic hasnt changed much
            # in the current action duration, lane
            target_action = self.current_action
            if self.i%self.action_duration>self.dynamic_action_duration:                
                # check if the traffic has changed much in the current action duration
                # if it hasnt then we get a new action  

                # get the last dynamic action duration states 
                # and check if the traffic has changed much
                last_states = self.states[self.i-self.dynamic_action_duration:self.i]            
                if not self.has_traffic_changed(last_states):                  
                    self.q_table_check_if_state_exist(next_state)
                    if np.random.rand() < self.epsilon:
                        nearest_state, target_action = self.find_nearest_state_in_q(next_state)
                    else:
                        target_action = self.actions[np.random.choice(len(self.actions))]
        elif self.i % self.action_duration==0 and self.i>0:
            self.q_table_check_if_state_exist(next_state)
            if np.random.rand() < self.epsilon:
                nearest_state, target_action = self.find_nearest_state_in_q(next_state)
            else:
                target_action = self.actions[np.random.choice(len(self.actions))]
        else:
            target_action = self.current_action
        return target_action
    

    def get_current_state(self):
        return self.states[self.i]
    
    
    def find_actions_for_max_traffic_displacement(self,state: X_state):
        max_action = self.actions[0]
        max_v = -1

        for directions,lanes in self.actions:
            curr_sum = 0
            for direction in directions:
                for lane in lanes:
                    curr_sum+=state.state[direction][lane]
            if curr_sum >max_v:
                max_v = curr_sum
                max_action = (directions,lanes)
        return max_action

    def greedy(self, next_state, first=False):
        if self.is_mem_based:
            return self.greedy_mem(next_state, first)
        else:
            return self.greedy_not_mem(next_state, first)
        
    def greedy_mem(self, next_state, first=False):
        '''
        Greedy policy

        return the index corresponding to the maximum action-state value
        '''

        if (self.i % self.action_duration == 0 and self.i>0) or first:
            in_memory, action = self.mem.get_action_from_memory(next_state)
            
            # not in memory, so the best greedy action would be to turn on the signal 
            # for lines which has current maximum traffic
            if not in_memory:
                action = self.find_actions_for_max_traffic_displacement(next_state)
                
        else:
            action = self.current_action

        return action

    def greedy_not_mem(self, next_state, first=False):
        '''
        Greedy policy

        return the index corresponding to the maximum action-state value
        '''

        if (self.i % self.action_duration == 0 and self.i>0) or first:
        
            if next_state in self.q_table.index:
                action = self.q_table.loc[next_state].idxmax()
                action =  self.parse_formatted_action(action)
            else:
                # return the nearest state action
                nearest_state, action = self.find_nearest_state_in_q(next_state)
        else:
            action = self.current_action
        return action

    
    def find_nearest_state_in_q(self, state2, tolerence=1):
        # to do 
        # Calculate distances to all states in the Q-table
        data = np.vstack(self.q_table.index.map(X_state.to_numpy).values)
        kd = KDTree(data)
        # Find the index of the closest state
        indexes = kd.query_ball_point(state2.to_numpy(), r=tolerence)
        nearby_points = data[indexes]
        nearby_states = [X_state.numpy_to_x_state(nearby_point) for nearby_point in nearby_points]
        nearest_state, action = self.q_table.loc[nearby_states].stack().idxmax()
        action = self.parse_formatted_action(action)
        return nearest_state, action
    
    # ------------------ Utility functions ------------------    
    # used to convert actions to strings
    def format_action(self, action):
        formatted_action = f"(({','.join(action[0])}),({','.join(action[1])}))"
        return formatted_action
        
    
    # convert the formatted action string back to action
    def parse_formatted_action(self, formatted_action):
        try:
            # Remove the parentheses and split the string into two parts
            parts = formatted_action.strip("()").split("),(")
            
            if len(parts) != 2:
                raise ValueError("Invalid formatted action string")

            # Split the two parts into lists
            part1 = parts[0].split(',')
            part2 = parts[1].split(',')

            # Return the parsed action
            return (part1, part2)
        except ValueError as e:
            print(f"Error parsing formatted action: {e}")
            return None


    def plot_all(self):
        # delete the individual images
        # for file in os.listdir('plots/individual/'):
        #     os.remove(f'plots/individual/{file}')
        
        image_path=[]

        image_path.append(self.plot_q_table())
        if self.is_mem_based:
            image_path.append(self.plot_memory())
        image_path.append(self.plot_states())     
        image_path.append(self.plot_traffic())


        output_path = f'plots/combined/Intersection{self.name}.png'
        self.combine_images(image_path, output_path)
      

    def plot_states(self):
        def generate_unique_line_styles(n):
            line_styles = []
            for i in range(n):
                style = (i + 1, i + 1)  # Custom dash pattern: alternates lengths based on index
                line_styles.append(style)
            return line_styles
        # Create separate plots for each direction
        # Create a separate subplot for each direction
        # calculate rows and columns based on the number of directions
        cols = int(np.ceil(len(self.Directions) / 2))
        rows = int(np.ceil(len(self.Directions) / cols))
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
        ax = ax.flatten()

        directions = self.Directions
        custom_line_styles = generate_unique_line_styles(len(self.Lanes))

        for i, direction in enumerate(directions):
            for j, lane in enumerate(self.Lanes):
                # Extract 'F' and 'L' values for the given direction from x_state_t.states
                values = [state.state[direction][lane] for state in self.states]
                
                # Plot 'F' values
                ax[i].plot(range(len(values)), values, label=f"{direction}-{lane}", linestyle='dashed', dashes=custom_line_styles[j])
            
            ax[i].set_title(f'Direction {direction}')
            ax[i].set_xlabel('Time (t)')
            ax[i].set_ylabel('Total Number of Vehicles (YD(t))')
            ax[i].grid(True)
            ax[i].legend()


        plt.title(f"Traffic for {self.name}")
        # Adjust subplot layout
        plt.tight_layout()
        # Show all the plots
        # plt.show()

        # Save the plot in a directory
        directory_path = 'plots/individual/'  
        file_name = f'Intersection{self.name}.png'  
        # Combine the directory path and file name
        file_path = directory_path + file_name
        # Save the plot
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return file_path

    def plot_q_table(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(self.q_table, cmap='viridis')
        ax.set_xlabel('Actions')
        ax.set_ylabel('States')
        ax.set_xticks(range(len(self.q_table.columns)))
        ax.set_xticklabels(self.q_table.columns, rotation=90)
        ax.set_yticks(range(len(self.q_table.index)))
        # ax.set_yticklabels(self.q_table.index)
        ax.set_title(f"Q-table for {self.name}")

        # Create a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Q-value')  # Set the label for the colorbar

        # Save the plot in a directory
        directory_path = 'plots/individual/'  
        file_name = f'Qtable{self.name}.png'  
        # Combine the directory path and file name
        file_path = directory_path + file_name
        # Save the plot
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return file_path

    
    def plot_memory(self):
        if self.is_mem_based:            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(range(self.mem.size_time.shape[0]), self.mem.size_time)
            ax.set_xlabel('Time (t)')
            ax.set_ylabel('Memory Size')
            ax.set_title(f"Memory size for {self.name}")
            # plt.show()

             # Save the plot in a directory
            directory_path = 'plots/individual/'  
            file_name = f'Memory{self.name}.png'  
            # Combine the directory path and file name
            file_path = directory_path + file_name
            # Save the plot
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            return file_path
        return None

    def plot_traffic(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(range(self.duration), [state.self_sum() for state in self.states])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Total Number of Vehicles (YD(t))')
        ax.set_title(f"Traffic for {self.name}")
        # plt.show()
        # Save the plot in a directory
        directory_path = 'plots/individual/'  
        file_name = f'Traffic{self.name}.png'  
        # Combine the directory path and file name
        file_path = directory_path + file_name
        # Save the plot
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return file_path


    def combine_images2(self, image_paths, output_path):
        images = [Image.open(path) for path in image_paths]
        fig, axes = plt.subplots(1, len(images), figsize=(12, 4))  # Adjust figsize as needed
        plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Adjust spacing between subplots

        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            ax.set_aspect('auto')  # Adjust aspect ratio to prevent stretching

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save with tight bounding box
        plt.close(fig)  # Close the figure

    
    def combine_images(self, image_paths, output_path):
        images = [Image.open(path) for path in image_paths]

        # Get dimensions of the images
        widths, heights = zip(*(i.size for i in images))

        # Calculate the combined image width and height
        total_width = sum(widths)
        max_height = max(heights)

         # Create a new blank image with a white background and the calculated dimensions
        combined_image = Image.new('RGBA', (total_width, max_height), color=(255, 255, 255, 0))  
        # Paste the images onto the combined image
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + 10

        # Save the combined image
        combined_image.save(output_path)
        print(f"Images combined and saved as '{output_path}'")
        # Display the combined image
        plt.imshow(combined_image)
        plt.axis('off')  # Hide axis
        plt.title(f'Intersection {self.name}')
        plt.show()

   
    def __str__(self) -> str:
        return f"Intersection {self.name}:\n{self.x_state}"

class Vehicle:
    def __init__(self, path, arrival_time: float, graph: Graph, min_speed: float=30, max_speed: float=50):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.path = path
        self.graph = graph
        # create a df to store arrival and departure times for nodes in the path with following structure {'node': {'in_d': arrival_direction, 'in_l': arrival lane, 'out_d': departure_direction }}}
        self.node_times = pd.DataFrame(columns=['node', 'arrival_direction', 'arrival_lane', 'arrival_time', 'departure_time'])
        # intialise the arrival time for the first node in the path
        # initialise the nodes_times with node from nodes and arrival lane and direction from path and set times to np.nan
        for node in path:
            node_deets = pd.Series({'node': node,
                                    'arrival_direction': path[node]['in_d'],
                                    'arrival_lane': path[node]['in_l'],
                                    'arrival_time': np.nan,
                                    'departure_time': np.nan
                                    })

            self.node_times = pd.concat([self.node_times, node_deets.to_frame().T], ignore_index=True)


        self.node_times['arrival_time'][0] = arrival_time
        self.current_node_index = 0
        self.curent_node = self.node_times.loc[self.current_node_index, 'node']
        self.arrival_lane = self.node_times.loc[self.current_node_index, 'arrival_lane']
        self.arrival_direction = self.node_times.loc[self.current_node_index, 'arrival_direction']
        self.graph.nodes[self.curent_node].intersection.insert_vehicle(vehicle=self, 
                                                                       lane=self.arrival_lane, 
                                                                       direction=self.arrival_direction)
        self.arrived_at_node = False
        self.reached_destination = False

        

    def get_current_speed(self):
        return np.random.uniform(self.min_speed, self.max_speed)

    def step(self, current_time, next_state: X_state, action: Tuple[List[str], List[str]], safe_right_turn=False):
        # move the vehicle to the next node in the path and update the departure time for the previous node
        # update the arrival time for the current node
        # set the in_buffer flag to False
        # if the vehicle has reached the end of the path, set the departure time for the current node
        # and set the in_buffer flag to False
        # if the vehicle has not reached the end of the path, set the in_buffer flag to True
        # and update the arrival time for the next node in the path

        # arrival 
        if current_time >= self.node_times.loc[self.current_node_index, 'arrival_time'] and not self.arrived_at_node:
            current_node = self.curent_node
            current_node_index = self.current_node_index
            direction = self.node_times.loc[current_node_index, 'arrival_direction']
            lane = self.node_times.loc[current_node_index, 'arrival_lane']
            self.arrived_at_node = True
            next_state.state[direction][lane] += 1
        
        departed = False
        # departure
        if self.arrived_at_node:
            # vehicle is at this node
            # check if the vehicle can depart and then depart
            if (self.arrival_lane == 'R' and safe_right_turn) \
                 or (self.arrival_direction in action[0] and self.arrival_lane in action[1]):
               departed = self.depart_vehicle_from_current_node(current_time)
          
        return departed
    
    def depart_vehicle_from_current_node(self, current_time):
        # get currnt and next node
        departed = False
        current_node = self.curent_node
        current_node_index = self.current_node_index
        
        # check if the current node is the last node in the path
        if current_node_index == len(self.node_times) - 1:
            # reached destination
            self.reached_destination = True
            self.node_times.loc[current_node_index, 'departure_time'] = current_time
            self.graph.nodes[current_node].intersection.remove_vehicle(vehicle=self, 
                                                                        lane=self.arrival_lane, 
                                                                        direction=self.arrival_direction)
            departed = True
        else:

            next_node_index = current_node_index + 1
            next_node = self.node_times.loc[next_node_index, 'node']

            # get the current speed and distance to next node
            current_speed = self.get_current_speed()
            distance = self.graph.nodes[current_node].edges[self.node_times.loc[next_node_index, 'arrival_direction']].length
            travel_time = distance / current_speed

            # update the departure time for the current node and arrival time for the next node
            self.node_times.loc[current_node_index, 'departure_time'] = current_time
            self.node_times.loc[next_node_index, 'arrival_time'] = current_time + travel_time
        
            # remove vehicle form the current node
            self.graph.nodes[current_node].intersection.remove_vehicle(vehicle=self, 
                                                                        lane=self.arrival_lane, 
                                                                        direction=self.arrival_direction)

            # update the current node and current node index
            self.curent_node = next_node
            self.current_node_index = next_node_index                         
            self.arrival_lane = self.node_times.loc[self.current_node_index, 'arrival_lane']
            self.arrival_direction = self.node_times.loc[self.current_node_index, 'arrival_direction']
            self.graph.nodes[self.curent_node].intersection.insert_vehicle(vehicle=self, 
                                                        lane=self.arrival_lane, 
                                                        direction=self.arrival_direction)
            departed = True
            self.arrived_at_node = False  

        return departed  



class Vehicles:
    def __init__(self, graph: Graph, duration,
                  min_speed=20, max_speed=50, lanes = ['F', 'L', 'R'],
                  arrival_rates={'E': 5/60, 'N': 5/60, 'W': 10/60, 'S': 7/60}):
        
        self.duration = duration
        self.arrival_rates = arrival_rates
        self.graph = graph
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.initialise_vehicle()
        self.num_vehicles = len(self.vehicles)

    def get_two_end_nodes(self):
        # get two random nodes from the input nodes, make sure these two are different
        # return as a tuple
        input_nodes = self.graph.input_nodes.copy()
        nodes = np.random.choice(input_nodes, 2, replace=False)
        return tuple(nodes)
    
    def get_random_end_node(self, input_node):
        # get a random node from the input nodes
        # return as a tuple
        input_nodes = self.graph.input_nodes.copy()
        input_nodes.remove(input_node)
        node = np.random.choice(input_nodes, 1)[0]
        return node
    
    # def step(self, current_time: int, current_action: Tuple[List[str], List[str]]):
    #     ## to do 

    #     for vehicle in self.vehicles:
    #         vehicle.step()

    def get_arrival_times(self):        
        nodes = self.graph.non_input_nodes
        arrival_times = []
        arrival_times_per_node = {node: [] for node in nodes} 

        for vehicle in self.vehicles:
            for node in vehicle.node_times['node'].values.tolist():
                arrival_times_per_node[node].append(vehicle.node_times['arrival_time'].values)
                arrival_times.append(vehicle.node_times['arrival_time'].values)

        for node in nodes:
            arrival_times_per_node[node] = np.concatenate(arrival_times_per_node[node]).ravel()

        arrival_times = np.concatenate(arrival_times).ravel()
        return arrival_times_per_node, arrival_times
    
    def get_departure_times(self):
        nodes = self.graph.non_input_nodes

        departure_times = []
        departure_times_per_node = {node: [] for node in nodes} 

        for vehicle in self.vehicles:
            for node in vehicle.node_times['node'].values.tolist():
                departure_times_per_node[node].append(vehicle.node_times['departure_time'].values)
                departure_times.append(vehicle.node_times['departure_time'].values)

        for node in nodes:
            departure_times_per_node[node] = np.concatenate(departure_times_per_node[node]).ravel()

        departure_times = np.concatenate(departure_times).ravel()
        return departure_times_per_node, departure_times
    
    def initialise_vehicle(self):
        self.vehicles = []
        for node in self.graph.input_nodes:
            node_dir = self.graph.opposite_d[list(self.graph.nodes[node].neighbors.keys())[0]]
            t = 0
            for i in range(self.duration):
                while t <= i:
                    inter_arrival_time = random.expovariate(self.arrival_rates[node_dir])
                    t += inter_arrival_time    
                    if t > self.duration:
                        break        
                    end = self.get_random_end_node(input_node=node)
                    path = self.generate_path(self.graph.graph_structure, start_node=node, end_node=end)
                    vehicle = Vehicle(graph = self.graph, min_speed=self.min_speed, max_speed=self.max_speed, 
                                                path=path, arrival_time=t)
                    self.vehicles.append(vehicle)


    def generate_path(self, graph, start_node, end_node):

        opposite_d = self.graph.opposite_d
        visited = set()
        queue = deque([(start_node, [])])

        def determine_turn(current_dir, next_dir):
            if current_dir == next_dir or next_dir == opposite_d[current_dir]:
                return 'F'
            elif current_dir == 'N' and next_dir == 'E':
                return 'L'
            elif current_dir == 'N' and next_dir == 'W':
                return 'R'
            elif current_dir == 'E' and next_dir == 'S':
                return 'L'
            elif current_dir == 'E' and next_dir == 'N':
                return 'R'
            elif current_dir == 'S' and next_dir == 'W':
                return 'L'
            elif current_dir == 'S' and next_dir == 'E':
                return 'R'
            elif current_dir == 'W' and next_dir == 'N':
                return 'L'
            elif current_dir == 'W' and next_dir == 'S':
                return 'R'

        while queue:
            current_node, path = queue.popleft()

            if current_node == end_node:
                # path = [start_node] + path
                # print(f"Path found: {' -> '.join(str(p) for p in path)}")
                formatted_path = {}
                for i in range(len(path) - 1):
                    # print(path[i])
                    node, direction = path[i]
                    next_node, next_direction = path[i + 1]
                
                    turn = determine_turn(direction, opposite_d[next_direction])
                    
                    formatted_path.update({next_node: {"in_d": direction, 
                                                    "in_l": turn, 
                                                    "out_d": opposite_d[next_direction]}})
                
                return formatted_path

            if current_node not in visited:
                visited.add(current_node)

                _, neighbors = graph[current_node]
                # print(current_node, neighbors)
                for neighbor, _, direction in neighbors:
                    new_path = list(path)
                    new_path.append( (current_node, opposite_d[direction]))
                    queue.append((neighbor, new_path))
        print("No path found.")
        return None


class Env:
    def __init__(self, duration, graph_structure_parameters, vehicle_parameters, intersection_parameters: Dict) -> None:
        self.directions = ['N', 'E', 'S', 'W']
        self.opposite_d = {'N': 'S', 'E':'W', 'S':'N', 'W':'E'}

        # generate graph
        self.graph_structure = self.generate_graph_structure(**graph_structure_parameters)
        self.graph = Graph(intersection_parameter_dic=intersection_parameters)
        self.graph.add_from_dict(graph_structure=self.graph_structure)
        self.graph.draw_graph_2()

        self.vehicles = Vehicles(graph = self.graph, **vehicle_parameters)
        self.vehicle_parameters = vehicle_parameters
        self.intresection_parameters = intersection_parameters
        self.duration = duration       
    
        self.departing_metrics_result = {}

    def generate_test_structures(self, graph_structure_parameters, vehicle_parameters, intersection_parameters: Dict):
         # generate graph
        self.test_graph_structure = self.generate_graph_structure(**graph_structure_parameters)
        self.test_graph = Graph(intersection_parameter_dic=intersection_parameters)
        self.test_graph.add_from_dict(graph_structure=self.graph_structure)
        self.test_graph.draw_graph_2()

        self.test_vehicles = Vehicles(graph=self.graph, **vehicle_parameters)
        return self.test_graph, self.test_vehicles



    def generate_graph_structure(self, rows: int, cols: int, length=None) -> Dict[Any, Tuple[int, List[Tuple[Any, int, str]]]]:
        directions = self.directions
        opposite_d = self.opposite_d
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
    
    def SARSA_run(self, n_episodes):
        for _ in range(n_episodes):
            # reset the environment
            self.graph.reset()
            self.vehicles = Vehicles(graph=self.graph, **self.vehicle_parameters)

            done_dict = self.done_dict_initialise()
            while not all(done_dict.values()):
                for node in self.graph.nodes:
                    if not node.startswith('in'):
                        if not done_dict[node]:
                            done = self.step_onestep(intersection=self.graph.nodes[node].intersection)
                            done_dict[node] = done

                        if done_dict[node]:
                            #if done, gather all departing vehicles count
                            self.departing_metrics_result[node] = self.graph.nodes[node].intersection.departing_metrics

    def done_dict_initialise(self):
        done_dict = {}
        for node in self.graph.nodes:
            if not node.startswith('in'):
                done_dict[node] = False
        return done_dict
    
    def step_onestep(self, intersection: environment.Intersection)-> bool:
        #move_vehicle function call
        next_state, reward, done = intersection.step()
        next_action = intersection.get_next_action(next_state=next_state)
        intersection.update_q_table(reward=reward, state_next=next_state, action_next=next_action, done=done)
        if not done:
            intersection.apply_state(state=next_state, action=next_action)
        return done

    def copy_q_table(self, test_graph, test_vehicles):
        graph = copy.deepcopy(test_graph)
        vehicles = copy.deepcopy(test_vehicles)
        for node in graph.nodes:
            if not node.startswith('in'):
                graph.nodes[node].intersection.q_table = copy.deepcopy(self.graph.nodes[node].intersection.q_table)
                graph.nodes[node].intersection.reset()
        
        return graph, vehicles

    def mem_test(self, n_test, test_graph, test_vehicles):
        # to do calculate average metrics
        
        graph, vehicles = self.copy_q_table(test_graph, test_vehicles)

        self.graph = graph
        self.vehicles = vehicles

        for _ in range(n_test):
            # reset the environment
            done_dict = self.done_dict_initialise()
            while not all(done_dict.values()):
                for node in test_graph.nodes:
                    if not node.startswith('in'):
                        if not done_dict[node]:
                            done = self.step_onestep(intersection=test_graph.nodes[node].intersection)
                            done_dict[node] = done

    def compute_waiting_time(self, departure, arrival):
        waiting_times = []
        for i in range(len(departure)):
            depart = departure[i]
            arrive = arrival[i]
            
            if depart is np.nan or arrive is np.nan:
                continue
            else:
                wait = depart-arrive
                waiting_times.append(wait)
        
        return np.array(waiting_times)

    def get_wait_time(self):
        departure_times_per_node, departure_times = self.vehicles.get_departure_times()
        arrival_times_per_node, arrival_times = self.vehicles.get_arrival_times()

        nodes = list(departure_times_per_node.keys())
        waiting_time_per_node = {}

        for node in nodes:
            tmp = self.compute_waiting_time(departure_times_per_node[node], arrival_times_per_node[node])
            waiting_time_per_node[node] = tmp

        waiting_time = self.compute_waiting_time(departure_times, arrival_times)
            
        return waiting_time_per_node, waiting_time

    def compute_departing_metrics(self):
        departing_metrics = self.departing_metrics_result
        cumsum_metrics = {}

        
        for node in departing_metrics.keys():
            cumsum_metrics[node] = np.cumsum(departing_metrics[node])

        agg_metrics = np.zeros(cumsum_metrics[node].shape)
        for key, values in cumsum_metrics.items():
            agg_metrics += values      
        
        return cumsum_metrics, agg_metrics

    def display_congestion_metric(self):
        # find W
        waiting_time_per_node, waiting_time = self.get_wait_time()
        avg_waiting_time_per_node = {node: np.mean(waiting_time_per_node[node]) for node in waiting_time_per_node.keys()}
        departing_metrics, agg_departing_metrics = self.compute_departing_metrics()

        if len(waiting_time) == 0:
            print(f"No cars have pass the intersection yet..")
        else:
            print(f"W: total average weight time per lane: {waiting_time.mean().mean()}s")
            print(f"average weight time per node in s (nan means no cars arrived):\n {avg_waiting_time_per_node}")

        
        plt.plot(range(len(agg_departing_metrics)), agg_departing_metrics)
        plt.xlabel("Time (t)")
        plt.ylabel('Destination reached by total #cars ($V_C(t)$)')
        plt.title("$V_C(t)$ for all intersections")
        plt.grid(True)
        # Show the plot
        plt.show()


        for node in departing_metrics.keys():
            plt.figure()
            plt.plot(range(len(departing_metrics[node])), departing_metrics[node])
            plt.xlabel("Time (t)")
            plt.ylabel('Destination reached by total #cars ($V_C(t)$)')
            plt.title(f"$V_C(t)$ for Intersection {node}")
            plt.grid(True)
            # Show the plot
            plt.show()




    def plot_env(self):
        self.graph.draw_graph_2()
        for node in self.graph.nodes:
            if not node.startswith('in'):
                self.graph.nodes[node].intersection.plot_all()


   


import pandas as pd
import numpy as np
import environment
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


class Memory():
    def __init__(self, q_table, gamma, alpha, duration) -> None:
         # columns: "distance metric-sum of elemnts ^2, optimal action, optimal R_a"
         self.columns =['d', 'a', 'R']
         self.memory_table = pd.DataFrame(columns=self.columns)
         self.append_using_q_table(q_table=q_table)
         self.size_time = np.zeros(duration)
         self.size_time[0] = self.memory_table.shape[0]
         self.short_term_Q = pd.DataFrame(columns=q_table.columns, dtype=np.float64)
         self.short_term_memory_size = 10
         self.gamma = gamma
         self.alpha = alpha
         self.duration=duration
    
    def update_size_time_array(self,i):
        self.size_time[i] = self.memory_table.shape[0]

    def append_using_q_table(self, q_table):
        for s in q_table.index:
            d = s.self_sum()
            a = q_table.columns[np.nanargmax(q_table.loc[s])]
            r = q_table.loc[s,a]
            self._insert(s,d,self.parse_formatted_action(a),r)

    def _insert(self, state, d, a, r):
        # find the eq state,action pair which has max rew
        best_state, best_action, in_memory = self.read_memory_table(memory_type='long', state=state)
        if not in_memory:
            new_row = pd.Series([d,a,r], 
                                index=self.columns, 
                                name=state)
            self.memory_table = pd.concat([self.memory_table, new_row.to_frame().T])
        else:
            # see if the reward is significantly better than update the memory
            if self.memory_table['R'][best_state]+0.005 < r:
                self.memory_table['a'][best_state] = a
                self.memory_table['R'][best_state] = r

    def insert(self,current_state, current_action, reward, state_next, action_next, done):
        # short term memory is full, time to append it to long term memory table
        if self.short_term_Q.shape[0] >= self.short_term_memory_size:
            # find the size//2 most frequent states in short term memory, i.e. ones with biggest rewards
            q = self.short_term_Q.nlargest(self.short_term_memory_size//2, self.short_term_Q.columns)
            self.append_using_q_table(q)
            #empty the short term memory buffer
            self.short_term_Q = pd.DataFrame(columns=self.short_term_Q.columns, dtype=np.float64)
        
        # short term memory is not empty, 
        # treat it as Q table in sarsa
        self.update_q_table(current_state, current_action, reward, state_next, action_next, done)



    def q_table_check_if_state_exist(self, state):
        nearby_state, action, in_memory = self.read_memory_table(memory_type='short', state=state)
        # no similar states, so a unique state found!
        if not in_memory:
            new_row = pd.Series([0] * len(self.short_term_Q.columns), 
                                index=self.short_term_Q.columns, 
                                name=state)
            self.short_term_Q = pd.concat([self.short_term_Q, new_row.to_frame().T])
            return state
        else:
            return nearby_state

    def update_q_table(self,current_state, current_action, reward, state_next, action_next, done):
        state = current_state
        action = current_action

        action_string = self.format_action(action)
        action_next_string = self.format_action(action_next)

        state = self.q_table_check_if_state_exist(state)
        state_next = self.q_table_check_if_state_exist(state_next)

        q_value_predict = self.short_term_Q.loc[state, action_string]
        if not done:
            q_value_real = reward + self.gamma * self.short_term_Q.loc[state_next, action_next_string]
        else:
            q_value_real = reward
        self.short_term_Q.loc[state, action_string] += self.alpha * (q_value_real - q_value_predict)
    
    def read_memory_table(self, memory_type, state, tolerence=1):
        if(memory_type=='long'):
            memory = self.memory_table  
        elif(memory_type=='short'):
            memory = self.short_term_Q

        if(memory.shape[0]==0):
            in_memory = False
            return None, None, in_memory

        if state in memory.index:
            in_memory = True
            # basically we know the state exists in the memory, so just use it as it is
            if memory_type =='long':
                nearby_state = state
                action = self.memory_table.loc[nearby_state]['a']
            else:
                nearby_state = state
                target_actions = self.short_term_Q.loc[state, :]
                target_actions = target_actions.reindex(np.random.permutation(target_actions.index))
                target_action = target_actions.idxmax()
                action = self.parse_formatted_action(target_action)
                
        else:
            # the state is not in memeory, find a nearby state and use it 
            
            data = np.vstack(memory.index.map(environment.X_state.to_numpy).values)
            kd = KDTree(data)
            indexes = kd.query_ball_point(state.to_numpy(), r=tolerence)
            in_memory = True

            # didnt find a nearyb state
            if(len(indexes)==0):
                in_memory = False
                return None, None, in_memory
            nearby_points = data[indexes]
            nearby_states = [environment.X_state.numpy_to_x_state(nearby_point) for nearby_point in nearby_points]
            if memory_type =='long':
                nearby_table =  self.memory_table.loc[nearby_states]
                nearby_state = nearby_table.index[nearby_table['R'].to_numpy().argmax()]
                action = nearby_table.loc[nearby_state]['a']
            else:
                nearby_state,action = self.short_term_Q.loc[nearby_states].stack().idxmax()
                action = self.parse_formatted_action(action)

        return nearby_state, action, in_memory

    def check_if_equiv_state_exist(self, memory_type, state, tolerence=1):
        if(memory_type=='long'):
            memory = self.memory_table  
        elif(memory_type=='short'):
            memory = self.short_term_Q

        if(memory.shape[0]==0):
            in_memory = False
            return None, None, in_memory
        
        all_equiv_states = self.get_equiv_states(state)
        matching_indices = self.memory_table.index.isin(list(all_equiv_states.keys()))
        # Get the states that match
        matching_states_list = self.memory_table.index[matching_indices].tolist()
        if len(matching_states_list) == 0:
            in_memory = False
            return None, None, in_memory
        in_memory = True
        if memory_type =='long':
            nearby_table =  self.memory_table.loc[matching_states_list]
            nearby_state = nearby_table.index[nearby_table['R'].to_numpy().argmax()]
            action = nearby_table.loc[nearby_state]['a']
        else:
            nearby_state,action = self.short_term_Q.loc[matching_states_list].stack().idxmax()
            action = self.parse_formatted_action(action)
        
        return nearby_state, action, in_memory



    def get_equiv_states(self, state: environment.X_state):
        state_arrays = self.generate_translated_arrays(state.to_numpy())
        all_equiv_classes = self.generate_shifted_arrays(state_arrays)
        return all_equiv_classes 

    def generate_shifted_arrays(input_list, shift_amount=2):
        # Initialize an empty dictionary to store shifted arrays and rotation amounts
        result_dict = {}

        # Iterate over each array in the input list
        for input_array in input_list:
            # Iterate over the shift positions for each array
            for i in range(0, len(input_array), shift_amount):
                # Create a copy of the input array
                shifted_array = np.roll(input_array, i)

                # Convert the shifted array to a tuple to make it hashable
                shifted_tuple = tuple(shifted_array)

                # Add the shifted array and rotation amount to the dictionary
                result_dict[environment.X_state.numpy_to_x_state(shifted_array)] = i

        return result_dict 


    def generate_shifted_arrays(input_list):
        # Initialize an empty set to store unique arrays
        unique_arrays = set()

        # Initialize an empty list to store the result arrays
        result_list = []

        # Iterate over each array in the input list
        for input_array in input_list:
            # Iterate over the shift positions for each array
            for i in range(0, len(input_array), 2):
                # Create a copy of the input array
                shifted_array = np.roll(input_array, i)

                # Convert the shifted array to a tuple to make it hashable
                shifted_tuple = tuple(shifted_array)

                # Append the shifted array to the result list if it's unique
                if shifted_tuple not in unique_arrays:
                    unique_arrays.add(shifted_tuple)
                    result_list.append()
        return result_list


    def generate_translated_arrays(input_array):
        # Initialize an empty list to store the result arrays
        result_list = []
        result_list.append(input_array)

        # Iterate over the indices of the input array
        for i in range(len(input_array)):
            # Create a copy of the input array
            new_array = np.copy(input_array)

            # Increase the value at the current index by 1
            new_array[i] += 1

            # Append the new array to the result list
            result_list.append(new_array)

        return result_list



    def plot_memory_tables(self, directory_path, name):
        # plot short term memory as heat map and display long term memory as a table
        fig, ax = plt.subplots(1,2, figsize=(20,10))
        ax[1].set_title('Short Term Memory')
        ax[0].set_title('Long Term Memory')
        im = ax[1].imshow(self.short_term_Q, cmap='viridis')
        ax[1].set_xlabel('Actions')
        ax[1].set_ylabel('States')
        ax[1].set_xticks(range(len(self.short_term_Q.columns)))
        ax[1].set_yticks(range(len(self.short_term_Q.index)))
        ax[1].set_xticklabels(self.short_term_Q.columns, rotation=90)
        ax[1].set_yticklabels(self.short_term_Q.index, rotation=0)
             # Create a colorbar
        cbar = ax[1].figure.colorbar(im, ax=ax[1])
        cbar.set_label('Q-value')  # Set the label for the colorbar
        
        # long term memory, display as a heatmap ingore distance metric, set columns to self.shortermQ.columns
        # as long term memory only has actions and rewards, set all other missing actions to 0
        # create a new df with columns as short term memory and index as long term memor
        # create an empty df as long term memory, fill all values to 0
        long_term_memory = pd.DataFrame(0, index=self.memory_table.index, columns=self.short_term_Q.columns)

        # update the long term memory with the values from long term memory table for an action 
        for state in self.memory_table.index:
            action = self.memory_table.loc[state]['a']
            action = self.format_action(action)
            reward = self.memory_table.loc[state]['R']
            long_term_memory.loc[state, action] = reward
        
        # plot the long term memory as a heat map
        im = ax[0].imshow(long_term_memory, cmap='viridis')
        ax[0].set_xlabel('Actions')
        ax[0].set_ylabel('States')
        ax[0].set_xticks(range(len(long_term_memory.columns)))
        ax[0].set_yticks(range(len(long_term_memory.index)))
        ax[0].set_xticklabels(long_term_memory.columns, rotation=90)
        ax[0].set_yticklabels(long_term_memory.index, rotation=0)
        # Create a colorbar
        cbar = ax[0].figure.colorbar(im, ax=ax[0])
        cbar.set_label('Reward')  # Set the label for the colorbar

        plt.tight_layout()
        fig.subplots_adjust(wspace=2)

         # Save the plot in a directory
        directory_path = directory_path + 'individual/'
        file_name = f'Memorytable{name}.png'  
        # Combine the directory path and file name
        file_path = directory_path + file_name
        # Save the plot
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return file_path
        



    # return the action when its in memory or close to state otherwise 
    def get_action_from_memory(self, next_state):
        d = next_state.self_sum()
        tolerance = 2
        in_memory = False
        a = None
    

        #check if its in long term memory first
        best_state, l_action, in_long_term_memory = self.read_memory_table(memory_type='long', state=next_state)
        best_state, s_action, in_short_term_memory = self.read_memory_table(memory_type='short', state=next_state)
        if in_long_term_memory:
            in_memory = True
            a = l_action
        #not in long term, check if its in short term
        elif in_short_term_memory:
            in_memory = True
            a = s_action    
        # not in memory sadlys
        else:
            in_memory = False

        return in_memory, a
            
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
    # used to convert actions to strings
    def format_action(self, action):
            formatted_action = f"(({','.join(action[0])}),({','.join(action[1])}))"
            return formatted_action

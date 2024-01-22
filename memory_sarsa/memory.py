import pandas as pd
import numpy as np
import environment
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


class Memory():
    def __init__(self, q_table, gamma, alpha, duration, Actions, Directions, short_term_memory_size=10) -> None:
         # columns: "distance metric-sum of elemnts ^2, optimal action, optimal R_a"
         self.columns =['d', 'a', 'R']
         self.short_term_memory_size = short_term_memory_size
         self.gamma = gamma
         self.alpha = alpha
         self.duration=duration
         self.Actions = Actions
         self.Directions = Directions
         self.size_time = np.zeros(duration)
         self.memory_table = pd.DataFrame(columns=self.columns)
         self.append_using_q_table(q_table=q_table)
         self.size_time[0] = self.memory_table.shape[0]
         self.short_term_Q = pd.DataFrame(columns=q_table.columns, dtype=np.float64)
         
    
    def update_size_time_array(self,i):
        self.size_time[i] = self.memory_table.shape[0]

    # adding to long_term_memory
    def append_using_q_table(self, q_table):
        for s in q_table.index:
            d = s.self_sum()
            a = q_table.columns[np.nanargmax(q_table.loc[s])]
            r = q_table.loc[s,a]
            self._insert(s,d,self.parse_formatted_action(a),r)
    
    # mainly used in communication module, the size isnt used
    def append_q_table_to_memory_q_table(self, q_table):
        for state in q_table.index:
            nearest_state, _, _, in_memory, shift_amount = self.read_memory_table(memory_type='short', state=state)
            if in_memory:
                shifted_row = np.roll(q_table.loc[state,:].to_numpy(), 3*shift_amount)
                new_row = (shifted_row + self.short_term_Q.loc[nearest_state,:].to_numpy())/2
                self.short_term_Q.loc[nearest_state,:] = new_row
            else:
                self.short_term_Q = pd.concat([self.short_term_Q, q_table.loc[state,:].to_frame().T])

    def _insert(self, state, d, a, r):
        # inserting to long term memory
        # find the eq state,action pair which has max rew
        best_state, best_action, best_reward, in_memory, shift_amount = self.read_memory_table(memory_type='long', state=state)
        a = self.get_shifted_action(a, shift_amount)
        if not in_memory:
            new_row = pd.Series([d,a,r], 
                                index=self.columns, 
                                name=state)
            self.memory_table = pd.concat([self.memory_table, new_row.to_frame().T])
        else:
            # see if the reward is significantly better than update the memory
            if self.memory_table['a'][best_state] == a:
                self.memory_table['R'][best_state] = (self.memory_table['R'][best_state]+r)/2
            elif self.memory_table['R'][best_state]+0.005 < r:
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



    def q_table_check_if_state_exist(self, state, action):
        nearby_state, a,r, in_memory, shift_amount = self.read_memory_table(memory_type='short', state=state)
        action = self.get_shifted_action(action, shift_amount)
        # no similar states, so a unique state found!
        if not in_memory:
            new_row = pd.Series([0] * len(self.short_term_Q.columns), 
                                index=self.short_term_Q.columns, 
                                name=state)
            self.short_term_Q = pd.concat([self.short_term_Q, new_row.to_frame().T])
            return state, action
        else:
            return nearby_state, action

    def update_q_table(self,current_state, current_action, reward, state_next, action_next, done):
        state = current_state
        action = current_action

        state, action = self.q_table_check_if_state_exist(state=state, action=action)
        state_next, action_next = self.q_table_check_if_state_exist(state=state_next, action=action_next)

        action_string = self.format_action(action)
        action_next_string = self.format_action(action_next)
        
        q_value_predict = self.short_term_Q.loc[state, action_string]
        if not done:
            q_value_real = reward + self.gamma * self.short_term_Q.loc[state_next, action_next_string]
        else:
            q_value_real = reward
        self.short_term_Q.loc[state, action_string] += self.alpha * (q_value_real - q_value_predict)
    
    # def read_memory_table(self, memory_type, state, tolerence=1):
    #     shift_amount = 0 
    #     reward = 0
    #     if(memory_type=='long'):
    #         memory = self.memory_table  
    #     elif(memory_type=='short'):
    #         memory = self.short_term_Q

    #     if(memory.shape[0]==0):
    #         in_memory = False
    #         return None, None, reward, in_memory, shift_amount

    #     if state in memory.index:
    #         in_memory = True
    #         # basically we know the state exists in the memory, so just use it as it is
    #         if memory_type =='long':
    #             nearby_state = state
    #             action = self.memory_table.loc[nearby_state]['a']
    #         else:
    #             nearby_state = state
    #             target_actions = self.short_term_Q.loc[state, :]
    #             target_actions = target_actions.reindex(np.random.permutation(target_actions.index))
    #             target_action = target_actions.idxmax()
    #             reward = self.short_term_Q.loc[nearby_state, target_action]
    #             action = self.parse_formatted_action(target_action)
                
    #     else:
    #         # the state is not in memeory, find a nearby state and use it 
            
    #         data = np.vstack(memory.index.map(environment.X_state.to_numpy).values)
    #         kd = KDTree(data)
    #         indexes = kd.query_ball_point(state.to_numpy(), r=tolerence)
    #         in_memory = True

    #         # didnt find a nearyb state
    #         if(len(indexes)==0):
    #             in_memory = False
    #             return None, None, reward, in_memory, shift_amount
    #         nearby_points = data[indexes]
    #         nearby_states = [environment.X_state.numpy_to_x_state(nearby_point) for nearby_point in nearby_points]
    #         if memory_type =='long':
    #             nearby_table =  self.memory_table.loc[nearby_states]
    #             nearby_state = nearby_table.index[nearby_table['R'].to_numpy().argmax()]
    #             action = nearby_table.loc[nearby_state]['a']
    #             reward = nearby_table.loc[nearby_state]['R']        
    #         else:
    #             nearby_state,action = self.short_term_Q.loc[nearby_states].stack().idxmax()
    #             action = self.parse_formatted_action(action)
    #             reward = self.short_term_Q.loc[nearby_state, action]
    #     return nearby_state, action, reward, in_memory, shift_amount

    def read_memory_table(self, memory_type, state, tolerence=1):
        shift_amount = 0
        reward = 0
        if(memory_type=='long'):
            memory = self.memory_table  
        elif(memory_type=='short'):
            memory = self.short_term_Q

        if(memory.shape[0]==0):
            in_memory = False
            return None, None, reward, in_memory, shift_amount
        all_equiv_states, matching_states_list = self.get_matching_states_list(state, memory)

        if len(matching_states_list) == 0:
            in_memory = False
            return None, None, reward, in_memory, shift_amount
        in_memory = True
        
        if memory_type =='long':
            nearby_table =  self.memory_table.loc[matching_states_list]
            nearby_state = nearby_table.index[nearby_table['R'].to_numpy().argmax()]
            shift_amount = all_equiv_states[nearby_state]
            action = nearby_table.loc[nearby_state]['a']
            reward = nearby_table.loc[nearby_state]['R']
            action = self.get_shifted_action(action, -shift_amount)
        else:
            nearby_state,action = self.short_term_Q.loc[matching_states_list].stack().idxmax()
            shift_amount = all_equiv_states[nearby_state]
            reward = self.short_term_Q.loc[nearby_state, action]
            action = self.parse_formatted_action(action)
            action = self.get_shifted_action(action, -shift_amount)
        
        return nearby_state, action, reward, in_memory, shift_amount

    def get_matching_states_list(self, state, memory):
        all_equiv_states = self.get_equiv_states(state)
        matching_indices = memory.index.isin(list(all_equiv_states.keys()))
        # Get the states that match 
        matching_states_list = memory.index[matching_indices].tolist()
        # also consider states that have same order 
        similar_order_states_dic = {}
        for state in all_equiv_states.keys():
            shift_amount = all_equiv_states[state]
            similar_order_states = memory.index[memory.index.map(lambda x: self.are_arrays_order_similar(x,state))]
            for similar_order_state in similar_order_states:
                if similar_order_state not in matching_states_list:
                    matching_states_list.append(similar_order_state)
                    similar_order_states_dic[similar_order_state] = shift_amount
        all_equiv_states.update(similar_order_states_dic)
        return all_equiv_states, matching_states_list

    def are_arrays_order_similar(self, state1, state2):
        # Get the sorted indices for each array
        sorted_indices_arr1 = np.argsort(state1.to_numpy())
        sorted_indices_arr2 = np.argsort(state2.to_numpy())
        # Check if the sorted indices are the same
        return np.array_equal(sorted_indices_arr1, sorted_indices_arr2)

    def get_equiv_states(self, state):
        state_arrays = self.generate_translated_arrays(state.to_numpy())
        all_equiv_classes = self.generate_shifted_arrays(state_arrays)
        return all_equiv_classes 

    def generate_shifted_arrays(self, input_list, shift_amount=3):
        # Initialize an empty dictionary to store shifted arrays and rotation amounts
        result_dict = {}

        # Iterate over each array in the input list
        for input_array in input_list:
            # Iterate over the shift positions for each array
            for i in range(0, len(input_array), shift_amount):
                # Create a copy of the input array
                shifted_array = np.roll(input_array, i)

                # Add the shifted array and rotation amount to the dictionary
                result_dict[environment.X_state.numpy_to_x_state(shifted_array)] = i//shift_amount

        return result_dict 

    def get_shifted_action(self, action, shift_amount):
        d,l = action
        shifted_d = []
        for k in d:
            j = self.Directions.index(k)
            shifted_d.append(self.Directions[(j+shift_amount)%len(self.Directions)])
        shifted_a = (shifted_d, l)
        if shifted_a not in self.Actions:
            shifted_a = (shifted_d[::-1], l)
        return shifted_a

    def generate_translated_arrays(self, input_array):
        # Initialize an empty list to store the result arrays
        result_list = []
        result_list.append(input_array)

        # Iterate over the indices of the input array
        for i in range(len(input_array)):
            # Create a copy of the input array
            new_array = np.copy(input_array)
            new_array_2 = np.copy(input_array)

            # Increase the value at the current index by 1
            new_array[i] += 1
            # Decrease the value at the current index by 1
            new_array_2[i] -= 1

            # Append the new array to the result list
            result_list.append(new_array)
            if new_array_2[i] >= 0:
                result_list.append(new_array_2)

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
    
    def convert_long_term_memory_to_q_table(self):
        q_table = pd.DataFrame(0, index=self.memory_table.index, columns=self.short_term_Q.columns, dtype=np.float64)
        for state in self.memory_table.index:
            action = self.memory_table.loc[state]['a']
            action = self.format_action(action)
            reward = self.memory_table.loc[state]['R']
            q_table.loc[state, action] = reward
        return q_table



    # return the action when its in memory or close to state otherwise 
    def get_action_from_memory(self, next_state):
        d = next_state.self_sum()
        tolerance = 2
        in_memory = True
        a = None
    

        #check if its in long term memory first
        best_state, l_action, l_reward, in_long_term_memory, shift_amount = self.read_memory_table(memory_type='long', state=next_state)
        best_state, s_action, s_reward, in_short_term_memory, shift_amount = self.read_memory_table(memory_type='short', state=next_state)
        if in_long_term_memory and in_short_term_memory:
            if l_reward >= s_reward:
                a = l_action
            else:
                a = s_action
        elif in_long_term_memory:
            a = l_action
        #not in long term, check if its in short term
        elif in_short_term_memory:
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

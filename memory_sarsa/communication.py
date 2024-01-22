import pandas as pd
import numpy as np
import utils
from typing import List, Callable
import environment
from memory import Memory
import copy

class Communication:
    def __init__(self, A, duration, mu: float = 0.0, sigma:float = 1.0, tau: float = 0.0,
                 agg_func: Callable = np.max, Directions: List[str] = ['E', 'N', 'W', 'S']):
        """
            tau : Noise parameter when updating local memory, 0 means no noise
        """

        self.action_strings = [utils.format_action(action) for action in A]
        self.central_memory = Memory(q_table=pd.DataFrame(columns=self.action_strings, dtype=np.float64), 
                                     gamma=0, alpha=0, duration=duration, Actions=A, Directions=Directions, short_term_memory_size=1000)
        self.tau = tau
        self.agg_func = agg_func
        self.mu = mu
        self.sigma = sigma
       
    def update_central_memory(self, memories: List[pd.DataFrame]) -> None:
        for memory in memories:
            self.central_memory.append_q_table_to_memory_q_table(memory)

    def update_local_memory(self, mem_module: Memory, update_type: str) -> pd.DataFrame:
        noise = np.random.normal(loc = self.mu, scale = self.sigma)
        central_q_table = self.central_memory.short_term_Q
        #Assuming the memory index  will not be used, we will create new X_state objects to the updated local memory which erases the previous X_state objects
        if update_type == 'all':
            memory = central_q_table + self.tau * noise
        elif update_type == 'partial':
            # create a list of states that are in the short term memory or long term memory
            states_in_memory = []
            for state in central_q_table.index:
                _, _,_, in_long_memory, _ = mem_module.read_memory_table(memory_type='long', state=state)
                if in_long_memory:
                    states_in_memory.append(state)
                
            # create a new memory table from central q table with index from states_in_memory
            memory = central_q_table.loc[states_in_memory, :]
            memory = memory + self.tau * noise

        # insert to the long term memory   
        mem_module.append_using_q_table(memory) 

        return mem_module

            
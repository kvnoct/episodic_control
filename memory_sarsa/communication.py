import pandas as pd
import numpy as np
import utils
from typing import List, Callable
import environment

class Communication:
    def __init__(self, A, mu: float = 0.0, sigma:float = 1.0, tau: float = 0.0,
                 agg_func: Callable = np.max):
        """
            tau : Noise parameter when updating local memory, 0 means no noise
        """

        self.action_strings = [utils.format_action(action) for action in A]
        self.central_q_table = pd.DataFrame(columns=self.action_strings, dtype=np.float64)
        self.tau = tau
        self.agg_func = agg_func
        self.mu = mu
        self.sigma = sigma
       
    def update_central_memory(self, memories: List[pd.DataFrame]) -> None:
        tmp = self.central_q_table.copy()
        for memory in memories:
            tmp = pd.concat([tmp, memory])  
            
        tmp['state'] = tmp.index
        tmp['state'] = tmp['state'].apply(lambda x: x.to_numpy()).apply(lambda x: '-'.join(x.astype(str)))

        melted_tmp = tmp.melt(id_vars = ['state'], var_name = 'action', value_name = 'Q(s, a)')

        #Aggregate the Q-value
        agg_tmp = melted_tmp.groupby(by = ['state', 'action'])['Q(s, a)'].apply(lambda x: self.agg_func(x)).reset_index()

        #Reorganize df to match the original shape
        agg_tmp = agg_tmp.pivot_table(index = ['state'], columns = 'action', values = 'Q(s, a)').reset_index()
        agg_tmp.index = agg_tmp['state'].apply(lambda x: np.array(x.split('-')).astype(int)).apply(lambda x: environment.X_state.numpy_to_x_state(x))
        agg_tmp = agg_tmp.drop(columns = ['state'])

        self.central_q_table = agg_tmp

    def update_local_memory(self, memory: pd.DataFrame, update_type: str) -> pd.DataFrame:
        noise = np.random.normal(loc = self.mu, scale = self.sigma)

        #Assuming the memory index method will not be used, we will create new X_state objects to the updated local memory which erases the previous X_state objects
        if update_type == 'all':
            memory = self.central_q_table + self.tau * noise
        elif update_type == 'partial':
            #Only update the known state
            self.central_q_table['state_idx'] = self.central_q_table.index
            self.central_q_table['state_idx'] = self.central_q_table['state_idx'].apply(lambda x: x.to_numpy()).apply(lambda x: '-'.join(x.astype(str)))

            memory['state_idx'] = memory.index
            memory['state_idx'] = memory['state_idx'].apply(lambda x: x.to_numpy()).apply(lambda x: '-'.join(x.astype(str)))

            tmp = pd.merge(left= self.central_q_table, right = memory, on = 'state_idx', how = 'inner', suffixes = ['', '_old'])
            tmp = tmp.loc[:, ~tmp.columns.str.endswith('_old')]
            memory = tmp


            self.central_q_table = self.central_q_table.drop(columns = ['state_idx'])
            memory.index = memory['state_idx'].apply(lambda x: np.array(x.split('-')).astype(int)).apply(lambda x: environment.X_state.numpy_to_x_state(x))
            memory.index.name = 'state'
            memory = memory.drop(columns = ['state_idx'])
            memory = memory + self.tau * noise
            

        return memory

            
## Author: Clara Rodrigo Gonzalez

import numpy as np
import pandas as pd
import numpy.random as random
from itertools import product

class Parameters():
    
    def __init__(self, algo_name, type='random', N=20, space_mult=2):
        """
        Inputs:
         - algo_name
         - type of search = 'random' / 'grid'
         - N : only used in grid search. Determines number of discretizations per param
         - space_mult : only used in random search. Determines bounds for a parameter, given a default. 
                        min = default/space_mult
                        max = default * space_mult
                        
        All parameter names are according to the algo documentation except for:
            - smoothing
            - spacing
        """
        super(Parameters, self).__init__()
        self.name = algo_name

        if type=='grid':
        #     self.file = pd.read_csv('bounds/' + algo_name + '_bounds.csv')
            self.N = N                      # used to create discretizations for grid search
        
        self.space_mult = space_mult        # this will be used to determine the min/maxs

        # ---------------------- Constant params (across algos) ---------------------- #
        self.constant_params = {'grad_step':        0.2,
                                'metric':           'SSD',
                                'num_iter':         200,
                                'num_pyramids':     4,
                                'opt_end_thresh':   1.e-3,
                                'num_dim':          2}

        # --------------------------- Params for each algo --------------------------- #
        # Format: 'param name': [min default max]
        if self.name == 'ants':
            self.var_params = {'flow_sigma':        3,                  # smoothing of update field
                               'total_sigma':       1,                  # smoothing of deformation field - default should be 0
                               'type_of_transform': ['SyN']}
            
        if self.name == 'ceritoglu':
            self.var_params = {'a':                 5,                  # smoothing kernel
                               'p':                 2,                  # smoothing power ?
                               'sigma':             2.0,
                               'sigmaR':            1.0,
                               'nt':                5}                  # int: num steps in velocity field

        if self.name == 'dipy':
            self.var_params = {'smoothing':   0.2}                      # smoothing of field 

        if self.name == 'dipy_custom':
            self.var_params = {'smoothing':   0.2}                      # smoothing of field 

        if self.name == 'dramms':
            self.var_params = {'spacing':           4,                  # How many control points per dimension
                               'smoothing':         0.2,                # smoothing of field
                               'saliency_weight':   [0, 2]}             # whether we use mutual saliency - read paper for info
              
        if self.name == 'imregdemons':
            self.var_params = {'smoothing':         1.0}

        if self.name == 'kroon':
            self.var_params = {'sigmaFluid':        4,
                               'sigmaDiff':         1,
                               'interpolation':     ['linear','cubic'],
                               'alpha':             4}
            
        if self.name == 'niftyreg':
            self.var_params = {'spacing':           5,
                               'be_weight':         0.005,
                               'le_weight':         0.0,
                               'l2_weight':         0.0,
                               'jac_weight':        0.0,
                               'smoothGrad':        0}

            
        # ------------------------------ Non-invertible ------------------------------ #
        if self.name == 'elastix':
            self.var_params = {'order':           [1,2,3],
                               'spacing':         16.0,
                               'passive_edge_w':  0,
                               'cyclic':          ['on','off']
                                }

        if self.name == 'fnirt':
            self.var_params = {'warpres':           10, # This applies to all directions as % of image size
                               'splineorder':       [2,3],
                               'regmod':            ['bending_energy', 'membrane_energy'],
                               'smoothing':         6,
                               'lambda':            150}

        if self.name == 'mirtk':
            self.var_params = {'spacing':           4,
                               'linear_en':         0.001,
                               'be_en':             0,
                               'top_en':            0,
                               'vol_en':            0,
                               'jac_en':            0}

        if self.name == 'scikit':
            self.var_params = {'smoothing':         15.0,
                               'tightness':         0.3,
                                'num_warp':         5}

        if self.name == 'twostage':
            self.var_params = {'smoothing':         30,
                               'spacing':           32}
            
    def set_manually(self, params):
        """
        Manually set any number of parameters for existing Parameter variable. 
        Ideally I recommend doing use_with_defaults() before using this function so there arent empty param values. 
        Inputs:
          - params: dictionary of form {param_name: param_value}
        """
        params = self.tester(params)
        self.params = {**self.constant_params, **self.var_params}
        
        for i in range(len(params)):
            self.params[list(params.keys())[i]] = params[list(params.keys())[i]]
        
    def set_all_manually(self, params):
        """
        Set all parameters for algo. 
        """
        self.params = params
        self.tester()

    def use_with_defaults(self):
        """
        Just sets all the defaults (seen above) for the specified implementation. 
        """
        self.params = {**self.constant_params, **self.var_params}
        keys = list(self.params.keys())

        for i in range(len(self.params)):
            if(type(self.params[keys[i]]) == list):
                self.params[keys[i]] = self.params[keys[i]][0]

    def get_defaults(self):
        """
        Returns default values (as above) for the implementation chosen as dict
        """
        return {**self.constant_params, **self.var_params}

    def get_bounds(self):
        """
        Returns bounds for chosen implementation. Defined as normal distribution centered
        around default, with the std being self.space_mult.
        Edge cases:
          - if default == 0, i have given it a small number range idk
          - if we can only do 1 categorical option, ive set it up so it'll be between 0 and 0.4, so that int(np.round()) can
            always be used to index it. Maybe this is redundant

        Output: dict {param_name: [param_min, param_max]}
        """

        self.params = {**self.constant_params, **self.var_params}
        keys = list(self.params.keys())

        bounds = {}
        for i in range(len(self.params)):
            
            default = self.params[keys[i]]
            if(default == 0): 
                default = 1.e-6
                values = [0.001, 0.1]

            if(type(default) == float):
                values = [default/self.space_mult, default*self.space_mult]

            if(type(default) == list):
                values = [0, len(default)-1]

            if(type(default) == int):
                values = [np.round(np.abs(default/self.space_mult)), np.round(np.abs(default*self.space_mult))]

            if(type(default) == str):
                values = [0,0.4]

            bounds[keys[i]] = values

        return bounds

    def get_random_combination(self):

        comb = self.var_params.copy()
        keys = list(self.var_params.keys())

        for i in range(len(self.var_params)):
            
            default = comb[keys[i]]
            if(default == 0): 
                default = 1.e-6

            if(type(default) == float):
                comb[keys[i]] = np.abs(random.normal(loc=default, scale=self.space_mult)).item()

            if(type(default) == list):
                index = round(random.uniform(0, len(comb[keys[i]]))) - 1
                comb[keys[i]] = comb[keys[i]][index]

            if(type(default) == int):
                comb[keys[i]] = int(np.abs(round(random.normal(loc=default, scale=self.space_mult)))) # doing round and abs in case its <0

        self.params = {**self.constant_params, **comb}
        self.tester()

    def tester(self, *kwargs):

        if(len(kwargs) == 0):
            to_test = self.params
        else:
            to_test = kwargs[0]

        if(to_test == 'ceritoglu'):
            if(to_test['nt'] <= 0):
                to_test['nt'] = 1

        # for categorical cases - we want to convert floats to one of the options 
        keys = list(to_test.keys())
        defaults = self.get_defaults()
        for i in range(len(to_test)):
            if(type(defaults[keys[i]]) in [str,list]):
                if(type(to_test[keys[i]]) in [np.float64, float, int]):   
                    to_test[keys[i]] = defaults[keys[i]][int(np.round(to_test[keys[i]]))]
        return to_test
            
    def set_discretized_space(self, bounds):
        """
        ¡¡THIS IS PROBABLY DEPRECATED!!
        This is from when I thought we were doing a grid search. 
        Given an N, discretizes the scale space of each parameter
        using the manually inputed bounds. 
        """
        self.bounds = {}

        self.debt = 0

        # deal with metric stuff first
        if(bounds.columns == 'metric').any():
            self.bounds['metric'] = bounds.iloc[0,2].split(',')
            bounds = bounds.drop(columns=['metric'])

        self.numvars = bounds.shape[1]
        self.n = int(self.N/self.numvars)                      

        for var_name in range(self.numvars):
            if(self.debt != 0): # if nonzero it will always be positive
                self.n = self.n + self.debt/bounds.shape[1]

            min = bounds.iloc[0, var_name]
            max = bounds.iloc[1, var_name]
            step = (max-min)/self.n

            if min == max:
                values = [min]

            elif(type(min) == np.float64):
                values = np.arange(min, max, step)

            elif(type(min) == int):
                if(round(step) == 0):
                    step = 1
                    values = list(range(min, max, step))

                    if(len(values) < self.n):
                        self.debt = self.n - len(values)
                    
                    if(len(values) > self.n):
                        values.pop(random.randint(0,len(values)-1))


                values = list(range(min, max, round(step)))

            self.bounds[bounds.columns[var_name]] = values

            bounds.drop(columns=bounds.columns[var_name])

    def get_combinations(self):
        values = list(self.bounds.values())

        # Find all combinations
        all_combinations = list(product(*values))
        return all_combinations
    
    def get_next_combination(self, idx):
        
        combination = self.get_combinations()[idx]

        params = {'name' : self.name}
        for i in range(len(self.bounds.keys())):
            params[list(self.bounds.keys())[i]] = combination[i]

        self.params = params

    def get_num_combinations(self):
        return len(self.get_combinations())
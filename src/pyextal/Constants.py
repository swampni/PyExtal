"""This module defines a dictionary of constants used throughout the PyExtal package.

The `constant` dictionary holds various configuration parameters for different
optimization routines, including method names, boundaries, and options for both
coarse and fine-grained optimizations.
"""

constant = {'NCORES':8,
            'coarse_geometry':{
                'method':'brent',
                'bracket':[-5, 5],
                'options':{'maxiter': 200, 'disp': True, 'xtol': 1e-3}                
            },
            'coarse_DWF':{
                'method':'Powell',
                'boundary':(0.0,2.0),
                'options':{'maxiter': 10000, 'disp': True, 'fatol': 1e-5, 'xatol':1e-5, 'adaptive':True}
            },
            'coarse_XYZ':{
                'method':'Nelder-Mead',
                'boundary':(-0.005,0.005),
                'options':{'maxiter': 10000, 'disp': True, 'fatol': 1e-5, 'xatol':1e-5, 'adaptive':True},
            },
            'fine':{
                'method':'Nelder-Mead',
                'options':{'maxiter': 1000, 'disp': True, 'fatol': 1e-4, 'xatol':1e-4, 'adaptive':True}
            },
            'fine_geometry':{
                'method':'Powell',                
                'options':{'maxiter': 1000, 'disp': False, 'xtol': 1e-1}
            },

}
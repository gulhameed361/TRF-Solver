# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:36:10 2024

@author: gh00616
"""

import sys
import logging

import numpy as np

from pyomo.environ import ConcreteModel, RangeSet, NonNegativeReals, Param, Var, Objective, Constraint, value, Reals, sin, cos, ExternalFunction, sqrt, exp, maximize, minimize, Expression
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveFloat, PositiveInt, NonNegativeFloat, In

from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverResults

from pyomo.environ import SolverFactory

# Import TRF components directly
from TRF import TRF  # Assuming you have this file
# from readgjh import readgjh  # Assuming you have this file
# import getGJH  # Assuming you have this file with `get_gjh` function

import GJHPseudoSolver

# Set up logging
logger = logging.getLogger('TRF Algorithm')
logger.setLevel(logging.DEBUG)

def load():
    pass

# Define the custom TrustRegionSolver directly
class TrustRegionSolver:
    CONFIG = ConfigBlock('Trust Region')

    CONFIG.declare('solver', ConfigValue(
        default='gams',
        description='solver to use, defaults to gams solvers, specify the solver at the end, you may choose independent solvers such as IPOPT as well',
    ))

    CONFIG.declare('solver_options', ConfigBlock(
        implicit=True,
        description='Options to pass to the subproblem solver',
    ))

    CONFIG.declare('max it', ConfigValue(
        default=100,
        domain=PositiveInt,
    ))

    # Initialize trust radius
    CONFIG.declare('trust radius', ConfigValue(
        default=1.0,
        domain=PositiveFloat,
    ))

    # Initialize sample region and radius
    CONFIG.declare('sample region', ConfigValue(
        default=True,
        domain=bool,
    ))

    CONFIG.declare('sample radius', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
    ))

    # Placeholder for 'radius max', value to be set in __init__
    CONFIG.declare('radius max', ConfigValue(
        default=None,
        domain=PositiveFloat,
    ))
    
    # # Placeholder for 'step max', value to be set in __init__
    # CONFIG.declare('step max', ConfigValue(
    #     default=None,
    #     domain=PositiveFloat,
    # ))

    # Termination tolerances
    CONFIG.declare('ep i', ConfigValue(
        default=1e-5,
        domain=PositiveFloat,
    ))
    
    CONFIG.declare('ep s', ConfigValue(
        default=1e-5,
        domain=PositiveFloat,
    ))

    CONFIG.declare('ep delta', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
    ))

    CONFIG.declare('ep chi', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
    ))

    CONFIG.declare('delta min', ConfigValue(
        default=1e-6,
        domain=PositiveFloat,
        description='delta min <= ep delta',
    ))

    # Compatibility Check Parameters
    CONFIG.declare('kappa delta', ConfigValue(
        default=0.8,
        domain=PositiveFloat,
    ))

    CONFIG.declare('kappa mu', ConfigValue(
        default=1.0,
        domain=PositiveFloat,
    ))

    CONFIG.declare('mu', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    CONFIG.declare('ep compatibility', ConfigValue(
        default=None,  # Placeholder, to be set in __init__
        domain=PositiveFloat,
        description='Suggested value: ep compatibility == ep i',
    ))

    CONFIG.declare('compatibility penalty', ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
    ))

    # Criticality Check Parameters
    CONFIG.declare('criticality check', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
    ))

    # Trust region update parameters
    CONFIG.declare('gamma c', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    CONFIG.declare('gamma e', ConfigValue(
        default=2.5,
        domain=PositiveFloat,
    ))

    # Switching Condition
    CONFIG.declare('gamma s', ConfigValue(
        default=2.0,
        domain=PositiveFloat,
    ))

    CONFIG.declare('kappa theta', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
    ))

    CONFIG.declare('theta min', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
    ))

    # Filter
    CONFIG.declare('gamma f', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('gamma theta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('theta max', ConfigValue(
        default=50,
        domain=PositiveInt,
    ))

    # Ratio test parameters (for theta steps)
    CONFIG.declare('eta1', ConfigValue(
        default=0.05,
        domain=PositiveFloat,
    ))

    CONFIG.declare('eta2', ConfigValue(
        default=0.2,
        domain=PositiveFloat,
    ))

    # Output level (replace with real print levels)
    CONFIG.declare('print variables', ConfigValue(
        default=False,
        domain=bool,
    ))

    # Sample Radius reset parameter
    CONFIG.declare('sample radius adjust', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    # Default reduced model type
    CONFIG.declare('reduced model type', ConfigValue(
        default=0,
        domain=In([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        description='0 = Linear, 1 = Quadratic, 2 = Simple Quadratic, 3 = Gaussian Processes, 4 = Taylor Series Approximation, 5 = Taylor Series Approximation with Linear Basis, 6 = Taylor Series Approximation with Quadratic Basis, 7 = Taylor Series Approximation with SImplified Quadratic Basis, 8 = Taylor Series Approximation with GP Basis, 9 = Hybrid Taylor Series and GP',
    ))
    
    # Default algorithm type
    CONFIG.declare('algorithm type', ConfigValue(
        default=0,
        domain=In([0, 1, 2, 3, 4]),
        description='0 = Normal Trust-region Subproblem, 1 = Trust-region Subproblem with Simple Diagonal Loading, 2 = Trust-region Subproblem with Projected Hessian_based Constraint (Clamped Version), 3 = Trust-region Subproblem with Projected Hessian_based Constraint (Absolute Version), 4 = Trust-region Subproblem with Adaptive Projected Hessian',
    ))

    def __init__(self, **kwargs):
        self.config = self.CONFIG(kwargs, preserve_implicit=True)

        # Set 'radius max' if not already provided
        if self.config['radius max'] is None:
            self.config['radius max'] = 1000.0 * self.config['trust radius']
            
        # # Set 'step max' if not already provided
        # if self.config['step max'] is None:
        #     self.config['step max'] = 1000.0 * self.config['trust radius']

        # Set 'ep compatibility' if not already provided
        if self.config['ep compatibility'] is None:
            self.config['ep compatibility'] = self.config['ep i']

    def solve(self, model, eflist, **kwds):
        # Store all data needed to change in the original model
        model._tmp_trf_data = (list(model.component_data_objects(Var)), eflist, self.config)

        # Clone the model to work on it
        inst = model.clone()

        # Call the TRF function on the cloned model
        TRF(inst, inst._tmp_trf_data[1], inst._tmp_trf_data[2])

        # Copy potentially changed variable values back to the original model
        for inst_var, orig_var in zip(inst._tmp_trf_data[0], model._tmp_trf_data[0]):
            orig_var.set_value(value(inst_var))
            
class Tee:
    """Custom class to write to both console and file simultaneously."""
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout  # Save original stdout

    def write(self, message):
        self.stdout.write(message)  # Print to console
        self.file.write(message)    # Write to file

    def flush(self):
        self.stdout.flush()
        self.file.flush()
    

# #################################### Grey Box Optimisation Model ###############################################
# Model 2 (Eason's Example), it is solved using the default values of the tuning parameter, see notes for the default values

# Define the optimisation model
model = ConcreteModel()
model.x = Var(range(5), domain=Reals, initialize=2.0)
model.x[4] = 1.0

# Define an external function
def blackbox(a, b):
    return sin(a - b)

bb = ExternalFunction(blackbox)

# Define the objective and constraints

model.obj = Objective(expr=(model.x[0] - 1.0)**2 + (model.x[0] - model.x[1])**2 + (model.x[2] - 1.0)**2 +
          (model.x[3] - 1.0)**4 + (model.x[4] - 1.0)**6)

model.c1 = Constraint(expr=model.x[3] * model.x[0]**2 + bb(model.x[3], model.x[4]) == 2 * sqrt(2.0))
model.c2 = Constraint(expr=model.x[2]**4 * model.x[1]**2 + model.x[1] == 8 + sqrt(2.0))

# Initialise the TrustRegionSolver with the necessary configurations/options, see default options (listed above in the same code)
solver = TrustRegionSolver(solver ='ipopt', algorithm_type=0, reduced_model_type=0)

# Define an external function list (eflist) as needed
eflist = [bb]
    
# Open the file and redirect stdout safely
filename = f"Model_1_Eason_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

try:
    with open(filename, 'w') as f:
        tee = Tee(f)
        sys.stdout = tee  # Redirect stdout to file and console

        # Solve the model and display results
        solver.solve(model, eflist)
        model.display()

finally:
    # Restore original stdout safely
    sys.stdout = sys.__stdout__

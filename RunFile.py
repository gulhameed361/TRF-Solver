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
    

# #################################### Model 0 ###############################################
# # Model 1 (Bieglers Example), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()
# model.x = Var(range(2), domain=Reals, initialize=0.)

# # Define an external function
# def blackbox(a):
#     return (a**3) + (a**2) - a

# bb = ExternalFunction(blackbox)

# # Define the objective and constraints
# model.obj = Objective(
#     expr=(model.x[0])**2 + (model.x[1])**2)

# model.c1 = Constraint(expr=bb(model.x[0]) + model.x[0] + 1 == model.x[1])


# # Initialize the TrustRegionSolver with necessary configurations ep_s=1e-3, delta_min=1e-5, config.ep_i, delta_min=1e-4, ep_i=1e-4, delta_min=0.1e-4, ep_i=1e-5, ... delta_min=1e-3, ep_delta=1e-2, ep_i=1e-3, ep_s=4.6e-3 
# solver = TrustRegionSolver(solver ='ipopt', max_it=500, trust_radius=10, sample_radius=1, algorithm_type=1, reduced_model_type=0, gamma_e=10, gamma_c=0.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # solver.solve(model, eflist)
# # model.display()
    
# # Open the file and redirect stdout safely
# filename = f"Model_0_Biegler_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ######################################## Model 1 ###############################################
# # Model 1 (Eason's Example), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, ep_delta=1e-5
# # Define the optimization model
# model = ConcreteModel()
# model.x = Var(range(5), domain=Reals, initialize=2.0)
# model.x[4] = 1.0

# # Define an external function
# def blackbox(a, b):
#     return sin(a - b)

# bb = ExternalFunction(blackbox)

# # Define the objective and constraints
# model.obj = Objective(
#     expr=(model.x[0] - 1.0)**2 + (model.x[0] - model.x[1])**2 + (model.x[2] - 1.0)**2 +
#           (model.x[3] - 1.0)**4 + (model.x[4] - 1.0)**6
# )

# model.c1 = Constraint(expr=model.x[3] * model.x[0]**2 + bb(model.x[3], model.x[4]) == 2 * sqrt(2.0))
# model.c2 = Constraint(expr=model.x[2]**4 * model.x[1]**2 + model.x[1] == 8 + sqrt(2.0))

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=1, sample_radius=0.1, algorithm_type=3, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(model, eflist)

# # # Display the solutio8
# # model.display()
    
# # Open the file and redirect stdout safely
# filename = f"Model_1_Eason_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__


# ####################################### Model 2 ###############################################
# # Model 2 (Yoshio Example), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=100, trust_radius=13, sample_radius=1.3, reduced_model_type=1, gamma_e=15, criticality_check=0.2, ep_delta=1e-5
# # Define the optimization model
# m = ConcreteModel()
# m.x1 = Var(initialize=0)
# m.x2 = Var(bounds=(-2.0, None), initialize=0)

# def blackbox(a,b):
#     return a**2 + b**2
# bb = ExternalFunction(blackbox)

# m.obj = Objective(expr=(m.x1 - 1) ** 2 + (m.x2 - 3) ** 2 + bb(m.x1, m.x2) ** 2)

# m.c1 = Constraint(expr = 2 * m.x1 + m.x2 + 10.0 == bb(m.x1, m.x2))

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=1)
# # solver = TrustRegionSolver(solver ='ipopt', max_it=2000, trust_radius=130, sample_radius=13, reduced_model_type=0, gamma_e=15, criticality_check=0.2, compatibility_penalty=1e-5, ep_compatibility=1e-5)
# # solver = TrustRegionSolver(solver ='ipopt', max_it=200, trust_radius=50, sample_radius=5, reduced_model_type=2, gamma_e=15, criticality_check=0.2) delta_min=1e-2, ep_delta=1e-1, ep_i=1e-4

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()
    
# # Open the file and redirect stdout safely
# filename = f"Model_2_Yoshio_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ######################################## Model 3 ###############################################
# # Model 3 (Rastrigin function), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=250, trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=10, criticality_check=0.1, ep_delta=1e-3
# # Define the optimization model

# from math import pi

# m = ConcreteModel()

# # Define constants and also add the terms in the objective function accordingly
# A = 10  # Constant in Rastrigin function
# n = 3   # Number of dimensions (you can increase this value for higher dimensions)

# # Define decision variables (you can add bounds if necessary)
# m.x = Var(range(n), initialize=0, bounds=(-5.12,5.12))

# def blackbox(a):
#     return a**2 - A*cos(2*pi*a)
# bb = ExternalFunction(blackbox)

# # Set the objective to minimize the Rastrigin function
# m.obj = Objective(expr = A * n + m.x[0]**2 - A*cos(2*pi*m.x[0]) + m.x[1]**2 - A*cos(2*pi*m.x[1]) + bb(m.x[2]))  # Minimize

# # Initialize the TrustRegionSolver with necessary configurations, delta_min=1e-4, ep_delta=1e-3
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=10, sample_radius=1, algorithm_type=0, reduced_model_type=4, gamma_e=10, criticality_check=0.1)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()
    
# # Open the file and redirect stdout safely
# filename = f"Model_3_Rastrigin_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ######################################## Model 4 ###############################################
# # Model 4 (Rosen-Suzuki function), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=2000, trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=10, criticality_check=0.1
# # Define the optimization model

# m = ConcreteModel()

# # Define decision variables (you can add bounds if necessary)
# m.x = Var(range(4), initialize=0., bounds=(-2,2))

# def blackbox1(a,b):
#     return 2*a**2 - 21*a + 7*b
# bb1 = ExternalFunction(blackbox1)

# def blackbox2(a,b):
#     return a**2 + 2*b**2
# bb2 = ExternalFunction(blackbox2)

# # Constraint

# m.c1 = Constraint(expr = -(8 - m.x[0]**2 - m.x[1]**2 - m.x[2]**2 - m.x[3]**2 - m.x[0] + m.x[1] - m.x[2] + m.x[3]) <= 0)
# m.c2 = Constraint(expr = -(10 - m.x[0]**2 - 2*m.x[1]**2 - bb2(m.x[2], m.x[3]) + m.x[0] + m.x[3]) <= 0)
# m.c3 = Constraint(expr = -(5 - 2*m.x[0]**2 - m.x[1]**2 - m.x[2]**2 - 2*m.x[0] + m.x[1] + m.x[3]) <= 0)

# # Set the objective to minimize the Rosen-Suzuki function
# m.obj = Objective(expr = m.x[0]**2 + m.x[1]**2 + m.x[3]**2 - 5*m.x[0] - 5*m.x[1] + bb1(m.x[2], m.x[3]))  # Minimize

# # , ep_s=3.3e-2 for A1S0,A2S0, , ep_s=0.015, ep_i=0.0004, , ep_s=0.04, ep_i=0.09
# solver = TrustRegionSolver(solver ='ipopt', max_it=450, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=4, gamma_e=15, criticality_check=0.2)

# # Define an external function list (eflist) as needed
# eflist = [bb1, bb2]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()
    
# # Open the file and redirect stdout safely
# filename = f"Model_4_Rosen-Suzuki_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ######################################## Model 5 ###############################################
# # Model 5 (Toy Hydrology function), it is not solved using default values of the tuning parameters, see notes for the values used
# # trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=10, criticality_check=0.1
# # Define the optimization model

# from math import pi

# m = ConcreteModel()

# # Define decision variables (you can add bounds if necessary)
# m.x = Var(range(2), initialize=0, bounds=(0,1))

# def blackbox(a):
#     return 2*pi*a**2
# bb = ExternalFunction(blackbox)

# # Constraint
# m.c1 = Constraint(expr = 1.5 - m.x[0] - 2*m.x[1] - 0.5*sin(-4*pi*m.x[1] + bb(m.x[0])) <= 0)
# m.c2 = Constraint(expr = m.x[0]**2 + m.x[1]**2 - 1.5 <= 0)

# m.obj = Objective(expr = sum(m.x[i] for i in range(2)))

# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=9, gamma_e=10, criticality_check=0.1)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_5_Toy_Hydrology_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ######################################## Model 6 ###############################################
# # Model 6 (Goldstein-Price function), it is not solved using default values of the tuning parameters, see notes for the values used
# # trust_radius=4, sample_radius=1, reduced_model_type=1, gamma_e=10, criticality_check=0.2
# # Define the optimization model

# m = ConcreteModel()

# # Variables: x1 and x2 are real numbers with no bounds
# m.x1 = Var(initialize=0, bounds=(-2,2))
# m.x2 = Var(initialize=-1, bounds=(-2,2))

# def blackbox1(a,b):
#     return - 14*b + 6*a*b + 3*b**2
# bb1 = ExternalFunction(blackbox1)

# def blackbox2(a,b):
#     return (2*a - 3*b)**2
# bb2 = ExternalFunction(blackbox2)

# # Objective: Minimize the Goldstein-Price function
# m.obj = Objective(expr = (1 + (m.x1 + m.x2 + 1)**2 * (19 - 14*m.x1 + 3*m.x1**2 + bb1(m.x1,m.x2))) * (30 + bb2(m.x1,m.x2) * (18 - 32*m.x1 + 12*m.x1**2 + 48*m.x2 - 36*m.x1*m.x2 + 27*m.x2**2)))


# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=4, sample_radius=1, algorithm_type=4, reduced_model_type=4, gamma_e=10, criticality_check=0.2)

# # Define an external function list (eflist) as needed
# eflist = [bb1, bb2]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_6_Goldstein-Price_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ######################################## Model 7 ###############################################
# # Model 7 (Colville Function, 3 surrogates), it is not solved using default values of the tuning parameters, see notes for the values used
# # trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=10, criticality_check=0.1
# # Define the optimization model

# m = ConcreteModel()

# # Define decision variables (you can add bounds if necessary)
# m.x1 = Var(initialize=80, bounds=(78,102))
# m.x2 = Var(initialize=35, bounds=(33,45))
# m.x3 = Var(initialize=30, bounds=(27,45))
# m.x4 = Var(initialize=30, bounds=(27,45))
# m.x5 = Var(initialize=30, bounds=(27,45))


# def blackbox2(a,b,c):
#     return 0.00002584*a*b - 0.00006663*c*b
# bb2 = ExternalFunction(blackbox2)

# def blackbox3(a,b,c):
#     return 2275.1327*((a*b)**(-1)) - 0.2668*c*((b)**(-1))
# bb3 = ExternalFunction(blackbox3)

# def blackbox4(a,b,c):
#     return 1330.3294*((a*b)**(-1)) - 0.42*c*((b)**(-1))
# bb4 = ExternalFunction(blackbox4)


# # Constraint
# m.c1 = Constraint(expr = bb2(m.x3, m.x5, m.x2) - 0.0000734*m.x1*m.x4 - 1 <= 0)
# m.c2 = Constraint(expr = 0.000853007*m.x2*m.x5 + 0.00009395*m.x1*m.x4 - 0.00033085*m.x3*m.x5 - 1 <= 0)
# m.c3 = Constraint(expr = bb4(m.x2, m.x5, m.x1) - 0.30586*((m.x2*m.x5)**(-1))*m.x3**2 - 1 <= 0)
# m.c4 = Constraint(expr = 0.00024186*m.x2*m.x5 + 0.00010159*m.x1*m.x2 + 0.00007379*m.x3**2 - 1 <= 0)
# m.c5 = Constraint(expr = bb3(m.x3, m.x5, m.x1) - 0.40584*((m.x5)**(-1))*m.x4 - 1 <= 0)
# m.c6 = Constraint(expr = 0.00029955*m.x3*m.x5 + 0.00007992*m.x1*m.x2 + 0.00012157*m.x3*m.x4 - 1 <= 0)


# m.obj = Objective(expr = 5.3578*m.x3**2 + 0.8357*m.x1*m.x5 + 37.2392*m.x1)  # Minimize

# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=100, sample_radius=10, algorithm_type=4, reduced_model_type=0, gamma_e=15, criticality_check=0.1)

# # Define an external function list (eflist) as needed
# eflist = [bb2, bb3, bb4]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_7_Colville_Function_1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ######################################## Model 8 ###############################################
# # Model 8 (Colville Function, 4 surrogates), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=2000, trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=5, criticality_check=1.1

# # Define the optimization model
# m = ConcreteModel()
# # Define decision variables (you can add bounds if necessary)
# m.x1 = Var(initialize=78, bounds=(78,102))
# m.x2 = Var(initialize=33, bounds=(33,45))
# m.x3 = Var(initialize=30, bounds=(27,45))
# m.x4 = Var(initialize=45, bounds=(27,45))
# m.x5 = Var(initialize=37, bounds=(27,45))

# def blackbox1(a,b):
#     return 0.8357*a*b 
# bb1 = ExternalFunction(blackbox1)

# def blackbox2(a,b,c):
#     return 0.00002584*a*b - 0.00006663*c*b
# bb2 = ExternalFunction(blackbox2)

# def blackbox3(a,b,c):
#     return 2275.1327*((a*b)**(-1)) - 0.2668*c*((b)**(-1))
# bb3 = ExternalFunction(blackbox3)

# def blackbox4(a,b,c):
#     return 1330.3294*((a*b)**(-1)) - 0.42*c*((b)**(-1))
# bb4 = ExternalFunction(blackbox4)


# # Constraint
# m.c1 = Constraint(expr = bb2(m.x3, m.x5, m.x2) - 0.0000734*m.x1*m.x4 - 1 <= 0)
# m.c2 = Constraint(expr = 0.000853007*m.x2*m.x5 + 0.00009395*m.x1*m.x4 - 0.00033085*m.x3*m.x5 - 1 <= 0)
# m.c3 = Constraint(expr = bb4(m.x2, m.x5, m.x1) - 0.30586*((m.x2*m.x5)**(-1))*m.x3**2 - 1 <= 0)
# m.c4 = Constraint(expr = 0.00024186*m.x2*m.x5 + 0.00010159*m.x1*m.x2 + 0.00007379*m.x3**2 - 1 <= 0)
# m.c5 = Constraint(expr = bb3(m.x3, m.x5, m.x1) - 0.40584*((m.x5)**(-1))*m.x4 - 1 <= 0)
# m.c6 = Constraint(expr = 0.00029955*m.x3*m.x5 + 0.00007992*m.x1*m.x2 + 0.00012157*m.x3*m.x4 - 1 <= 0)

# # Set the objective to minimize the colville function
# m.obj = Objective(expr = 5.3578*m.x3**2 + bb1(m.x1, m.x5) + 37.2392*m.x1)  # Minimize

# solver = TrustRegionSolver(solver ='ipopt', max_it=2500, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=0, gamma_e=10, criticality_check=0.1) #, ep_i=1e-4, ep_s=1
# # # max_it=2500, trust_radius=1002, sample_radius=10.02, reduced_model_type=2, gamma_e=10, criticality_check=0.2, delta_min=1e-1, ep_delta=1, , ep_s=1e-1
# # (Matern(length_scale=1.0, nu=0.5) + WhiteKernel(noise_level=1e-15, noise_level_bounds=(1e-15, 1e-1))) Use this for GP
# # trust_radius=9199, sample_radius=919.9, A4-A3,S1

# # Define an external function list (eflist) as needed
# eflist = [bb1, bb2, bb3, bb4]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'knitro'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_8_Colville_Function_2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

######################################## Model 9 ###############################################
# Model 9 (Williams Otto), it is not solved using default values of the tuning parameters, see notes for the values used
# max_it=2000, trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=5, criticality_check=1.1
# Define the optimization model

m = ConcreteModel()

# Parameters

m.p = Param(default=50)

# Variables
m.V = Var(bounds=(0.03,0.1), initialize = 0.065)
m.T = Var(bounds=(5.8,6.8), initialize = 6.519028421456637)
m.Fp = Var(bounds=(0,4.763), initialize = 4.763000047375733)
m.Fpurge = Var(bounds=(0, None), initialize=32.875015708556624)
m.Fg = Var(bounds=(0, None), initialize=2.873098603873709)
m.Feff = Var(range(6))
m.Feff[0] = 37.573051115542235
m.Feff[1] = 119.20155505418737
m.Feff[2] = 7.261892475334615
m.Feff[3] = 136.84864098634483
m.Feff[4] = 18.447864146010218
m.Feff[5] = 2.873098603873709
m.Feff_sum = Var(initialize=322.206102381293)
m.FR = Var(range(6))
m.FR[0] = 33.64637460158599
m.FR[1] = 106.74406403971133
m.FR[2] = 6.50296814571112
m.FR[3] = 122.54689203134407
m.FR[4] = 16.51991864453813
m.FR[5] = 2.572837419989088
m.Fa = Var(bounds=(1, None), initialize=12.414712690892943)
m.Fb = Var(bounds=(1, None), initialize=28.096401668913114)
# m.r = Var(range(3), initialize=2.)
# m.r[0] = 8.51497833068342
# m.r[1] = 7.165267379439701
# m.r[2] = 1.9384277124071796
m.x = Var(range(6), bounds=(0,1))
m.x[0] = 0.11661185445544377
m.x[1] = 0.36995436825440636
m.x[2] = 0.02253803519444592
m.x[3] = 0.42472392662632236
m.x[4] = 0.05725485647127553
m.x[5] = 0.008916958998106003
m.n = Var(bounds=(0,1), initialize=0.10450778942281834)

def blackbox1(a,b,c,d):
    return (5.9755*(10**9) * exp(-120/(a)) * b * c * d * 50)

def blackbox2(a,b,c,d):      
    return (2.5962*(10**12) * exp(-150/(a)) * b * c * d * 50)

def blackbox3(a,b,c,d):
    return (9.6283*(10**15) * exp(-200/(a)) * b * c * d * 50)

bb1 = ExternalFunction(blackbox1)
bb2 = ExternalFunction(blackbox2)
bb3 = ExternalFunction(blackbox3)


# Objective
m.obj = Objective(expr = (100 * ((2207 * (m.Fp)) + (50 * (m.Fpurge)) - (168 * (m.Fa)) -(252 * (m.Fb)) - (2.22 * (m.Feff_sum)) - (84 * (m.Fg)) - (60 * (m.V) * m.p)) / (600 * (m.V) * m.p)), sense=maximize)

# Constraints
# m.c1 = Constraint(expr = m.r[0] == bb1(m.T, m.x[0], m.x[1], m.V))
# m.c2 = Constraint(expr = m.r[1] == bb2(m.T, m.x[1], m.x[2], m.V))
# m.c3 = Constraint(expr = m.r[2] == bb3(m.T, m.x[4], m.x[2], m.V))

m.c4 = Constraint(expr = m.Feff[0] == m.Fa + (m.FR[0]) - bb1(m.T, m.x[0], m.x[1], m.V))
m.c5 = Constraint(expr = m.Feff[1] == m.Fb + (m.FR[1]) - (bb1(m.T, m.x[0], m.x[1], m.V) + bb2(m.T, m.x[1], m.x[2], m.V)))
m.c6 = Constraint(expr = m.Feff[2] == (m.FR[2]) + (2 * bb1(m.T, m.x[0], m.x[1], m.V)) - (2 * bb2(m.T, m.x[1], m.x[2], m.V)) - bb3(m.T, m.x[4], m.x[2], m.V))
m.c7 = Constraint(expr = m.Feff[3] == (m.FR[3]) + (2 * bb2(m.T, m.x[1], m.x[2], m.V)))
m.c8 = Constraint(expr = m.Feff[4] == (0.1 * (m.FR[3])) + bb2(m.T, m.x[1], m.x[2], m.V) - (0.5 * bb3(m.T, m.x[4], m.x[2], m.V)))
m.c9 = Constraint(expr = m.Feff[5] == 1.5 * bb3(m.T, m.x[4], m.x[2], m.V))
m.c10 = Constraint(expr = m.Feff_sum == sum(list(m.Feff.values())))
m.c11 = Constraint(expr = m.Feff[0] == m.Feff_sum * m.x[0])
m.c12 = Constraint(expr = m.Feff[1] == m.Feff_sum * m.x[1])
m.c13 = Constraint(expr = m.Feff[2] == m.Feff_sum * m.x[2])
m.c14 = Constraint(expr = m.Feff[3] == m.Feff_sum * m.x[3])
m.c15 = Constraint(expr = m.Feff[4] == m.Feff_sum * m.x[4])
m.c16 = Constraint(expr = m.Feff[5] == m.Feff_sum * m.x[5])

m.c17 = Constraint(expr = m.Fg == m.Feff[5])

m.c18 = Constraint(expr = m.Fp == m.Feff[4] - (0.1 * m.Feff[3]))

m.c19 = Constraint(expr = m.Fpurge == m.n *((m.Feff[0]) + (m.Feff[1]) + (m.Feff[2]) + (1.1 * (m.Feff[3]))))

m.c20 = Constraint(expr = m.FR[0] == (1 - m.n) * (m.Feff[0]))
m.c21 = Constraint(expr = m.FR[1] == (1 - m.n) * (m.Feff[1]))
m.c22 = Constraint(expr = m.FR[2] == (1 - m.n) * (m.Feff[2]))
m.c23 = Constraint(expr = m.FR[3] == (1 - m.n) * (m.Feff[3]))
m.c24 = Constraint(expr = m.FR[4] == (1 - m.n) * (m.Feff[4]))
m.c25 = Constraint(expr = m.FR[5] == (1 - m.n) * (m.Feff[5]))

solver = TrustRegionSolver(solver ='ipopt', max_it=5000, algorithm_type=4, reduced_model_type=4, gamma_e=2.5)
# # max_it=2500, trust_radius=1002, sample_radius=10.02, reduced_model_type=2, gamma_e=10, criticality_check=0.2, delta_min=1e-1, ep_delta=1, ep_s=1e-1, ep_delta=1e-2

# Define an external function list (eflist) as needed
eflist = [bb1, bb2, bb3]

# If using default solver settings, you need to give a solver name available in gams
# solver.config['solver_options']['solver'] = 'knitro'

# # Solve the model using TrustRegionSolver
# solver.solve(m, eflist)

# # Display the solution
# m.display()

# Open the file and redirect stdout safely
filename = f"Model_9_Williams_Otto_1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

try:
    with open(filename, 'w') as f:
        tee = Tee(f)
        sys.stdout = tee  # Redirect stdout to file and console

        # Solve the model and display results
        solver.solve(m, eflist)
        m.display()

finally:
    # Restore original stdout safely
    sys.stdout = sys.__stdout__

# # ######################################## Model 10 ###############################################
# # Model 10 (Alkylation Process, 1 surrogates), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=2000, trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=5, criticality_check=1.1

# # Define the optimization model
# m = ConcreteModel()

# # Variables
# m.x = Var(RangeSet(0,9), bounds={ 
#     0: (0, 2000), 
#     1: (0, 16000),
#     2: (0, 120),
#     3: (0, 5000),
#     4: (0, 2000),
#     5: (85, 93),
#     6: (90, 95),
#     7: (5.69, 12),
#     8: (1.2, 4),
#     9: (145, 162)
# }, initialize={
#     0: 1309.276241883202,
#     1: 6210.726557172016,
#     2: 120.0,
#     3: 2088.796312316891,
#     4: 1239.055259143405,
#     5: 92.99796451282054,
#     6: 92.66666666666667,
#     7: 5.69,
#     8: 3.090000000000005,
#     9: 145.0
# })

# def blackbox1(a):
#     return (1.12 + (0.12167*a) - (0.0067 * (a**2)))

# bb1 = ExternalFunction(blackbox1)

# # def blackbox2(a):
# #     return (86.35 + (1.098*a) - (0.038 * (a**2)))

# # bb2 = ExternalFunction(blackbox2)

# # Constraints
# m.c1 = Constraint(expr = m.x[3] == m.x[0] * bb1(m.x[7]))
# m.c2 = Constraint(expr = m.x[6] == 86.35 + (1.098 * m.x[7]) - (0.038 * (m.x[7]**2)) + (0.325 * (m.x[5] - 89)))
# m.c3 = Constraint(expr = m.x[8] == 35.28 - (0.222 * m.x[9]))
# m.c4 = Constraint(expr = m.x[9] == (3 * m.x[6]) - 133)
# m.c5 = Constraint(expr = m.x[7] * m.x[0] == m.x[1] + m.x[4])
# m.c6 = Constraint(expr = m.x[4] == (1.22 * m.x[3]) - m.x[0])
# m.c7 = Constraint(expr = (m.x[5] * m.x[3] * m.x[8]) + (m.x[5] * (1000 * m.x[2])) == 98000 * m.x[2])


# # Objective
# m.obj = Objective(expr = ((0.063 * m.x[3] * m.x[6]) - (5.04 * m.x[0]) - (0.035 * m.x[1]) - (10 * m.x[2]) - (3.36 * m.x[4])), sense=maximize)


# solver = TrustRegionSolver(solver ='ipopt', max_it=10000, algorithm_type=4, reduced_model_type=1, gamma_e=13) # , delta_min=1e-2, ep_delta=1e-1
# # max_it=2500, trust_radius=1002, sample_radius=10.02, reduced_model_type=2, gamma_e=10, criticality_check=0.2, delta_min=1e-1, ep_delta=1
# # (Matern(length_scale=1.0, nu=0.5) + WhiteKernel(noise_level=1e-15, noise_level_bounds=(1e-15, 1e-1)))

# # Define an external function list (eflist) as needed
# eflist = [bb1]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'knitro'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_10_Alkylation_Process_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
        


# ####################################### Model 11 ###############################################
# # Model 11 (Powell's Function), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=100, trust_radius=13, sample_radius=1.3, reduced_model_type=1, gamma_e=15, criticality_check=0.2, ep_delta=1e-5
# # Define the optimization model
# m = ConcreteModel()

# # Define variables x1, x2, x3, x4 with no bounds (they are free variables)
# m.x1 = Var(initialize=3, bounds=(-4,5))  # Starting point for x1
# m.x2 = Var(initialize=-0.256837, bounds=(-4,5)) # Starting point for x2
# m.x3 = Var(initialize=0.517729, bounds=(-4,5))  # Starting point for x3
# m.x4 = Var(initialize=2.244261, bounds=(-4,5))  # Starting point for x4

# def blackbox1(a,b):
#     return (a - 2*b)**4
# bb1 = ExternalFunction(blackbox1)

# def blackbox2(a,b):
#     return 10*(a - b)**4
# bb2 = ExternalFunction(blackbox2)

# m.obj = Objective(expr=(m.x1 + 10 * m.x2)**2 + 5 * (m.x3 - m.x4)**2 + bb1(m.x2,m.x3) + bb2(m.x1,m.x4))

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=2, gamma_e=10, criticality_check=0.2)

# # Define an external function list (eflist) as needed
# eflist = [bb1,bb2]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_11_Powell_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################### Model 12 ###############################################
# # Model 12 (Williams Otto, 2 surrogates), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=2000, trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=5, criticality_check=1.1
# # Define the optimization model

# m = ConcreteModel()

# # Parameters

# m.p = Param(default=50)

# m.V = Var(bounds=(0.03,0.1), initialize = 0.029999990031902183)
# m.T = Var(bounds=(5.8,6.8), initialize = 6.334381620347808)
# m.Fp = Var(bounds=(0,4.763), initialize = 1.182545043774607)
# m.Fpurge = Var(bounds=(0, None), initialize=7.645830710736873)
# m.Fg = Var(bounds=(0, None), initialize=0.5756367174825368)
# m.Feff = Var(range(6))
# m.Feff[0] = 7.733801198801471
# m.Feff[1] = 27.920735704270268
# m.Feff[2] = 1.7429052502938363
# m.Feff[3] = 33.23703343789073
# m.Feff[4] = 4.50624838756368
# m.Feff[5] = 0.5756367174825368
# m.Feff_sum = Var(initialize=75.71636069630252)
# m.FR = Var(range(6))
# m.FR[0] = 6.9342772574607485
# m.FR[1] = 25.034277146365998
# m.FR[2] = 1.5627229002077854
# m.FR[3] = 29.80097356388543
# m.FR[4] = 4.04039034714191
# m.FR[5] = 0.5161271276558164
# m.Fa = Var(bounds=(1, None), initialize=2.7995239592139094)
# m.Fb = Var(bounds=(1, None), initialize=6.6044885127801045)
# # m.r = Var(range(3), initialize=2.)
# # m.r[0] = 8.51497833068342
# # m.r[1] = 7.165267379439701
# # m.r[2] = 1.9384277124071796
# m.r3 = Var(initialize=2)
# m.x = Var(range(6))
# m.x[0] = 0.10214174489740287
# m.x[1] = 0.36875432796169405
# m.x[2] = 0.02301887246383394
# m.x[3] = 0.43896765682127936
# m.x[4] = 0.0595148571078078
# m.x[5] = 0.007602540747981918
# m.n = Var(bounds=(0,1), initialize=0.10338046205074775)

# # # Variables
# # m.V = Var(bounds=(0.03,0.1), initialize = 0.06)
# # m.T = Var(bounds=(5.8,6.8), initialize = 6.0)
# # m.Fp = Var(bounds=(0,4.763), initialize = 1.3028288606710166)
# # m.Fpurge = Var(bounds=(0, None), initialize=9.221837313469361)
# # m.Fg = Var(bounds=(0, None), initialize=0.4135102351089589)
# # m.Feff = Var(range(6))
# # m.Feff[0] = 13.754312507475527
# # m.Feff[1] = 24.311093854225625
# # m.Feff[2] = 3.877928753163475
# # m.Feff[3] = 28.71969631552761
# # m.Feff[4] = 4.174798492223778
# # m.Feff[5] = 0.4135102351089589
# # m.Feff_sum = Var(initialize=75.25134015772495)
# # m.FR = Var(range(6))
# # m.FR[0] = 12.02941918496403
# # m.FR[1] = 21.26230145334887
# # m.FR[2] = 3.391607578777536
# # m.FR[3] = 25.11803230125927
# # m.FR[4] = 3.6512476394897178
# # m.FR[5] = 0.36165296903767546
# # m.Fa = Var(bounds=(1, None), initialize=3.906722661874956)
# # m.Fb = Var(bounds=(1, None), initialize=7.031453747374385)
# # m.r3 = Var(initialize=0.27567349007263925)
# # m.x = Var(range(6))
# # m.x[0] = 0.18277830638840575
# # m.x[1] = 0.3230652610739775
# # m.x[2] = 0.05153301914674682
# # m.x[3] = 0.38165029692927394
# # m.x[4] = 0.05547806169923503
# # m.x[5] = 0.0054950547623611925
# # m.n = Var(bounds=(0,1), initialize=0.1254074546851814)

# def blackbox1(a,b,c,d):
#     return (5.9755*(10**9) * exp(-120/(a)) * b * c * d * 50)

# def blackbox2(a,b,c,d):      
#     return (2.5962*(10**12) * exp(-150/(a)) * b * c * d * 50)


# bb1 = ExternalFunction(blackbox1)
# bb2 = ExternalFunction(blackbox2)


# # Objective
# m.obj = Objective(expr = (100 * ((2207 * (m.Fp)) + (50 * (m.Fpurge)) - (168 * (m.Fa)) -(252 * (m.Fb)) - (2.22 * (m.Feff_sum)) - (84 * (m.Fg)) - (60 * (m.V) * m.p)) / (600 * (m.V) * m.p)), sense=maximize)

# # Constraints
# # m.c1 = Constraint(expr = m.r[0] == bb1(m.T, m.x[0], m.x[1], m.V))
# # m.c2 = Constraint(expr = m.r[1] == bb2(m.T, m.x[1], m.x[2], m.V))
# m.c3 = Constraint(expr = m.r3 == (9.6283*(10**15) * exp(-200/(m.T)) * m.x[4] * m.x[2] * m.V * 50))

# m.c4 = Constraint(expr = m.Feff[0] == m.Fa + (m.FR[0]) - bb1(m.T, m.x[0], m.x[1], m.V))
# m.c5 = Constraint(expr = m.Feff[1] == m.Fb + (m.FR[1]) - (bb1(m.T, m.x[0], m.x[1], m.V) + bb2(m.T, m.x[1], m.x[2], m.V)))
# m.c6 = Constraint(expr = m.Feff[2] == (m.FR[2]) + (2 * bb1(m.T, m.x[0], m.x[1], m.V)) - (2 * bb2(m.T, m.x[1], m.x[2], m.V)) - m.r3)
# m.c7 = Constraint(expr = m.Feff[3] == (m.FR[3]) + (2 * bb2(m.T, m.x[1], m.x[2], m.V)))
# m.c8 = Constraint(expr = m.Feff[4] == (0.1 * (m.FR[3])) + bb2(m.T, m.x[1], m.x[2], m.V) - (0.5 * m.r3))
# m.c9 = Constraint(expr = m.Feff[5] == 1.5 * m.r3)
# m.c10 = Constraint(expr = m.Feff_sum == sum(list(m.Feff.values())))
# m.c11 = Constraint(expr = m.Feff[0] == m.Feff_sum * m.x[0])
# m.c12 = Constraint(expr = m.Feff[1] == m.Feff_sum * m.x[1])
# m.c13 = Constraint(expr = m.Feff[2] == m.Feff_sum * m.x[2])
# m.c14 = Constraint(expr = m.Feff[3] == m.Feff_sum * m.x[3])
# m.c15 = Constraint(expr = m.Feff[4] == m.Feff_sum * m.x[4])
# m.c16 = Constraint(expr = m.Feff[5] == m.Feff_sum * m.x[5])

# m.c17 = Constraint(expr = m.Fg == m.Feff[5])

# m.c18 = Constraint(expr = m.Fp == m.Feff[4] - (0.1 * m.Feff[3]))

# m.c19 = Constraint(expr = m.Fpurge == m.n *((m.Feff[0]) + (m.Feff[1]) + (m.Feff[2]) + (1.1 * (m.Feff[3]))))

# m.c20 = Constraint(expr = m.FR[0] == (1 - m.n) * (m.Feff[0]))
# m.c21 = Constraint(expr = m.FR[1] == (1 - m.n) * (m.Feff[1]))
# m.c22 = Constraint(expr = m.FR[2] == (1 - m.n) * (m.Feff[2]))
# m.c23 = Constraint(expr = m.FR[3] == (1 - m.n) * (m.Feff[3]))
# m.c24 = Constraint(expr = m.FR[4] == (1 - m.n) * (m.Feff[4]))
# m.c25 = Constraint(expr = m.FR[5] == (1 - m.n) * (m.Feff[5]))

# solver = TrustRegionSolver(solver ='ipopt', max_it=2500, trust_radius=1002, sample_radius=10.02, algorithm_type=5, reduced_model_type=4, gamma_e=10, criticality_check=0.2, delta_min=1e-1, ep_delta=1)
# # # max_it=2500, trust_radius=1002, sample_radius=10.02, reduced_model_type=2, gamma_e=10, criticality_check=0.2, delta_min=1e-1, ep_delta=1

# # Define an external function list (eflist) as needed
# eflist = [bb1, bb2]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'knitro'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_12_Williams_Otto_2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# #################################### Model 13 ###############################################
# # Model 13 (Ishan Example), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()
# model.x = Var(range(2), bounds=(-2,2))
# model.x[0] = 0
# model.x[1] = -2.0000000065741883

# # Define an external function
# def blackbox(a,b):
#     return (a**2) + (b**2)

# bb = ExternalFunction(blackbox)

# # Define the objective and constraints
# model.obj = Objective(
#     expr=model.x[0] + model.x[1])

# model.c1 = Constraint(expr= -bb(model.x[0],model.x[1]) <= -1)
# model.c2 = Constraint(expr= bb(model.x[0],model.x[1]) <= 4)
# model.c3 = Constraint(expr= model.x[1] - model.x[0] <= 1)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=30, trust_radius=1, sample_radius=0.1, algorithm_type=5, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_13_Ishan_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ################################ Model 14 ###############################################
# # Model 14 (Gramacy & Lee (2012) Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()
# model.x = Var(bounds=(0.5,2.5), initialize=1)

# # Define an external function
# def blackbox(a):
#     return sin(10*(22/7)*a)/(2*a)

# bb = ExternalFunction(blackbox)


# # Define the objective
# model.obj = Objective(
#     expr=bb(model.x) + (model.x - 1)**4)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=1, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_14_Gramacy&Lee_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ################################ Model 15 ###############################################
# # Model 15 (Easom Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()
# model.x = Var(range(2), bounds=(-100,100), initialize=0.)

# # Define an external function
# def blackbox(a):
#     return cos(a)

# bb = ExternalFunction(blackbox)


# # Define the objective
# model.obj = Objective(
#     expr= -bb(model.x[0]) * bb(model.x[1]) * exp(-(model.x[0]-(22/7))**2 - (model.x[1]-(22/7))**2)
#     )

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=2500, algorithm_type=0, reduced_model_type=4, gamma_e=2.5, ep_compatibility=1e-20, delta_min = 1e-4, ep_delta=1e-3)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_15_Easom_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ################################ Model 16 ###############################################
# # Model 16 (Six-Hump Camel Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()
# model.x1 = Var(bounds=(-3,2), initialize=0.1)
# model.x2 = Var(bounds=(-2,2), initialize=0.1)

# # Define an external function
# def blackbox(a):
#     return 4*(a**2) - 2.1*(a**4) + ((a**6)/3)

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= bb(model.x1) + (model.x1*model.x2) + (-4*(model.x2**2) + 4*(model.x2**4))
#     )

# # Initialize the TrustRegionSolver with necessary configurations delta_min=1e-1, ep_delta=1, ep_compatibility=1e-6
# solver = TrustRegionSolver(solver ='ipopt', max_it=1500, algorithm_type=4, reduced_model_type=4, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_16_Six-Hump_Camel_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ################################ Model 17 ###############################################
# # Model 17 (Three-Hump Camel Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-5,5), initialize=0.)

# # Define an external function
# def blackbox(a):
#     return 2*(a**2) - 1.05*(a**4) + ((a**6)/6)

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= bb(model.x[0]) + (model.x[0]*model.x[1]) + (model.x[1]**2)
#     )

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=1, sample_radius=0.1, algorithm_type=2, reduced_model_type=3, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(model, eflist)

# # # Display the solution
# # model.display()

# # Open the file and redirect stdout safely
# filename = f"Model_17_Three-Hump_Camel_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ################################ Model 18 ###############################################
# # Model 18 (McCormick Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds={ 
#     0: (-1.5, 4), 
#     1: (-3, 4)},
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return sin(a+b)

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= bb(model.x[0],model.x[1]) + (model.x[0]-model.x[1])**2 - 1.5*model.x[0] + 2.5*model.x[1] + 1
#     )

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1...........ep_compatibility=1e-6, criticality_check=0.75, ep_i=1e-2, ep_s=1e-2
# solver = TrustRegionSolver(solver ='ipopt', max_it=300, trust_radius=1100, sample_radius=110, algorithm_type=2, reduced_model_type=3)
# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_18_McCormick_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 19 ###############################################
# # Model 19 (Matyas Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-10,10),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return 0.26*(a**2 + b**2)

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= bb(model.x[0], model.x[1]) - 0.48*model.x[0]*model.x[1]
#     )

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, A4S1, ep_i=1.5e-3, delta_min=1e-3, ep_delta=1e-2, A4S0 ep_i=1.5e-3, delta_min=1e-3, ep_delta=1.5e-3
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=1, gamma_e=10) # A3S1,ep_i=1e-3, ep_s=0.011, delta_min=1e-4, ep_delta=1e-3, A3S2, ep_i=1e-2, ep_s=1e-1, delta_min=1e-4, ep_delta=1e-3

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_19_Matyas_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 20 ###############################################
# # Model 20 (Bohachevsky Function 1), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-100,100),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a):
#     return 0.3*cos(3*(22/7)*a)

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= model.x[0]**2 + 2*model.x[1]**2 - bb(model.x[0]) - 0.4*cos(4*(22/7)*model.x[1]) + 0.7
#     )

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=100, sample_radius=10, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_20_Bohachevsky_Function_1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 21 ###############################################
# # Model 21 (Bohachevsky Function 2), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-100,100),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return 0.3*cos(3*(22/7)*a)*cos(4*(22/7)*b)

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= model.x[0]**2 + 2*model.x[1]**2 - bb(model.x[0],model.x[1]) + 0.3
#     )

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1000, sample_radius=100, algorithm_type=4, reduced_model_type=9, gamma_e=2.5, ep_s=1e-3)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_21_Bohachevsky_Function_2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 22 ###############################################
# # Model 22 (Bohachevsky Function 3), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-100,100),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return 0.3*cos((3*(22/7)*a) + (4*(22/7)*b))

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= model.x[0]**2 + 2*model.x[1]**2 - bb(model.x[0],model.x[1]) + 0.3
#     )

# # Initialize the TrustRegionSolver with necessary configurations, , ep_s=1e-4, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1000, sample_radius=100, algorithm_type=4, reduced_model_type=9, gamma_e=2.5, ep_s=1e-4)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_22_Bohachevsky_Function_3_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 23 ###############################################
# # Model 23 (Himmelblau's Problem), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-5,5),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return a + b**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=(model.x[0]**2 + model.x[1] - 11)**2 + (model.x[0] + model.x[1]**2 - 7)**2, sense=minimize)

# # Constraints
# model.con1 = Constraint(expr=model.x[0]**2 + model.x[1] >= 0)
# model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 7)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, algorithm_type=4, reduced_model_type=3, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_23_Himmelblau's_Problem_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
    
# ############################## Model 24 ###############################################
# # Model 24 (Rosenbrock_Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-2,2),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return a**2 + b**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=(1 - model.x[0])**2 + 100 * (model.x[1] - model.x[0]**2)**2, sense=minimize)

# # Constraints
# model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 1)
# model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 1)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=2, sample_radius=0.2, algorithm_type=4, reduced_model_type=4, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_24_Rosenbrock_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 25 ###############################################
# # Model 25 (Beale's_Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-4.5,4.5),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return a**2 + b**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=(1.5 - model.x[0] + model.x[0] * model.x[1])**2 +
#                         (2.25 - model.x[0] + model.x[0] * model.x[1]**2)**2 +
#                         (2.625 - model.x[0] + model.x[0] * model.x[1]**3)**2, sense=minimize)

# # Constraints
# model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 3)
# model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 5)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=750, trust_radius=45, sample_radius=4.5, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_25_Beale's_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
    
# ############################## Model 26 ###############################################
# # Model 26 (Booth's_Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-10,10),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return a**2 + b**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=(model.x[0] + 2 * model.x[1] - 7)**2 + (2 * model.x[0] + model.x[1] - 5)**2, sense=minimize)

# # Constraints
# model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
# model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 50)

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-3, ep_delta=1e-2
# solver = TrustRegionSolver(solver ='ipopt', max_it=1500, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=0, gamma_e=10)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_26_Booth's_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__


# ############################## Model 27 ###############################################
# # Model 27 (Styblinski-Tang_Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4

# # Define the optimization model
# model = ConcreteModel()

# model.x = Var(range(2), bounds=(-5,5),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return a**2 + b**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=(model.x[0]**4 - 16 * model.x[0]**2 + 5 * model.x[0] + model.x[1]**4 - 16 * model.x[1]**2 + 5 * model.x[1]) / 2, sense=minimize)

# # Constraints
# model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
# model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 50)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=2500, trust_radius=1, sample_radius=0.1, algorithm_type=4, reduced_model_type=2, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_27_Styblinski-Tang_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 28 ###############################################
# # Model 28 (Optimal_Design_of_a_Beam), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Number of beam segments
# N = 10  

# # Define index set for segments
# model.i = RangeSet(1, N)

# # Design variables: Cross-sectional areas of beam segments (must be positive)
# model.A = Var(model.i, bounds=(0.01, 5.0), initialize=1.0)  # Cross-sectional areas

# # External load applied at the end of the beam
# P = 100.0  

# # Material properties
# E = 200e9  # Young's modulus (Pa)
# L = 2.0    # Length of each segment (m)
# rho = 7850  # Density of the material (kg/m³)

# # Define an external function
# def blackbox(a):
#     return 100 / a

# bb = ExternalFunction(blackbox)

# # Define the objective (Minimize total weight of the beam)
# def objective_rule(m):
#     return sum(rho * L * m.A[i] for i in m.i)  # Mass = density * volume
# model.obj = Objective(rule=objective_rule, sense=minimize)

# # Constraints

# # Stress constraints: σ = P/A ≤ allowable stress
# sigma_max = 250e6  # Maximum allowable stress (Pa)

# def stress_constraint_rule(m, i):
#     return (bb(m.A[i])) <= sigma_max
# model.stress_constraints = Constraint(model.i, rule=stress_constraint_rule)

# # Deflection constraints (simplified for demonstration)
# deflection_max = 0.01  # Maximum allowable deflection (m)

# def deflection_constraint_rule(m):
#     return sum((P * L**3) / (3 * E * m.A[i]) for i in m.i) <= deflection_max
# model.deflection_constraint = Constraint(rule=deflection_constraint_rule)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=300, trust_radius=10, sample_radius=1, algorithm_type=0, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_28_Optimal_Design_of_a_Beam_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 29 ###############################################
# # Model 29 (Parameter_Estimation_NLP), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Experimental data (observed values)
# x_data = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])  # Input values
# y_obs = np.array([0.2, 0.4, 1.2, 2.5, 4.5, 7.0])  # Observed output values

# # Define parameter variables (unknowns to estimate)
# model.theta1 = Var(initialize=1.0)  # First parameter
# model.theta2 = Var(initialize=1.0)  # Second parameter

# # Define an external function
# def blackbox1(a):
#     return a

# bb1 = ExternalFunction(blackbox1)

# def blackbox2(a):
#     return a

# bb2 = ExternalFunction(blackbox2)

# # Define the objective function (Least Squares Error)
# def objective_rule(m):
#     return sum((y_obs[i] - (bb1(m.theta1) * x_data[i]**2 + bb2(m.theta2) * x_data[i]))**2 for i in range(len(x_data)))

# model.obj = Objective(rule=objective_rule, sense=minimize)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=300, trust_radius=10, sample_radius=1, algorithm_type=1, reduced_model_type=4, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb1,bb2]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_29_Parameter_Estimation_NLP_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(m, eflist)
#         m.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 30 ###############################################
# # Model 30 (Optimal_Design_of_a_Pressure_Vessel), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Define the decision variables
# model.d = Var(initialize=10.0, bounds=(5.0, 20.0))  # Diameter of the vessel (m)
# model.t = Var(initialize=0.5, bounds=(0.1, 5.0))  # Thickness of the shell (m)

# # Material properties and other constants
# p = 2e7  # Internal pressure (Pa)
# sigma_max = 1.5e8  # Maximum allowable stress (Pa)
# E = 2e11  # Young's modulus (Pa)
# rho = 7800  # Density of steel (kg/m³)
# l = 3.0  # Length of the pressure vessel (m)

# # Define an external function
# def blackbox(a,b):
#     return a*b

# bb = ExternalFunction(blackbox)

# # Objective function: Minimize the total cost of the vessel
# def objective_rule(m):
#     cost_material = rho * m.t * m.d**2 * l  # Cost due to material volume (proportional to volume)
#     cost_manufacture = 500 * (m.d**2)  # Fixed cost (related to diameter)
#     return cost_material + cost_manufacture
# model.obj = Objective(rule=objective_rule, sense=minimize)

# # Constraints
# # Strength constraint: Ensure the stress inside the vessel does not exceed the maximum allowable stress
# def strength_constraint_rule(m):
#     return (4 * p * m.d**2) / (3 * m.t) <= sigma_max
# model.strength_constraint = Constraint(rule=strength_constraint_rule)

# # Geometric constraints: Ensure that the vessel diameter and thickness meet certain limits
# def geometric_constraint_rule(m):
#     return bb(m.d,m.t) <= 50.0  # Maximum product of diameter and thickness
# model.geometric_constraint = Constraint(rule=geometric_constraint_rule)

# # Initialize the TrustRegionSolver with necessary configurations
# solver = TrustRegionSolver(solver ='ipopt', max_it=300, trust_radius=10, sample_radius=1, algorithm_type=3, reduced_model_type=4, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_30_Optimal_Design_of_a_Pressure_Vessel_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 31 ###############################################
# # Model 31 (Portfolio Selection), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Parameters
# model.theta = Param(default=5)

# # Variables
# model.x = Var(range(2), bounds={ 
#     0: (0, 5), 
#     1: (0, 5)},
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return (2*a**2 + b**2)

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(
#     expr= 20*model.x[0] + 16*model.x[1] - model.theta*(bb(model.x[0],model.x[1]) + (model.x[0] + model.x[1])**2), sense=maximize)

# # Constraints
# model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 5)

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, algorithm_type=0, reduced_model_type=0, gamma_e=2.5, delta_min=1e-3, ep_delta=1e-2)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_31_Portfolio_Selection_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
    
# ############################## Model 32 ###############################################
# # Model 32 (P1), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Variables
# model.x = Var(range(2),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return (a - 1)**2 + b**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=bb(model.x[0], model.x[1]), sense=maximize)

# # Constraints
# model.con1 = Constraint(expr=model.x[0]**2 + 6*model.x[1] - 36 <= 0)
# model.con2 = Constraint(expr=-4*model.x[0] + model.x[1]**2 - 2*model.x[1] <= 0)

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_32_P1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
    
# ############################## Model 33 ###############################################
# # Model 33 (Custom-molder Example), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Variables
# model.x = Var(range(2),
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a):
#     return 60*a - 5*a**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=bb(model.x[0]) + 80*model.x[1] - 4*model.x[1]**2, sense=maximize)

# # Constraints
# model.con1 = Constraint(expr=6*model.x[0] + 5*model.x[1] <= 60)
# model.con2 = Constraint(expr=10*model.x[0] + 12*model.x[1] <= 150)

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=30000, trust_radius=10, sample_radius=1, algorithm_type=0, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_33_Custom-molder_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
    
# ############################## Model 34 ###############################################
# # Model 34 (P2), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Variables
# model.x = Var(range(2),
#     bounds={ 
#     0: (0, None), 
#     1: (0, 1.8)},
#     initialize={
#     0: 0,
#     1: 0})

# # Define an external function
# def blackbox(a,b):
#     return a**2 + b**2

# bb = ExternalFunction(blackbox)

# # Define the objective
# model.obj = Objective(expr=2*model.x[0] - model.x[0]**2 + model.x[1], sense=maximize)

# # Constraints
# model.con1 = Constraint(expr=bb(model.x[0],model.x[1]) <= 4)

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=250, algorithm_type=4, reduced_model_type=0, gamma_e=10)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_34_P2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 35 ###############################################
# # Model 35 (Pressure Vessel Design), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# a = 0.0625

# # Variables
# model.x1 = Var(bounds=(a, 99*a), initialize=0.7780270626316168)   
# model.x2 = Var(bounds=(a, 99*a), initialize=15*a)   
# model.x3 = Var(bounds=(10, 200), initialize=40.312283554875876)   
# model.x4 = Var(bounds=(10, 200), initialize=200) 

# # Define an external function

# def blackbox(a,b,c,d):
#     return 0.6224*(a*c*d) + 1.7781*b*(c**2) + 3.1661*(a**2)*d + 19.84*(a**2)*c

# bb = ExternalFunction(blackbox)

# # def blackbox1(a,b,c):
# #     return 0.6224*a*b*c

# # bb1 = ExternalFunction(blackbox1)

# # def blackbox2(a,b):
# #     return 1.7781*a*(b**2)

# # bb2 = ExternalFunction(blackbox2)

# # def blackbox3(a,b):
# #     return 3.1661*(a**2)*b

# # bb3 = ExternalFunction(blackbox3)

# # Define the objective
# model.obj = Objective(expr=bb(model.x1,model.x2,model.x3,model.x4), sense=minimize)
# # model.obj = Objective(expr=0.6224*(model.x1*model.x3*model.x4) + 1.7781*model.x2*(model.x3**2) + 3.1661*(model.x1**2)*model.x4 + 19.84*(model.x1**2)*model.x3, sense=minimize)

# # Constraints
# model.c1 = Constraint(expr= - model.x1 + 0.0193*model.x3 <= 0)
# model.c2 = Constraint(expr= - model.x2 + 0.0095*model.x3 <= 0)
# model.c3 = Constraint(expr= - (22/7)*(model.x3**2)*model.x4 - (4/3)*(22/7)*(model.x3**3) + 1296000 <= 0)
# model.c4 = Constraint(expr= model.x4 - 240 <= 0)

# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4, , delta_min=1, ep_delta=2
# solver = TrustRegionSolver(solver ='ipopt', max_it=15000, trust_radius=1, sample_radius=0.1, algorithm_type=4, reduced_model_type=0, gamma_e=10)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_35_Pressure_Vessel_Design_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 36 ###############################################
# # Model 36 (Liquid_Liquid_Extraction_Column), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # PARAMETERS
# model.m = Param(default=1.5)       # distribution coefficient

# # VARIABLES
# model.vx = Var(bounds=(0.05, 0.25), initialize=0.08)  # vx lower bound >0 to avoid division by zero # Raffinate velocity
# model.vy = Var(bounds=(0.05, 0.30), initialize=0.10)  # vy lower bound >0 # Extract velocity
# model.F = Var(bounds=(0, None), initialize=1.2) # Extraction factor F
# model.Y0 = Var(bounds=(0, None), initialize=1.3862070832433895) # Extract outlet concentration
# # model.Nox = Var(bounds=(0, None), initialize=4.559179311599174) # Experimental number of transfer units

# # Define an external function
# # number of transfer units # A black-box because it is calculated from experimental data
# def blackbox(a,b):
#     return 4.81*((a/b)**0.24)

# bb = ExternalFunction(blackbox)

# # OBJECTIVE: maximize vy * Y0
# model.obj = Objective(expr = model.vy * model.Y0, sense=maximize)

# # CONSTRAINTS
# # Define F = m * vx / vy
# model.F_definition = Constraint(expr = model.F == model.m * model.vx / model.vy)


# # Define Y0 using nonlinear constraint
# model.Y0_definition = Constraint(
#     expr = model.Y0 * (1 - model.F * exp(bb(model.vx, model.vy) * (1 - model.F))) ==
#           model.F * (1 - exp(bb(model.vx, model.vy) * (1 - model.F)))
# )

# # (Optional) Constraints
# # Example flooding constraint (you can modify or add based on detailed flooding curve)
# model.flooding = Constraint(expr = model.vx + model.vy <= 0.20)


# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=100, algorithm_type=4, reduced_model_type=0, gamma_e=8.35) # gamma_e=8.35,9.99

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_36_Liquid_Liquid_Extraction_Column_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
    
    
# ############################## Model 37 ###############################################
# # Model 37 (Himmelblau’s Problem), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # === VARIABLES ===
# model.x1 = Var(bounds=(78, 102), initialize=100)   
# model.x2 = Var(bounds=(33, 45), initialize=40)   
# model.x3 = Var(bounds=(27, 45), initialize=40)   
# model.x4 = Var(bounds=(27, 45), initialize=35)   
# model.x5 = Var(bounds=(27, 45), initialize=30)

# model.g1 = Var(bounds=(0, 92), initialize=30)
# model.g2 = Var(bounds=(90, 110), initialize=100)
# model.g3 = Var(bounds=(20, 25), initialize=20)

# # Define an external function
# def blackbox1(a):
#     return a**2

# bb1 = ExternalFunction(blackbox1)

# # Define an external function
# def blackbox2(a,b):
#     return a*b

# bb2 = ExternalFunction(blackbox2)

# # === OBJECTIVE ===
# model.obj = Objective(expr=5.3578547*bb1(model.x3) + 0.8356891*model.x1*model.x5 + 37.2932239*model.x1 - 40792.141, sense=minimize)

# # === CONSTRAINTS ===
# model.c1 = Constraint(expr= model.g1 == 85.334407 + 0.0056858*bb2(model.x2,model.x5) + 0.00026*model.x1*model.x4 - 0.0022053*model.x3*model.x5)
# model.c2 = Constraint(expr= model.g2 == 80.51249 + 0.0071317*bb2(model.x2,model.x5) + 0.0029955*model.x1*model.x2 - 0.0021813*(model.x3**2))
# model.c3 = Constraint(expr= model.g3 == 9.300961 + 0.0047026*model.x3*model.x5 + 0.0012547*model.x1*model.x3 - 0.0019085*model.x3*model.x4)



# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=10000, sample_radius=1000, algorithm_type=0, reduced_model_type=4, gamma_e=15) # gamma_e=8.35,9.99, , ep_s=1e-2, ep_i=1e-3
# # for A3,S3, GP(2)
# # A5,S4 and S3 = trust_radius=10, sample_radius=2.4, ep_s=1e0
# # 2/1 ratio in tr radius and sampling radius
# # Define an external function list (eflist) as needed
# eflist = [bb1, bb2]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_37_Himmelblau_Problem_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__

# ############################## Model 38 ###############################################
# # Model 38 (Spring Design), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Decision variables
# model.x1 = Var(bounds=(0.05, 2.0), initialize=1.0)     # Wire diameter
# model.x2 = Var(bounds=(0.25, 1.3), initialize=0.5)     # Mean coil diameter
# model.x3 = Var(bounds=(2.0, 15.0), initialize=5.0)     # Number of active coils

# # Define an external function

# def blackbox(a,b,c):
#     return (c + 2) * b * a**2

# bb = ExternalFunction(blackbox)

# # Objective function: Minimize weight (x3 + 2)x2*x1^2
# model.obj = Objective(expr=bb(model.x1,model.x2,model.x3), sense=minimize)

# # Constraints
# model.g1 = Constraint(expr=1 - (model.x2**3 * model.x3) / (71785 * model.x1**4) <= 0)

# model.g2 = Constraint(
#     expr=((4 * model.x2**2 - model.x1 * model.x2) / (12566 * ((model.x2 * model.x1**3) - (model.x1**4)))) + (1 / (5108 * model.x1**2)) - 1 <= 0)

# model.g3 = Constraint(expr=1 - (140.45 * model.x1 / (model.x2**2 * model.x3)) <= 0)

# model.g4 = Constraint(expr=((model.x1 + model.x2)/1.5) - 1 <= 0)


# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-15
# solver = TrustRegionSolver(solver ='ipopt', max_it=15000, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_38_Spring_Design_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
    
    
# ############################## Model 39 ###############################################
# # Model 39 (Welded Beam), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# model = ConcreteModel()

# # Constants
# model.c1 = Param(default=0.10471)
# model.c2 = Param(default=0.04811)
# model.P = Param(default=6000.0)          # Load
# model.L = Param(default=14.0)            # Length
# model.E = Param(default=30e6)            # Young's modulus
# model.G = Param(default=12e6)            # Shear modulus
# model.tau_max = Param(default=13600.0)
# model.sigma_max = Param(default=30000.0)
# model.delta_max = Param(default=0.25)

# # Design Variables with bounds
# model.x1 = Var(bounds=(0.125, 5), initialize=1.0)    # h (thickness of weld)
# model.x2 = Var(bounds=(0.1, 10), initialize=5.0)     # l (length of weld)
# model.x3 = Var(bounds=(0.1, 10), initialize=5.0)     # t (height of the beam)
# model.x4 = Var(bounds=(0.1, 5), initialize=1.0)      # b (width of the beam)

# # Define an external function

# def blackbox(a,b,c,d):
#     return (1 + 0.10471) * a**2 * b + 0.04811 * c * d * (14 + b)

# bb = ExternalFunction(blackbox)

# # Objective Function

# model.obj = Objective(expr=bb(model.x1,model.x2,model.x3,model.x4), sense=minimize)

# # Shear stress τ
# def tau(model):
#     R = sqrt((model.x2**2) / 4 + ((model.x1 + model.x3)/2)**2)
#     M = model.P * (model.L + model.x2 / 2)
#     J = 2 * (sqrt(2) * model.x1 * model.x2) * ((model.x2**2)/12 + ((model.x1 + model.x3)/2)**2)
#     t1 = model.P / (sqrt(2) * model.x1 * model.x2)
#     t2 = M * R / J
#     return sqrt(t1**2 + 2*t1*t2*model.x2/(2*R) + t2**2)
# model.tau_expr = Expression(rule=tau)

# # Bending stress σ
# def sigma(model):
#     return (6 * model.P * model.L) / (model.x4 * model.x3**2)
# model.sigma_expr = Expression(rule=sigma)

# # Deflection δ
# def delta(model):
#     return (4 * model.P * model.L**3) / (model.E * model.x3**3 * model.x4)
# model.delta_expr = Expression(rule=delta)

# # Buckling load P_c
# def Pc(model):
#     return ((4.013 * model.E * sqrt(model.x3**2 * model.x4**6 / 36)) / (model.L**2)) * (1 - (model.x3 / (2*model.L)) * sqrt(model.E / (4 * model.G)))
# model.pc_expr = Expression(rule=Pc)

# # Constraints
# model.g1 = Constraint(expr = model.tau_expr <= model.tau_max)
# model.g2 = Constraint(expr = model.sigma_expr <= model.sigma_max)
# model.g3 = Constraint(expr = model.x1 - model.x4 <= 0)
# model.g4 = Constraint(expr = model.c1 * model.x1**2 * model.x2 + model.c2 * model.x3 * model.x4 * (model.L + model.x2) - 5 <= 0)
# model.g5 = Constraint(expr = model.delta_expr <= model.delta_max)
# model.g6 = Constraint(expr = model.P - model.pc_expr <= 0)


# # Initialize the TrustRegionSolver with necessary configurations , , delta_min=1e-2, ep_delta=1e-1
# solver = TrustRegionSolver(solver ='ipopt', max_it=15000, algorithm_type=2, reduced_model_type=3, gamma_e=10)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_39_Welded_Beam_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

# try:
#     with open(filename, 'w') as f:
#         tee = Tee(f)
#         sys.stdout = tee  # Redirect stdout to file and console

#         # Solve the model and display results
#         solver.solve(model, eflist)
#         model.display()

# finally:
#     # Restore original stdout safely
#     sys.stdout = sys.__stdout__
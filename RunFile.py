# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:36:10 2024

@author: gh00616
"""

import sys, os
import logging

import numpy as np
import pandas as pd
import math

from pyomo.environ import *
from pyomo.opt import *
from pyomo.gdp import *

from pyomo.common.config import ConfigBlock, ConfigValue, PositiveFloat, PositiveInt, NonNegativeFloat, In

from pyomo.common.tempfiles import TempfileManager

import locale
import csv  # Import the csv module

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
        default=1e-4,
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
    
    CONFIG.declare('kappa f', ConfigValue(
        default=0.25,
        domain=PositiveFloat,
        description='funnel‑shrink factor after f‑type',
    ))
    
    CONFIG.declare('kappa r', ConfigValue(
        default=1.1,
        domain=PositiveFloat,
        description='funnel expand factor for relax theta step',
    ))

    CONFIG.declare('theta min', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
    ))
    
    CONFIG.declare('phi min', ConfigValue(
        default=1e-8,
        domain=PositiveFloat,
        description='hard floor on funnel width',
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
    
    CONFIG.declare('alpha', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='curvature exponent ( (theta)^alpha )',
    ))
    
    CONFIG.declare('beta', ConfigValue(
        default=0.8,
        domain=PositiveFloat,
        description='extra shrink required for theta‑type',
    ))
    
    CONFIG.declare('mu s', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='switching coefficient δ',
    ))
    
    CONFIG.declare('eta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='Armijo coefficient for f‑type',
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
    
    # Default globalization strategy
    CONFIG.declare('globalization strategy', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = Filter, 1 = Funnel',
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
    



# ######################################## Model A1 ###############################################
# # Model A1 (Colville Function, 4 surrogates), it is not solved using default values of the tuning parameters, see notes for the values used
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

# solver = TrustRegionSolver(solver ='ipopt', max_it=25000, trust_radius=1000, sample_radius=100, algorithm_type=0, reduced_model_type=3, globalization_strategy=1, gamma_e=10, criticality_check=0.1) #, ep_i=1e-4, ep_s=1
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
# filename = f"Model_A1_Colville_Function_2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}_GS{solver.config['globalization strategy']}.txt"

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

# ############################## Model A2 ###############################################
# # Model A2 (Himmelblau’s Problem), it is solved using default values of tuning parameter, see notes for the default values
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
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=10000, sample_radius=1000, algorithm_type=0, reduced_model_type=9, globalization_strategy=1, gamma_e=15) # gamma_e=8.35,9.99, , ep_s=1e-2, ep_i=1e-3
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
# filename = f"Model_A2_Himmelblau_Problem_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}_GS{solver.config['globalization strategy']}.txt"

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


# ############################## Model A3 ###############################################
# # Model A3 (Loeppky Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# m = ConcreteModel()

# # Indices
# m.I = RangeSet(1, 7)

# # === VARIABLES ===
# m.x = Var(m.I, bounds=(0, 1), initialize=0.5)

# # Define an external function
# def blackbox1(a,b,c):
#     return 3*a*b + 2.2*a*c

# bb1 = ExternalFunction(blackbox1)


# # === OBJECTIVE ===
# m.obj = Objective(
#     expr= 6*m.x[1] + 4*m.x[2] + 5.5*m.x[3] + bb1(m.x[1],m.x[2],m.x[3]) + 1.4*m.x[2]*m.x[3] + m.x[4] + 0.5*m.x[5] + 0.2*m.x[6] + 0.1*m.x[7],
#     sense=minimize,
# )



# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=4, globalization_strategy=0, gamma_e=15) # gamma_e=8.35,9.99, , ep_s=1e-2, ep_i=1e-3
# # for A3,S3, GP(2)
# # A5,S4 and S3 = trust_radius=10, sample_radius=2.4, ep_s=1e0
# # 2/1 ratio in tr radius and sampling radius
# # Define an external function list (eflist) as needed
# eflist = [bb1]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_A3_Loeppky_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}_GS{solver.config['globalization strategy']}.txt"

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


# ############################## Model A4 ###############################################
# # Model A4 (Wing Weight Function), it is solved using default values of tuning parameter, see notes for the default values
# # max_it=20, trust_radius=1, sample_radius=0.1, reduced_model_type=1, gamma_e=2.5, criticality_check=0.1, delta_min = 1.3e-3, ep_delta=1.5e-3, ep_i=6.7e-4
# # Define the optimization model
# m = ConcreteModel()

# # wing area (ft²)
# m.Sw  = Var(bounds=(150.0, 200.0),   initialize=175.0)
# # fuel weight (lb)
# m.Wfw = Var(bounds=(220.0, 300.0),   initialize=260.0)
# # aspect ratio (—)
# m.A   = Var(bounds=(6.0, 10.0),      initialize=8.0)
# # quarter‑chord sweep Λ (deg; can take ±)
# m.Lam = Var(bounds=(-10.0, 10.0),    initialize=0.0)
# # dynamic pressure (lb/ft²)
# m.q   = Var(bounds=(16.0, 45.0),     initialize=30.0)
# # taper ratio (—)
# m.lam = Var(bounds=(0.5, 1.0),       initialize=0.75)
# # thickness‑to‑chord ratio (—)
# m.tc  = Var(bounds=(0.08, 0.18),     initialize=0.13)
# # ultimate load factor (—)
# m.Nz  = Var(bounds=(2.5, 6.0),       initialize=4.0)
# # design gross weight (lb)
# m.Wdg = Var(bounds=(1700.0, 2500.0), initialize=2100.0)
# # paint weight (lb/ft²)
# m.Wp  = Var(bounds=(0.025, 0.08),    initialize=0.05)

# # Define an external function
# def blackbox1(a,b):
#     return a*b

# bb1 = ExternalFunction(blackbox1)


# # === OBJECTIVE ===
# m.obj = Objective(
#     expr= 0.036
#     * m.Sw**0.758
#     * m.Wfw**0.0035
#     * (m.A / (cos(m.Lam * (22/7) / 180.0)**2))**0.6
#     * m.q**0.006
#     * m.lam**0.04
#     * ((100.0 * m.tc / cos(m.Lam * (22/7) / 180.0))**-0.3)
#     * (m.Nz * m.Wdg)**0.49
#     + bb1(m.Sw,m.Wp),
#     sense=minimize,
# )



# # Initialize the TrustRegionSolver with necessary configurations , delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4
# solver = TrustRegionSolver(solver ='ipopt', max_it=10000, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=4, globalization_strategy=0, gamma_e=15) # gamma_e=8.35,9.99, , ep_s=1e-2, ep_i=1e-3
# # for A3,S3, GP(2)
# # A5,S4 and S3 = trust_radius=10, sample_radius=2.4, ep_s=1e0
# # 2/1 ratio in tr radius and sampling radius
# # Define an external function list (eflist) as needed
# eflist = [bb1]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_A4_Wing_Weight_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}_GS{solver.config['globalization strategy']}.txt"

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


# ############################## Model A5 ###############################################
# # Model A5 (Welded Beam), it is solved using default values of tuning parameter, see notes for the default values
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
# solver = TrustRegionSolver(solver ='ipopt', max_it=50, algorithm_type=0, reduced_model_type=4, globalization_strategy=0, gamma_e=10)

# # Define an external function list (eflist) as needed
# eflist = [bb]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_A5_Welded_Beam_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}_GS{solver.config['globalization strategy']}.txt"

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

# ######################################## Model A6 ###############################################
# # Model A6 (Williams Otto), it is not solved using default values of the tuning parameters, see notes for the values used
# # max_it=2000, trust_radius=10, sample_radius=1, reduced_model_type=1, gamma_e=5, criticality_check=1.1
# # Define the optimization model

# m = ConcreteModel()

# # Parameters

# m.p = Param(default=50)

# # Variables

# m.V = Var(bounds=(0.03, 0.1), initialize=0.0952213880274209)
# m.T = Var(bounds=(5.8, 6.8), initialize=5.8)
# m.Fp = Var(bounds=(0, 4.763), initialize=1.0220471993316949)
# m.Fpurge = Var(bounds=(0, None), initialize=7.092944066046297)
# m.Fg = Var(bounds=(0, None), initialize=0.2546416177080731)
# m.Feff_sum = Var(initialize=53.73152359591559)
# m.Fa = Var(bounds=(1, None), initialize=3.032853228628288)
# m.Fb = Var(bounds=(1, None), initialize=5.3367796544577795)
# m.n = Var(bounds=(0, 1), initialize=0.1352200249213798)

# # Component-wise variables (index: 0=A, 1=B, 2=C, 3=E, 4=P, 5=G)
# m.Feff = Var(range(6))
# m.FR = Var(range(6))
# m.x = Var(range(6), bounds=(0, 1))

# # Initialization values
# Feff_vals = [9.921446738565438, 16.72714547405174, 3.294404672195415,
#              20.465307176421106, 3.0685779169738057, 0.2546416177080731]

# FR_vals = [8.579868463320505, 14.465300446186964, 2.8489351903200504,
#            17.697987830001807, 2.6536447345674206, 0.22020897181556767]

# x_vals = [0.1846485279885039, 0.3113097182921363, 0.061312325646500375,
#           0.38088082761860875, 0.05710945291726458, 0.004739147536985909]

# # Apply initialization to indexed variables
# for i in range(6):
#     m.Feff[i] = Feff_vals[i]
#     m.FR[i] = FR_vals[i]
#     m.x[i] = x_vals[i]

# # Black-box

# def blackbox1(a,b,c,d):
    
#     return (5.9755*(10**9) * exp(-120/(a)) * b * c * d * 50)

# def blackbox2(a,b,c,d):      
#     return (2.5962*(10**12) * exp(-150/(a)) * b * c * d * 50)

# def blackbox3(a,b,c,d):
#     return (9.6283*(10**15) * exp(-200/(a)) * b * c * d * 50)

# bb1 = ExternalFunction(blackbox1)
# bb2 = ExternalFunction(blackbox2)
# bb3 = ExternalFunction(blackbox3)


# # Objective
# m.obj = Objective(expr = (100 * ((2207 * (m.Fp)) + (50 * (m.Fpurge)) - (168 * (m.Fa)) -(252 * (m.Fb)) - (2.22 * (m.Feff_sum)) - (84 * (m.Fg)) - (60 * (m.V) * m.p)) / (600 * (m.V) * m.p)), sense=maximize)

# # Constraints
# # m.c1 = Constraint(expr = m.r[0] == bb1(m.T, m.x[0], m.x[1], m.V))
# # m.c2 = Constraint(expr = m.r[1] == bb2(m.T, m.x[1], m.x[2], m.V))
# # m.c3 = Constraint(expr = m.r[2] == bb3(m.T, m.x[4], m.x[2], m.V))

# m.c4 = Constraint(expr = m.Feff[0] == m.Fa + (m.FR[0]) - bb1(m.T, m.x[0], m.x[1], m.V))
# m.c5 = Constraint(expr = m.Feff[1] == m.Fb + (m.FR[1]) - (bb1(m.T, m.x[0], m.x[1], m.V) + bb2(m.T, m.x[1], m.x[2], m.V)))
# m.c6 = Constraint(expr = m.Feff[2] == (m.FR[2]) + (2 * bb1(m.T, m.x[0], m.x[1], m.V)) - (2 * bb2(m.T, m.x[1], m.x[2], m.V)) - bb3(m.T, m.x[4], m.x[2], m.V))
# m.c7 = Constraint(expr = m.Feff[3] == (m.FR[3]) + (2 * bb2(m.T, m.x[1], m.x[2], m.V)))
# m.c8 = Constraint(expr = m.Feff[4] == (0.1 * (m.FR[3])) + bb2(m.T, m.x[1], m.x[2], m.V) - (0.5 * bb3(m.T, m.x[4], m.x[2], m.V)))
# m.c9 = Constraint(expr = m.Feff[5] == 1.5 * bb3(m.T, m.x[4], m.x[2], m.V))
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

# solver = TrustRegionSolver(solver ='ipopt', max_it=5000, trust_radius=10, sample_radius=1, algorithm_type=3, reduced_model_type=4, globalization_strategy=0, gamma_e=10, delta_min=1e-2, ep_delta=1e-1)
# # # max_it=2500, trust_radius=1002, sample_radius=10.02, reduced_model_type=2, gamma_e=10, criticality_check=0.2, delta_min=1e-1, ep_delta=1, ep_s=1e-1, ep_delta=1e-2

# # Define an external function list (eflist) as needed
# eflist = [bb1, bb2, bb3]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'knitro'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_A6_Williams_Otto_1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}_GS{solver.config['globalization strategy']}.txt"

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


# ############################## Model Biomass-based Hydrogen Production (BHP) ###############################################
# # Model BHP

# from openpyxl import load_workbook

# # ----------------------------------------------------------------------
# # 1.  READ ALL WORKBOOK DATA AT ONCE  ###########################
# # ----------------------------------------------------------------------
# FILE = 'B-HYPSYS User Interface (Data).xlsx'
# if not os.path.exists(FILE):
#     raise FileNotFoundError(FILE)


# # ----------------------------------------------------------------------
# # 2.  PYOMO MODEL & SETS ################################################
# # ----------------------------------------------------------------------
# m = ConcreteModel()

# # Biomass data ----------------------------------------------------

# m.biomass_database = pd.read_excel(FILE, sheet_name='BIOMASS_DATABASE',
#                         header=50, nrows=17, index_col=0).to_dict()

# wb = load_workbook(FILE, data_only=True)
# oper = wb['OPERATING_CONDITIONS']
# heatcap = wb['HEAT_CAPACITY']
# gibbs = wb['HEAT_GIBBS']


# # Other small sets -----------------------------------------------------
# m.Components = Set(initialize=['H2','CO','CO2','H2O','CH4'])

# # ----------------------------------------------------------------------
# # 3.  PARAMETERS (everything pulled from the Pandas objects)  ###########
# # ----------------------------------------------------------------------
# BIOMASS   = oper['D10'].value
# GA        = oper['D9'].value
# AF_ratio  = oper['D7'].value;  SB_ratio = oper['D8'].value
# Max_Feed  = oper['D11'].value
# Temp_Fix  = oper['D5'].value;  Temp = oper['D6'].value if Temp_Fix == 'Yes' else None
# Inlet_temp   = oper['D2'].value + 273.15
# Drying_fix = oper['D3'].value;  Drying_temp = oper['D4'].value + 273.15 if Drying_fix=='Yes' else None

# # -------- demand & feed  ------------------------------------------
# m.feed_HR = Param(default=2.5) #kg/s

# m.demand_H2 = Param(default=320) #kg/h

# ## Are these variables?

# m.Max_Feed = Param(initialize=oper['D11'].value)
# Max_Demand = oper['D12'].value

# # ####1####

# # m.biomass_feed = Var(bounds=(1e-5, m.feed_HR), initialize=2.4260361106658928)
    
# # m.dry_biomass_feed = Var(domain=NonNegativeReals, initialize=2.1931366440419673)

# # ####2####

# # m.biomass_feed = Var(bounds=(1e-5, m.feed_HR), initialize=2.4254955352642953)
    
# # m.dry_biomass_feed = Var(domain=NonNegativeReals, initialize=2.1926479638789234)

# # ####3####

# # m.biomass_feed = Var(bounds=(1e-5, m.feed_HR), initialize=2.453215147359572)
    
# # m.dry_biomass_feed = Var(domain=NonNegativeReals, initialize=2.217706493213053)

# ####4####

# m.biomass_feed = Var(bounds=(1e-5, m.feed_HR), initialize=2.4260361106658928)
    
# m.dry_biomass_feed = Var(domain=NonNegativeReals, initialize=2.1931366440419673)


# # ---- quick helpers from tables ---------------------------------------
# Cp_air  = oper['I6'].value;  Cp_water = oper['I5'].value; Cp_vap = oper['I7'].value
# H_vap   = oper['I8'].value
# R       = gibbs['K8'].value

# # Reaction rate constants
# delA_m = heatcap['N8'].value  # Coefficient for methanation reaction
# delB_m = heatcap['N9'].value
# delC_m = heatcap['N10'].value
# delD_m = heatcap['N11'].value
# J_m = gibbs['K14'].value
# I_m = gibbs['K16'].value

# delA_wg = heatcap['P8'].value  # Coefficient for water-gas shift reaction
# delB_wg = heatcap['P9'].value
# delC_wg = heatcap['P10'].value
# delD_wg = heatcap['P11'].value
# J_wg = gibbs['K15'].value
# I_wg = gibbs['K17'].value

# # molar masses ---------------------------------------------------------
# mm_val = dict(H2=2.016, CO=28.01, CO2=44.01, H2O=18.015, CH4=16.04)
# m.MM_gases = Param(m.Components, initialize=mm_val)

# # WGS
# m.th_eff_CHP_engine = Param(default=0.8)  # Thermal efficiency
# m.elec_eff_CHP_engine = Param(default=0.8)  # Electrical efficiency
# m.syngas_cooler_eff = Param(initialize=0.8)  # Efficiency of syngas-to-WGS heat exchanger
# m.Cp_syngas = Param(default=1.1) # kJ/kg·K

# # Heat recovery from WGS
# m.wgs_heat_recovery_eff = Param(default=0.7)  # Efficiency of heat recovery from WGS
# m.T_wgs = Param(initialize=673.15)    # Target WGS inlet temp (400°C = 673.15 K)

# # CEPCI and cost parameters
# m.CEPCI_current = Param(initialize=800)  # Current CEPCI value (e.g., 2023)
# m.CEPCI_base = Param(initialize=607.5)  # Base CEPCI value (e.g., 2019)
# m.scaling_exponent = Param(initialize=0.67)  # Scaling exponent for cost estimation

# # Base costs for equipment (example values)
# m.base_cost_gasifier = Param(initialize=2.95e6)  # Base cost for gasifier (USD)
# m.base_cost_gas_cleaner = Param(initialize=1.83e6) # Base cost for gas cleaner (USD)
# m.base_cost_chp = Param(initialize=93300)  # Base cost for CHP engine (USD)
# m.base_cost_wgs = Param(initialize=1.94e6)  # Base cost for WGS reactor (USD)
# m.base_cost_separator = Param(initialize=1e5)  # Base cost for separator/PSA (USD)

# # Economic parameters
# m.labor_cost_factor = Param(initialize=0.12)  # 12% of biomass+steam costs
# m.maintenance_factor = Param(initialize=0.03)  # 3% of CAPEX
# m.insurance_tax_factor = Param(initialize= 0.015)  # 1.5% of CAPEX
# m.waste_factor = Param(initialize=0.07)  # 7% of biomass cost
# m.catalyst_unit_cost = Param(initialize=50)  # $/kg
# m.catalyst_replacement_rate = Param(initialize=0.02)  # 2% of capacity/year

# # Financial constants
# m.hydrogen_price = Param(initialize=3)  # Hydrogen price (USD/kg), 2.3 worked
# m.CO_price = Param(initialize=7.02)  # Hydrogen price (USD/kg)
# m.CO2_price = Param(initialize=0.15)  # CO2 (food-grade and industrial use) price (USD/kg)
# m.electricity_price = Param(initialize=0.16583) # Feed-in-tarrif industrial power price (USD/KWh)
# m.grid_electricity_price = Param(initialize=0.15) # industrial power price (USD/KWh)
# m.biomass_cost = Param(initialize=0.1)  # Biomass cost (USD/kg)
# m.steam_cost = Param(initialize=0.02)  # Steam cost (USD/kg)


# m.investment = Param(initialize=0)

# # NPV-related parameters
# m.project_life = Param(initialize=20)  # Project life in years
# m.discount_rate_project = Param(initialize=0.10)  # Discount rate (10%)
# m.discount_rate_h2 = Param(initialize=0.03)

# # ----------------------------------------------------------------------
# # 4.  VARIABLES #########################################################
# # ----------------------------------------------------------------------


# # ####1####

# # # === Reactor operating variables ===
# # m.T_opt = Var(bounds=(873.15, 1143.15), initialize=1143.1500099999837)  # K
# # m.k1 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.02518130161661987)
# # m.k2 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.8766772277369185)

# # # === Syngas composition (mol/s) ===
# # m.x = Var(m.Components, bounds=(1e-4, 5))
# # m.x['CH4'].set_value(0.022991912653426046)
# # m.x['CO'].set_value(0.9649033993560977)
# # m.x['CO2'].set_value(0.0121046879904762)
# # m.x['H2'].set_value(0.9555391004215552)
# # m.x['H2O'].set_value(0.013673462393232359)

# # m.m = Var(domain=PositiveReals, bounds=(0, 150), initialize=0.04684646083102462)  # mol/s
# # m.x_total = Var(bounds=(1e-5, 500), initialize=1.9692125628147878)  # mol/s

# # # === Syngas composition (kg/s) ===
# # m.x_kg = Var(m.Components, bounds=(1e-5, None))
# # m.x_kg['CH4'].set_value(0.01701195698890788)
# # m.x_kg['CO'].set_value(1.2467281237428638)
# # m.x_kg['CO2'].set_value(0.024574222113462683)
# # m.x_kg['H2'].set_value(0.0888615331422378)
# # m.x_kg['H2O'].set_value(0.011362857967228723)

# # m.x_kg_total = Var(bounds=(1e-5, None), initialize=1.3885386939547009)  # kg/s
# # m.LHV_sng = Var(bounds=(1e-5, None), initialize=17.360651151651144)  # MJ/kg
# # m.CGE = Var(bounds=(61, 100), initialize=96.5920740949808)  # %

# # # === Energy demands (kW) ===
# # m.P_drying = Var(domain=PositiveReals, initialize=28.989835048759886)
# # m.P_biomass_heat = Var(domain=PositiveReals, initialize=3100.3018440893043)
# # m.P_air_heat = Var(bounds=(-1e-10,None), initialize=0.0)
# # m.P_water_heat = Var(bounds=(-1e-10,None), initialize=0.0)
# # m.P_evap = Var(domain=PositiveReals, initialize=526.3527945700722)
# # m.P_heat_vap = Var(domain=PositiveReals, initialize=331.76529451441536)
# # m.P_heat_water_total = Var(domain=PositiveReals, initialize=858.1180890844876)

# # m.gasifier_elec_power_req = Var(domain=PositiveReals, initialize=20.225412997406295)
# # m.grid_power_req = Var(domain=Reals, initialize=-10689.654852419744)
# # m.gasifier_th_power_req = Var(domain=PositiveReals, initialize=3987.4097682225515)
# # m.Th_power_total = Var(domain=PositiveReals, initialize=4984.262210278189)
# # m.Elec_power_total = Var(domain=PositiveReals, initialize=13387.350331771437)

# # # === Syngas split ===
# # m.sng_flow_rate_CHP_engine = Var(domain=PositiveReals, initialize=0.8465863351123133)
# # m.sng_flow_rate_WGS = Var(bounds=(1e-4, 100), initialize=0.5419523588423875)

# # # === WGS outlet (mol/s) ===
# # m.x_WGS = Var(m.Components, bounds=(1e-4, None))
# # m.x_WGS['CH4'].set_value(0)
# # m.x_WGS['CO'].set_value(9.385867948178515)
# # m.x_WGS['CO2'].set_value(0.17050227008212188)
# # m.x_WGS['H2'].set_value(0.24619788897732542)
# # m.x_WGS['H2O'].set_value(0.030782009127793556)

# # m.x_total_WGS = Var(bounds=(1e-5, None), initialize=9.833350116365756)
# # m.k_wgs = Var(bounds=(1e-4, 100), initialize=12.337789289814305)

# # # === WGS outlet (kg/s) ===
# # m.x_WGS_kg = Var(m.Components, bounds=(1e-5, None))
# # m.x_WGS_kg['CH4'].set_value(0)
# # m.x_WGS_kg['CO'].set_value(0.2628981612284802)
# # m.x_WGS_kg['CO2'].set_value(0.00041356249102752785)
# # m.x_WGS_kg['H2'].set_value(2.73548577636471e-05)
# # m.x_WGS_kg['H2O'].set_value(9.990000168286174e-06)

# # # === WGS conversion and steam ===
# # m.WGS_C1 = Var(bounds=(0.01, 1), initialize=0.009999990000001055)
# # m.WGS_C2 = Var(bounds=(0.01, 1), initialize=0.8526384382995184)
# # m.steam_mol_wgs = Var(bounds=(1e-5, None), initialize=0.06809225622222943)
# # m.steam_kg_wgs = Var(bounds=(1e-5, None), initialize=0.0012266819958434631)

# # m.heat_recovered_wgs = Var(domain=PositiveReals, initialize=2.727587354647641)
# # m.heat_syngas_to_wgs = Var(domain=PositiveReals, initialize=574.299616038785)
# # m.power_CHP_engine = Var(bounds=(0, 15000), initialize=14697.290033639703)

# # # === System-level and economic variables (unchanged) ===

# # m.syngas_max_flow = Var(domain=PositiveReals, initialize=4998.739298227836)

# # m.annual_labor_cost = Var(domain=NonNegativeReals, initialize=918182.5405757254)
# # m.annual_maintenance_cost = Var(domain=NonNegativeReals, initialize=224511.08744293015)
# # m.annual_insurance_cost = Var(domain=NonNegativeReals, initialize=112255.54372146507)
# # m.annual_waste_cost = Var(domain=NonNegativeReals, initialize=535552.3235017173)
# # m.annual_catalyst_cost = Var(bounds=(0,1e4), initialize=955.993303901613)

# # # Economic variables (equipment sizing, cost) -- assumed time-independent here

# # m.gasifier_capacity = Var(bounds=(1e-5,None), initialize=9.0)  # Gasifier capacity (t/h)
# # m.gas_cleaning_capacity = Var(bounds=(1e-5,None), initialize=4.998739288228749) # Gas cleaning capcity (t/h)
# # m.chp_capacity = Var(bounds=(1e-5,None), initialize=14697.290033639703)  # CHP capacity (kW)
# # m.wgs_capacity = Var(bounds=(1e-5,None), initialize=0.955993303901613)  # WGS capacity (t/h)
# # m.separator_capacity = Var(bounds=(1e-5,None), initialize=0.32)  # WGS capacity (t/h)

# # m.gasifier_cost = Var(domain=PositiveReals, initialize=1838041.4336851789)  # Gasifier cost (USD)
# # m.sng_cleaning_cost = Var(domain=PositiveReals, initialize=743237.9381467237)  # Gas cleaner cost (USD)
# # m.chp_cost = Var(domain=PositiveReals, initialize=4643253.200605756)  # CHP cost (USD)
# # m.wgs_cost = Var(domain=PositiveReals, initialize=238292.56374947034)  # WGS cost (USD)
# # m.separator_cost = Var(domain=PositiveReals, initialize=20877.7785772101)  # Separator cost (USD)
# # m.total_capital_cost = Var(domain=PositiveReals, initialize=7483702.914764338)  # Total capital cost (USD)


# # # Annual revenue and costs
# # m.annual_revenue = Var(domain=Reals, initialize=82141215.71385965)  # Annual revenue from hydrogen sales
# # m.annual_biomass_cost = Var(domain=Reals, initialize=7650747.47859596)  # Annual biomass cost
# # m.annual_power_cost = Var(domain=Reals, initialize=-14046206.47607954)  # Annual power cost
# # m.annual_steam_cost = Var(domain=Reals, initialize=773.692868418389)  # Annual steam cost
# # m.annual_operating_cost = Var(domain=Reals, initialize=9442978.660010118)  # Total annual operating cost
# # m.annual_cash_flow = Var(domain=Reals, initialize=72698237.05384953)  # Annual cash flow (revenue - operating costs)

# # # NPV calculation
# # m.NPV = Var(domain=Reals, initialize=611437370.5572964)  # Net Present Value
# # m.NPV_hydrogen_prod = Var(bounds=(1e-5,None), initialize=41704537.111784175)

# # # Levelised cost of hydrogen
# # m.LCOH = Var(domain=Reals, initialize=2.0040757048453006)

# # ####2####

# # # === Reactor operating variables ===
# # m.T_opt = Var(bounds=(873.15, 1143.15), initialize=1143.1500099993436)  # K
# # m.k1 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.025181301616753547)
# # m.k2 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.8766772277386379)

# # # === Syngas composition (mol/s) ===
# # m.x = Var(m.Components, bounds=(1e-4, 5))
# # m.x['CH4'].set_value(0.022897214950971582)
# # m.x['CO'].set_value(0.9672520289683236)
# # m.x['CO2'].set_value(0.009850756080704813)
# # m.x['H2'].set_value(0.9535692616191472)
# # m.x['H2O'].set_value(0.011077517212407776)

# # m.m = Var(domain=PositiveReals, bounds=(0, 150), initialize=0.042091281442883134)  # mol/s
# # m.x_total = Var(bounds=(1e-5, 500), initialize=1.9646467788315551)  # mol/s

# # # === Syngas composition (kg/s) ===
# # m.x_kg = Var(m.Components, bounds=(1e-5, None))
# # m.x_kg['CH4'].set_value(0.01697747783504695)
# # m.x_kg['CO'].set_value(1.2523880218285959)
# # m.x_kg['CO2'].set_value(0.020040432087541873)
# # m.x_kg['H2'].set_value(0.08886462609806892)
# # m.x_kg['H2O'].set_value(0.009224925064670065)

# # m.x_kg_total = Var(bounds=(1e-5, None), initialize=1.3874954829139239)  # kg/s
# # m.LHV_sng = Var(bounds=(1e-5, None), initialize=17.41392915618476)  # MJ/kg
# # m.CGE = Var(bounds=(61, 100), initialize=96.83729002522337)  # %

# # # === Energy demands (kW) ===
# # m.P_drying = Var(domain=PositiveReals, initialize=28.98337546159429)
# # m.P_biomass_heat = Var(domain=PositiveReals, initialize=3099.611027116991)
# # m.P_air_heat = Var(bounds=(-1e-10,None), initialize=-2.2642136467519704e-41)
# # m.P_water_heat = Var(bounds=(-1e-10,None), initialize=1.8456531016627344e-40)
# # m.P_evap = Var(domain=PositiveReals, initialize=526.2355113309416)
# # m.P_heat_vap = Var(domain=PositiveReals, initialize=331.6913697458603)
# # m.P_heat_water_total = Var(domain=PositiveReals, initialize=857.9268810768019)

# # m.gasifier_elec_power_req = Var(domain=PositiveReals, initialize=20.225408670704986)
# # m.grid_power_req = Var(domain=Reals, initialize=-10730.469865921663)
# # m.gasifier_th_power_req = Var(domain=PositiveReals, initialize=3986.5212836553874)
# # m.Th_power_total = Var(domain=PositiveReals, initialize=4983.151604569234 )
# # m.Elec_power_total = Var(domain=PositiveReals, initialize=13438.369093240459)

# # # === Syngas split ===
# # m.sng_flow_rate_CHP_engine = Var(domain=PositiveReals, initialize=0.8462889923388518)
# # m.sng_flow_rate_WGS = Var(bounds=(1e-4, 100), initialize=0.5412064905750721)

# # # === WGS outlet (mol/s) ===
# # m.x_WGS = Var(m.Components, bounds=(1e-4, None))
# # m.x_WGS['CH4'].set_value(0)
# # m.x_WGS['CO'].set_value(9.417600324863823)
# # m.x_WGS['CO2'].set_value(0.1567862133931502)
# # m.x_WGS['H2'].set_value(0.21844524686379166)
# # m.x_WGS['H2O'].set_value(0.030782008751241784)

# # m.x_total_WGS = Var(bounds=(1e-5, None), initialize=9.823613793872006)
# # m.k_wgs = Var(bounds=(1e-4, 100), initialize=12.337789289814305)

# # # === WGS outlet (kg/s) ===
# # m.x_WGS_kg = Var(m.Components, bounds=(1e-5, None))
# # m.x_WGS_kg['CH4'].set_value(0)
# # m.x_WGS_kg['CO'].set_value(0.2637869850994357)
# # m.x_WGS_kg['CO2'].set_value(0.0003801464647988744)
# # m.x_WGS_kg['H2'].set_value(2.4261901948099795e-05)
# # m.x_WGS_kg['H2O'].set_value(9.990000046079979e-06)

# # # === WGS conversion and steam ===
# # m.WGS_C1 = Var(bounds=(0.01, 1), initialize=0.009999990000201879)
# # m.WGS_C2 = Var(bounds=(0.01, 1), initialize=0.8182768424195)
# # m.steam_mol_wgs = Var(bounds=(1e-5, None), initialize=0.07983767884848443)
# # m.steam_kg_wgs = Var(bounds=(1e-5, None), initialize=0.0014382757844554471)

# # m.heat_recovered_wgs = Var(domain=PositiveReals, initialize=2.7368089663705746)
# # m.heat_syngas_to_wgs = Var(domain=PositiveReals, initialize=573.868143942358)
# # m.power_CHP_engine = Var(bounds=(0, 15000), initialize=14737.216558247754)

# # # === System-level and economic variables (unchanged) ===

# # m.syngas_max_flow = Var(domain=PositiveReals, initialize=4994.983738675022)

# # m.annual_labor_cost = Var(domain=NonNegativeReals, initialize=917993.9843174705)
# # m.annual_maintenance_cost = Var(domain=NonNegativeReals, initialize=224769.47081757322)
# # m.annual_insurance_cost = Var(domain=NonNegativeReals, initialize=112384.73540878661)
# # m.annual_waste_cost = Var(domain=NonNegativeReals, initialize=535432.9904006638)
# # m.annual_catalyst_cost = Var(bounds=(0,1e4), initialize=959.2253906726346)

# # # Economic variables (equipment sizing, cost) -- assumed time-independent here

# # m.gasifier_capacity = Var(bounds=(1e-5,None), initialize=9.0)  # Gasifier capacity (t/h)
# # m.gas_cleaning_capacity = Var(bounds=(1e-5,None), initialize=4.994983728869918) # Gas cleaning capcity (t/h)
# # m.chp_capacity = Var(bounds=(1e-5,None), initialize=14737.216558247754)  # CHP capacity (kW)
# # m.wgs_capacity = Var(bounds=(1e-5,None), initialize=0.9592253906726347)  # WGS capacity (t/h)
# # m.separator_capacity = Var(bounds=(1e-5,None), initialize=0.32)  # WGS capacity (t/h)

# # m.gasifier_cost = Var(domain=PositiveReals, initialize=1838041.4336851789 )  # Gasifier cost (USD)
# # m.sng_cleaning_cost = Var(domain=PositiveReals, initialize=742863.766678128)  # Gas cleaner cost (USD)
# # m.chp_cost = Var(domain=PositiveReals, initialize=4651700.676034188)  # CHP cost (USD)
# # m.wgs_cost = Var(domain=PositiveReals, initialize=238832.03894440253)  # WGS cost (USD)
# # m.separator_cost = Var(domain=PositiveReals, initialize=20877.7785772101)  # Separator cost (USD)
# # m.total_capital_cost = Var(domain=PositiveReals, initialize=7492315.693919107)  # Total capital cost (USD)


# # # Annual revenue and costs
# # m.annual_revenue = Var(domain=Reals, initialize=82397118.66715956)  # Annual revenue from hydrogen sales
# # m.annual_biomass_cost = Var(domain=Reals, initialize=7649042.7200094825)  # Annual biomass cost
# # m.annual_power_cost = Var(domain=Reals, initialize=-14099837.403821062)  # Annual power cost
# # m.annual_steam_cost = Var(domain=Reals, initialize=907.1493027717395)  # Annual steam cost
# # m.annual_operating_cost = Var(domain=Reals, initialize=9441490.27564742)  # Total annual operating cost
# # m.annual_cash_flow = Var(domain=Reals, initialize=72955628.39151214)  # Annual cash flow (revenue - operating costs)

# # # NPV calculation
# # m.NPV = Var(domain=Reals, initialize=613620075.3322461)  # Net Present Value
# # m.NPV_hydrogen_prod = Var(bounds=(1e-5,None), initialize=41704537.11179148)

# # # Levelised cost of hydrogen
# # m.LCOH = Var(domain=Reals, initialize=2.0038597767103137)


# # ####3####

# # # === Reactor operating variables ===
# # m.T_opt = Var(bounds=(873.15, 1143.15), initialize=1143.1500099999976)  # K
# # m.k1 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.025181301616617226)
# # m.k2 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.8766772277368828)

# # # === Syngas composition (mol/s) ===
# # m.x = Var(m.Components, bounds=(1e-4, 5))
# # m.x['CH4'].set_value(0.02291112993929634)
# # m.x['CO'].set_value(0.9669066636955784)
# # m.x['CO2'].set_value(0.010182206365125271)
# # m.x['H2'].set_value(0.9538589669385934)
# # m.x['H2O'].set_value(0.01145781438733937)

# # m.m = Var(domain=PositiveReals, bounds=(0, 150), initialize=0.042789113913910444)  # mol/s
# # m.x_total = Var(bounds=(1e-5, 500), initialize=1.965316781325933)  # mol/s

# # # === Syngas composition (kg/s) ===
# # m.x_kg = Var(m.Components, bounds=(1e-5, None))
# # m.x_kg['CH4'].set_value(0.017176081630673638)
# # m.x_kg['CO'].set_value(1.2658168870237347)
# # m.x_kg['CO2'].set_value(0.020944330692879605)
# # m.x_kg['H2'].set_value(0.08987686546295008)
# # m.x_kg['H2O'].set_value(0.009647377486402832)

# # m.x_kg_total = Var(bounds=(1e-5, None), initialize=1.4034615422966408)  # kg/s
# # m.LHV_sng = Var(bounds=(1e-5, None), initialize=17.4060904127461)  # MJ/kg
# # m.CGE = Var(bounds=(61, 100), initialize=96.80122865312057)  # %

# # # === Energy demands (kW) ===
# # m.P_drying = Var(domain=PositiveReals, initialize=29.31461001277214)
# # m.P_biomass_heat = Var(domain=PositiveReals, initialize=3135.0347226361914)
# # m.P_air_heat = Var(bounds=(-1e-10,None), initialize=-1.3502754460177738e-42)
# # m.P_water_heat = Var(bounds=(-1e-10,None), initialize=-3.6782257107470875e-40)
# # m.P_evap = Var(domain=PositiveReals, initialize=532.2495583711327)
# # m.P_heat_vap = Var(domain=PositiveReals, initialize=335.4820821886252)
# # m.P_heat_water_total = Var(domain=PositiveReals, initialize=867.7316405597579)

# # m.gasifier_elec_power_req = Var(domain=PositiveReals, initialize=20.225630535790522)
# # m.grid_power_req = Var(domain=Reals, initialize=-10947.6934062553)
# # m.gasifier_th_power_req = Var(domain=PositiveReals, initialize=4032.0809732087214)
# # m.Th_power_total = Var(domain=PositiveReals, initialize=5040.101216510901)
# # m.Elec_power_total = Var(domain=PositiveReals, initialize=13709.89879598886)

# # # === Syngas split ===
# # m.sng_flow_rate_CHP_engine = Var(domain=PositiveReals, initialize=0.8617673270854458)
# # m.sng_flow_rate_WGS = Var(bounds=(1e-4, 100), initialize=0.541694215211195)

# # # === WGS outlet (mol/s) ===
# # m.x_WGS = Var(m.Components, bounds=(1e-4, None))
# # m.x_WGS['CH4'].set_value(0)
# # m.x_WGS['CO'].set_value(9.419509288626891)
# # m.x_WGS['CO2'].set_value(0.15891583749828544)
# # m.x_WGS['H2'].set_value(0.2226852126332253)
# # m.x_WGS['H2O'].set_value(0.03078200862568246)

# # m.x_total_WGS = Var(bounds=(1e-5, None), initialize=9.831892347384084)
# # m.k_wgs = Var(bounds=(1e-4, 100), initialize=12.337789289814307)

# # # === WGS outlet (kg/s) ===
# # m.x_WGS_kg = Var(m.Components, bounds=(1e-5, None))
# # m.x_WGS_kg['CH4'].set_value(0)
# # m.x_WGS_kg['CO'].set_value(0.2638404551744392)
# # m.x_WGS_kg['CO2'].set_value(0.0003853324933475674)
# # m.x_WGS_kg['H2'].set_value(2.473426386952179e-05)
# # m.x_WGS_kg['H2O'].set_value(9.990000005330926e-06)

# # # === WGS conversion and steam ===
# # m.WGS_C1 = Var(bounds=(0.01, 1), initialize=0.009999990000000075)
# # m.WGS_C2 = Var(bounds=(0.01, 1), initialize=0.8244067552299448)
# # m.steam_mol_wgs = Var(bounds=(1e-5, None), initialize=0.07816430077360861)
# # m.steam_kg_wgs = Var(bounds=(1e-5, None), initialize=0.0014081298784365593)

# # m.heat_recovered_wgs = Var(domain=PositiveReals, initialize=2.737363722193452)
# # m.heat_syngas_to_wgs = Var(domain=PositiveReals, initialize=580.4717062443492)
# # m.power_CHP_engine = Var(bounds=(0, 15000), initialize=15000.00000999981)

# # # === System-level and economic variables (unchanged) ===

# # m.syngas_max_flow = Var(domain=PositiveReals, initialize=5052.461552271344)

# # m.annual_labor_cost = Var(domain=NonNegativeReals, initialize=928481.6909268087)
# # m.annual_maintenance_cost = Var(domain=NonNegativeReals, initialize=226604.2817727188)
# # m.annual_insurance_cost = Var(domain=NonNegativeReals, initialize=113302.1408863594)
# # m.annual_waste_cost = Var(domain=NonNegativeReals, initialize=541552.1502099202)
# # m.annual_catalyst_cost = Var(bounds=(0,1e4), initialize=959.4198273068516)

# # # Economic variables (equipment sizing, cost) -- assumed time-independent here

# # m.gasifier_capacity = Var(bounds=(1e-5,None), initialize=9.0)  # Gasifier capacity (t/h)
# # m.gas_cleaning_capacity = Var(bounds=(1e-5,None), initialize=5.052461542284782) # Gas cleaning capcity (t/h)
# # m.chp_capacity = Var(bounds=(1e-5,None), initialize=15000.00000999981)  # CHP capacity (kW)
# # m.wgs_capacity = Var(bounds=(1e-5,None), initialize=0.9594198273068516)  # WGS capacity (t/h)
# # m.separator_capacity = Var(bounds=(1e-5,None), initialize=0.32)  # WGS capacity (t/h)

# # m.gasifier_cost = Var(domain=PositiveReals, initialize=1838041.4336851789)  # Gasifier cost (USD)
# # m.sng_cleaning_cost = Var(domain=PositiveReals, initialize=748580.2502378989)  # Gas cleaner cost (USD)
# # m.chp_cost = Var(domain=PositiveReals, initialize=4707112.122915464)  # CHP cost (USD)
# # m.wgs_cost = Var(domain=PositiveReals, initialize=238864.47367487545)  # WGS cost (USD)
# # m.separator_cost = Var(domain=PositiveReals, initialize=20877.7785772101)  # Separator cost (USD)
# # m.total_capital_cost = Var(domain=PositiveReals, initialize=7553476.059090627)  # Total capital cost (USD)


# # # Annual revenue and costs
# # m.annual_revenue = Var(domain=Reals, initialize=82820345.47495653)  # Annual revenue from hydrogen sales
# # m.annual_biomass_cost = Var(domain=Reals, initialize=7736459.288713145)  # Annual biomass cost
# # m.annual_power_cost = Var(domain=Reals, initialize=-14385269.13581946)  # Annual power cost
# # m.annual_steam_cost = Var(domain=Reals, initialize=888.1356769275067)  # Annual steam cost
# # m.annual_operating_cost = Var(domain=Reals, initialize=9548247.108013187)  # Total annual operating cost
# # m.annual_cash_flow = Var(domain=Reals, initialize=73272098.36694333)  # Annual cash flow (revenue - operating costs)

# # # NPV calculation
# # m.NPV = Var(domain=Reals, initialize=616253202.2682985)  # Net Present Value
# # m.NPV_hydrogen_prod = Var(bounds=(1e-5,None), initialize=42179677.196725115)

# # # Levelised cost of hydrogen
# # m.LCOH = Var(domain=Reals, initialize=2.00345203588634)

# ####4####

# # === Reactor operating variables ===
# m.T_opt = Var(bounds=(873.15, 1143.15), initialize=1143.1500099999837)  # K
# m.k1 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.02518130161661987)
# m.k2 = Var(domain=PositiveReals, bounds=(1e-4, 10), initialize=0.8766772277369185)

# # === Syngas composition (mol/s) ===
# m.x = Var(m.Components, bounds=(1e-4, 5))
# m.x['CH4'].set_value(0.022991912653426046)
# m.x['CO'].set_value(0.9649033993560977)
# m.x['CO2'].set_value(0.0121046879904762)
# m.x['H2'].set_value(0.9555391004215552)
# m.x['H2O'].set_value(0.013673462393232359)

# m.m = Var(domain=PositiveReals, bounds=(0, 150), initialize=0.04684646083102462)  # mol/s
# m.x_total = Var(bounds=(1e-5, 500), initialize=1.9692125628147878)  # mol/s

# # === Syngas composition (kg/s) ===
# m.x_kg = Var(m.Components, bounds=(1e-5, None))
# m.x_kg['CH4'].set_value(0.01697747783504695)
# m.x_kg['CO'].set_value(1.2523880218285959)
# m.x_kg['CO2'].set_value(0.020040432087541873)
# m.x_kg['H2'].set_value(0.08886462609806892)
# m.x_kg['H2O'].set_value(0.009224925064670065)

# m.x_kg_total = Var(bounds=(1e-5, None), initialize=1.3874954829139239)  # kg/s
# m.LHV_sng = Var(bounds=(1e-5, None), initialize=17.360651151651144)  # MJ/kg
# m.CGE = Var(bounds=(61, 100), initialize=96.5920740949808)  # %

# # === Energy demands (kW) ===
# m.P_drying = Var(domain=PositiveReals, initialize=28.989835048759886)
# m.P_biomass_heat = Var(domain=PositiveReals, initialize=3100.3018440893043)
# m.P_air_heat = Var(bounds=(-1e-10,None), initialize=0.0)
# m.P_water_heat = Var(bounds=(-1e-10,None), initialize=0.0)
# m.P_evap = Var(domain=PositiveReals, initialize=526.3527945700722)
# m.P_heat_vap = Var(domain=PositiveReals, initialize=331.76529451441536)
# m.P_heat_water_total = Var(domain=PositiveReals, initialize=858.1180890844876)

# m.gasifier_elec_power_req = Var(domain=PositiveReals, initialize=20.225412997406295)
# m.grid_power_req = Var(domain=Reals, initialize=-10689.654852419744)
# m.gasifier_th_power_req = Var(domain=PositiveReals, initialize=3987.4097682225515)
# m.Th_power_total = Var(domain=PositiveReals, initialize=4984.262210278189)
# m.Elec_power_total = Var(domain=PositiveReals, initialize=13387.350331771437)

# # === Syngas split ===
# m.sng_flow_rate_CHP_engine = Var(domain=PositiveReals, initialize=0.8465863351123133)
# m.sng_flow_rate_WGS = Var(bounds=(1e-4, 100), initialize=0.5419523588423875)

# # === WGS outlet (mol/s) ===
# m.x_WGS = Var(m.Components, bounds=(1e-4, None))
# m.x_WGS['CH4'].set_value(0)
# m.x_WGS['CO'].set_value(9.417600324863823)
# m.x_WGS['CO2'].set_value(0.1567862133931502)
# m.x_WGS['H2'].set_value(0.21844524686379166)
# m.x_WGS['H2O'].set_value(0.030782008751241784)

# m.x_total_WGS = Var(bounds=(1e-5, None), initialize=9.823613793872006)
# m.k_wgs = Var(bounds=(1e-4, 100), initialize=12.337789289814305)

# # === WGS outlet (kg/s) ===
# m.x_WGS_kg = Var(m.Components, bounds=(1e-5, None))
# m.x_WGS_kg['CH4'].set_value(0)
# m.x_WGS_kg['CO'].set_value(0.2637869850994357)
# m.x_WGS_kg['CO2'].set_value(0.0003801464647988744)
# m.x_WGS_kg['H2'].set_value(2.4261901948099795e-05)
# m.x_WGS_kg['H2O'].set_value(9.990000046079979e-06)

# # === WGS conversion and steam ===
# m.WGS_C1 = Var(bounds=(0.01, 1), initialize=0.009999990000001055)
# m.WGS_C2 = Var(bounds=(0.01, 1), initialize=0.8526384382995184)
# m.steam_mol_wgs = Var(bounds=(1e-5, None), initialize=0.06809225622222943)
# m.steam_kg_wgs = Var(bounds=(1e-5, None), initialize=0.0012266819958434631)

# m.heat_recovered_wgs = Var(domain=PositiveReals, initialize=2.727587354647641)
# m.heat_syngas_to_wgs = Var(domain=PositiveReals, initialize=574.299616038785)
# m.power_CHP_engine = Var(bounds=(0, 15000), initialize=14697.290033639703)

# # === System-level and economic variables (unchanged) ===

# m.syngas_max_flow = Var(domain=PositiveReals, initialize=4998.739298227836)

# m.annual_labor_cost = Var(domain=NonNegativeReals, initialize=918182.5405757254)
# m.annual_maintenance_cost = Var(domain=NonNegativeReals, initialize=224511.08744293015)
# m.annual_insurance_cost = Var(domain=NonNegativeReals, initialize=112255.54372146507)
# m.annual_waste_cost = Var(domain=NonNegativeReals, initialize=535552.3235017173)
# m.annual_catalyst_cost = Var(bounds=(0,1e4), initialize=955.993303901613)

# # Economic variables (equipment sizing, cost) -- assumed time-independent here

# m.gasifier_capacity = Var(bounds=(1e-5,None), initialize=9.0)  # Gasifier capacity (t/h)
# m.gas_cleaning_capacity = Var(bounds=(1e-5,None), initialize=4.998739288228749) # Gas cleaning capcity (t/h)
# m.chp_capacity = Var(bounds=(1e-5,None), initialize=14697.290033639703)  # CHP capacity (kW)
# m.wgs_capacity = Var(bounds=(1e-5,None), initialize=0.955993303901613)  # WGS capacity (t/h)
# m.separator_capacity = Var(bounds=(1e-5,None), initialize=0.32)  # WGS capacity (t/h)

# m.gasifier_cost = Var(domain=PositiveReals, initialize=1838041.4336851789)  # Gasifier cost (USD)
# m.sng_cleaning_cost = Var(domain=PositiveReals, initialize=743237.9381467237)  # Gas cleaner cost (USD)
# m.chp_cost = Var(domain=PositiveReals, initialize=4643253.200605756)  # CHP cost (USD)
# m.wgs_cost = Var(domain=PositiveReals, initialize=238292.56374947034)  # WGS cost (USD)
# m.separator_cost = Var(domain=PositiveReals, initialize=20877.7785772101)  # Separator cost (USD)
# m.total_capital_cost = Var(domain=PositiveReals, initialize=7483702.914764338)  # Total capital cost (USD)


# # Annual revenue and costs
# m.annual_revenue = Var(domain=Reals, initialize=82141215.71385965)  # Annual revenue from hydrogen sales
# m.annual_biomass_cost = Var(domain=Reals, initialize=7650747.47859596)  # Annual biomass cost
# m.annual_power_cost = Var(domain=Reals, initialize=-14046206.47607954)  # Annual power cost
# m.annual_steam_cost = Var(domain=Reals, initialize=773.692868418389)  # Annual steam cost
# m.annual_operating_cost = Var(domain=Reals, initialize=9442978.660010118)  # Total annual operating cost
# m.annual_cash_flow = Var(domain=Reals, initialize=72698237.05384953)  # Annual cash flow (revenue - operating costs)

# # NPV calculation
# m.NPV = Var(domain=Reals, initialize=611437370.5572964)  # Net Present Value
# m.NPV_hydrogen_prod = Var(bounds=(1e-5,None), initialize=41704537.111784175)

# # Levelised cost of hydrogen
# m.LCOH = Var(domain=Reals, initialize=2.0040757048453006)


# # Define an external function
# # def blackbox1(a,b):
# #     return a/(b**2)

# # bb1 = ExternalFunction(blackbox1)

# def blackbox2(a,b,c,d):
#     return (a*b)/(c*d)

# bb2 = ExternalFunction(blackbox2)

# def blackbox3(a,b,c,d,e):
#     return (a*b)/((c/e)*(d/e))

# bb3 = ExternalFunction(blackbox3)

# def blackbox4(a):
#     return (-(J_m / (R * a)) +
#              (delA_m * log(a)) +
#              ((delB_m * a) / 2) +
#              ((delC_m * a**2) / 6) +
#              (delD_m / (2 * a**2)) +
#              I_m)

# bb4 = ExternalFunction(blackbox4)

# def blackbox5(a):
#     return (-(J_wg / (R * a))
#             +  delA_wg * log(a)
#             + (delB_wg * a) / 2
#             + (delC_wg * a**2) / 6
#             +  delD_wg / (2 * a**2)
#             +  I_wg)

# bb5 = ExternalFunction(blackbox5)





# # ----------------------------------------------------------------------
# # 5.  CONSTRAINTS ######################
# # ----------------------------------------------------------------------

# # ------------------------------------------------------------------
# # Constraint 0  •  Dry biomass calculation
# # ------------------------------------------------------------------
# def dry_biomass(m):
#     return m.dry_biomass_feed == m.biomass_feed * (1 - m.biomass_database[BIOMASS]['MC'])

# m.Cons_0 = Constraint(rule=dry_biomass)

# # ------------------------------------------------------------------
# # Constraint 1  •  Power required for drying (P_drying)
# # ------------------------------------------------------------------
# def P_drying_biomass(m):
#     """Dryer thermal duty (no time indexing)."""
#     if Drying_fix == 'Yes':
#         return m.P_drying == (
#             m.biomass_feed *
#             m.biomass_database[BIOMASS]['MC'] *
#             m.biomass_database[BIOMASS]['CP'] *
#             (Drying_temp - Inlet_temp)
#         )
#     else:
#         return m.P_drying == 0

# m.Cons_1 = Constraint(rule=P_drying_biomass)

# # ------------------------------------------------------------------
# # Constraint 2  •  Power required to heat biomass (P_biomass_heat)
# # ------------------------------------------------------------------
# def P_heat_biomass(m):
#     """Biomass preheating load (no time indexing)."""
#     cp = m.biomass_database[BIOMASS]['CP']
#     if Drying_fix == 'Yes':
#         deltaT = m.T_opt - Drying_temp
#     else:
#         deltaT = m.T_opt - Inlet_temp
#     return m.P_biomass_heat == m.biomass_feed * cp * deltaT

# m.Cons_2 = Constraint(rule=P_heat_biomass)

# # ------------------------------------------------------------------
# # Constraint 3  •  Power required to heat air (P_air_heat)
# # ------------------------------------------------------------------
# def P_heat_air(m):
#     """Air preheating power requirement (no time indexing)."""
#     if GA == 'Air':
#         return m.P_air_heat == AF_ratio * m.biomass_feed * Cp_air * (m.T_opt - Inlet_temp)
#     else:
#         return m.P_air_heat == 0

# m.Cons_3 = Constraint(rule=P_heat_air)

# # ------------------------------------------------------------------
# # Constraint 4  •  Power required to heat moisture (water) in biomass
# # ------------------------------------------------------------------
# def P_heat_water(m):
#     """Water heating power (no time indexing; only if drying is skipped)."""
#     if Drying_fix == 'Yes':
#         return m.P_water_heat == 0
#     else:
#         return m.P_water_heat == (
#             m.biomass_feed *
#             m.biomass_database[BIOMASS]['MC'] *
#             Cp_water *
#             (373.15 - Inlet_temp)
#         )

# m.Cons_4 = Constraint(rule=P_heat_water)

# # ------------------------------------------------------------------
# # Constraint 5  •  Power required for evaporating biomass moisture
# # ------------------------------------------------------------------
# def P_evap(m):
#     """Evaporation energy (no time indexing)."""
#     return m.P_evap == m.biomass_feed * m.biomass_database[BIOMASS]['MC'] * H_vap

# m.Cons_5 = Constraint(rule=P_evap)

# # ------------------------------------------------------------------
# # Constraint 6  •  Power for heating evaporated vapors (steam)
# # ------------------------------------------------------------------
# def P_vap_heat(m):
#     """Power required to heat water vapor (no time indexing)."""
#     return m.P_heat_vap == (
#         m.biomass_feed *
#         m.biomass_database[BIOMASS]['MC'] *
#         Cp_vap *
#         (m.T_opt - 373.15)
#     )

# m.Cons_6 = Constraint(rule=P_vap_heat)

# # ------------------------------------------------------------------
# # Constraint 7  •  Total power for heating water (moisture)
# # ------------------------------------------------------------------
# def P_total_heat_water(m):
#     """Total power needed to heat moisture (liquid + vapor)."""
#     return m.P_heat_water_total == (
#         m.P_water_heat +
#         m.P_evap +
#         m.P_heat_vap
#     )

# m.Cons_7 = Constraint(rule=P_total_heat_water)

# # ------------------------------------------------------------------
# # Constraint 8  •  Total thermal power required by the gasifier
# # ------------------------------------------------------------------
# def TP_total(m):
#     """Total thermal power requirement of the gasifier for hour h, month mm."""
#     return m.gasifier_th_power_req == (
#         m.P_drying +
#         m.P_biomass_heat +
#         m.P_air_heat +
#         m.P_heat_water_total
#     )

# m.Cons_8 = Constraint(rule=TP_total)

# # ------------------------------------------------------------------
# # Constraint 9  •  Total electrical power requirement of the gasifier
# # ------------------------------------------------------------------
# def EP_total(m):
#     """Electrical power required by the gasifier per hour and month."""
#     return m.gasifier_elec_power_req == (
#         (8e-7) * (m.biomass_feed**2) +
#         (0.008 * m.biomass_feed) +
#         20.206
#     )

# m.Cons_9 = Constraint(rule=EP_total)

# # ------------------------------------------------------------------
# # Constraint 10  •  Reaction rate constant for methanation (k₁)
# # ------------------------------------------------------------------
# def rate_constant_k1(m):
#     """Temperature-dependent equilibrium constant for methanation (k₁)."""
#     return log(m.k1) == bb4(m.T_opt)

# m.Cons_10 = Constraint(rule=rate_constant_k1)

# # ------------------------------------------------------------------
# # Constraint 11  •  Temperature–dependent equilibrium constant k₂
# #                  for the Water‑Gas‑Shift (WGS) reaction
# # ------------------------------------------------------------------
# def rate_constant_k2(m):
#     """Calculate ln(k₂) for the WGS reaction."""
#     return log(bb2(m.x['H2'],m.x['CO2'],m.x['CO'],m.x['H2O'])) == bb5(m.T_opt)

# m.Cons_11 = Constraint(rule=rate_constant_k2)

# # ------------------------------------------------------------------
# # Constraint 12  •  Elemental carbon balance in the gasifier
# # ------------------------------------------------------------------
# def carbon_balance(m):
#     """
#     CO + CO₂ + CH₄ carbon atoms  =  Carbon atoms from one biomass 'mole'
#     """
#     # lhs  = moles of C leaving (1 C per molecule of CO / CO₂ / CH₄)
#     lhs = (m.x['CO'] +
#            m.x['CO2'] +
#            m.x['CH4'])

#     # rhs  = reference moles of C in feed  (currently normalised to 1)
#     rhs = 1.0
#     # If your biomass entry gives 'C_mol_per_kg', use:
#     # rhs = m.biomass_database[BIOMASS]['C_mol_per_kg'] * m.dry_biomass_feed[h, mm]

#     return lhs == rhs

# m.Cons_12 = Constraint(rule=carbon_balance)

# # ------------------------------------------------------------------
# # Constraint 13  •  Elemental hydrogen balance
# # ------------------------------------------------------------------
# def hydrogen_balance(m):
#     # moles of H leaving the gas phase
#     lhs = (2 * m.x['H2'] +
#            2 * m.x['H2O'] +
#            4 * m.x['CH4'])

#     # H brought in with dry biomass (per reference mole)
#     H_biomass = m.biomass_database[BIOMASS]['H']
#     # H from intrinsic biomass moisture (‘w’ moles of H₂O each with 2 H)
#     H_intrinsic = 2 * m.biomass_database[BIOMASS]['w']

#     if GA == 'Air':
#         rhs = H_biomass + H_intrinsic
#     else:  # GA == 'Steam'
#         # Additional steam feed m.m[h,mm]  (mol s⁻¹); each mol has 2 H
#         rhs = H_biomass + H_intrinsic + 2 * m.m

#     return lhs == rhs

# m.Cons_13 = Constraint(rule=hydrogen_balance)

# # ------------------------------------------------------------------
# # Constraint 14  •  Elemental oxygen balance
# # ------------------------------------------------------------------
# def oxygen_balance(m):
#     # Oxygen in the output syngas
#     lhs = (m.x['CO'] + 
#            2 * m.x['CO2'] +
#            m.x['H2O'])

#     # Oxygen from biomass and moisture
#     O_biomass = m.biomass_database[BIOMASS]['O']     # mol O/mol biomass
#     O_intrinsic = m.biomass_database[BIOMASS]['w']   # mol H2O/mol → mol O

#     # Add oxygen from gasifying agent
#     if GA == 'Air':
#         O_GA = 2 * m.m   # mol O2 × 2 O atoms
#     else:  # GA == 'Steam'
#         O_GA = m.m       # mol H2O × 1 O atom

#     rhs = O_biomass + O_intrinsic + O_GA

#     return lhs == rhs

# m.Cons_14 = Constraint(rule=oxygen_balance)

# # ------------------------------------------------------------------
# # Constraint 15  •  Methanation reaction equilibrium
# # ------------------------------------------------------------------
# def equilibrium_constant_k1(m):
#     return m.k1 * (m.x['H2'] ** 2) == m.x['CH4']

# m.Cons_15 = Constraint(rule=equilibrium_constant_k1)

# # # ------------------------------------------------------------------
# # # Constraint 16  •  Water-Gas Shift Reaction (CO + H₂O ⇌ CO₂ + H₂)
# # # ------------------------------------------------------------------
# # def equilibrium_constant_k2(m):
# #     return m.k2 * m.x['CO'] * m.x['H2O'] == m.x['H2'] * m.x['CO2']

# # m.Cons_16 = Constraint(rule=equilibrium_constant_k2)

# # ------------------------------------------------------------------
# # Constraint 17  •  Total Molar Flow Rate of Syngas (mol/s)
# # ------------------------------------------------------------------
# def syngas_mol(m):
#     if GA == 'Air':
#         return m.x_total == (
#             m.x['H2'] + m.x['CO'] + m.x['CO2'] +
#             m.x['H2O'] + m.x['CH4'] + 3.76 * m.m
#         )
#     elif GA == 'Steam':
#         return m.x_total == (
#             m.x['H2'] + m.x['CO'] + m.x['CO2'] +
#             m.x['H2O'] + m.x['CH4']
#         )

# m.Cons_17 = Constraint(rule=syngas_mol)

# # ------------------------------------------------------------------
# # Constraint 18  •  Mass Flow Rate of Syngas Components (kg/s)
# # ------------------------------------------------------------------
# def syngas_kg_rule(m, comp):
#     return m.x_kg[comp] == m.biomass_feed * (
#         ((m.x[comp]) * m.MM_gases[comp])
#         / (m.x_total * (
#             m.biomass_database[BIOMASS]['C'] * 12.011 +
#             m.biomass_database[BIOMASS]['H'] * 1.008 +
#             m.biomass_database[BIOMASS]['O'] * 16
#         ))
#     )

# m.Cons_18 = Constraint(m.Components, rule=syngas_kg_rule)

# # ------------------------------------------------------------------
# # Constraint 19  •  Total mass‑flow of raw syngas (kg s⁻¹)
# # ------------------------------------------------------------------
# def syngas_kg_total_rule(m):
#     # Mass of explicit components (H₂, CO, CO₂, H₂O, CH₄)  [kg s⁻¹]
#     explicit_mass = sum(m.x_kg[comp] for comp in m.Components)

#     if GA == 'Air':
#         # Add N₂ formed from air feed   ( 28.013 g mol⁻¹ )
#         # m.m is O₂ mol‑flow (mol s⁻¹)
#         # 3.76 × mol O₂  =  mol N₂
#         nitrogen_mass = 3.76 * m.m * 28.013 / 1000  # → kg s⁻¹
#         return m.x_kg_total == explicit_mass + nitrogen_mass
#     else:  # GA == 'Steam'
#         return m.x_kg_total == explicit_mass

# m.Cons_19 = Constraint(rule=syngas_kg_total_rule)

# # ------------------------------------------------------------------
# # Constraint 20  •  Lower‑Heating‑Value (LHV) of raw syngas  (MJ kg⁻¹)
# # ------------------------------------------------------------------
# # Per‑kilogram LHVs (MJ kg⁻¹) – declared once for clarity
# LHV_per_kg = {
#     "H2": 120.0,    # MJ/kg
#     "CO": 10.1,
#     "CH4": 50.0}
# def LHV_syngas_rule(m):
#     # Total energy contribution (MJ/s)
#     heating_MJ_per_s = (
#         LHV_per_kg["H2"]  * m.x_kg["H2"] +
#         LHV_per_kg["CO"]  * m.x_kg["CO"] +
#         LHV_per_kg["CH4"] * m.x_kg["CH4"]
#     )
#     # Normalize per kg of syngas to get MJ/kg
#     return m.LHV_sng == heating_MJ_per_s / m.x_kg_total

# m.Cons_20 = Constraint(rule=LHV_syngas_rule)

# # -------------------------------------------------------------------------------
# # Constraint 21  •  Cold Gas Efficiency (CGE) of the gasifier  (%)
# # -------------------------------------------------------------------------------
# def CGE_rule(m):
#     numerator = m.LHV_sng * m.x_kg_total              # MJ/s
#     denominator = m.dry_biomass_feed * m.biomass_database[BIOMASS]['LHV']  # MJ/s
#     return m.CGE == (numerator / denominator) * 100

# m.Cons_21 = Constraint(rule=CGE_rule)

# # ---------------------------------------------------------------------------------
# # Constraint 22 • Mass balance that splits the raw‑syngas stream between the
# #                CHP engine and the WGS/PSA line                           (kg s⁻¹)
# # ---------------------------------------------------------------------------------
# def sng_flow_streams_rule(m):
#     # Units of every term:  kg s⁻¹
#     return m.x_kg_total == m.sng_flow_rate_CHP_engine + m.sng_flow_rate_WGS

# m.Cons_22 = Constraint(rule=sng_flow_streams_rule)

# # ---------------------------------------------------------------------------------
# # Constraint 23 • Converts the CHP shaft‑power set‑point to the required
# #                mass flow of fuel syngas (kg s⁻¹ → kW relation).
# # ---------------------------------------------------------------------------------
# def sng_flow_CHP_engine_rule(m):
#     return (
#         m.sng_flow_rate_CHP_engine ==
#         m.power_CHP_engine / (m.LHV_sng * 1000.0)
#     )

# m.Cons_23 = Constraint(rule=sng_flow_CHP_engine_rule)

# # ---------------------------------------------------------------------------------
# # Constraint 24 • Overall CHP energy balance:
# #                mechanical shaft power equals the sum of the useful *thermal*
# #                and *electrical* outputs demanded by downstream users.
# # ---------------------------------------------------------------------------------
# def CHP_generation_rule(m):
#     return (
#         m.power_CHP_engine ==
#         m.Th_power_total   * m.th_eff_CHP_engine +
#         m.Elec_power_total * m.elec_eff_CHP_engine
#     )

# m.Cons_24 = Constraint(rule=CHP_generation_rule)

# # ---------------------------------------------------------------------------------
# # Constraint 25 • Ensures that the heat recovered from the CHP engine
# #                (at the stated efficiency) fully covers the gasifier’s
# #                thermal requirement.  (Additional heaters could be added.)
# # ---------------------------------------------------------------------------------
# def Gasifier_Th_rule(m):
#     return (
#         m.Th_power_total * m.th_eff_CHP_engine ==
#         m.gasifier_th_power_req
#     )

# m.Cons_25 = Constraint(rule=Gasifier_Th_rule)

# # ---------------------------------------------------------------------------------
# # Constraint 26 • Electrical balance for each hour‑month slice.
# #                Grid import (could be negative for export) closes the balance.
# # ---------------------------------------------------------------------------------
# def Gasifier_Elec_rule(m):
#     return (
#         m.Elec_power_total * m.elec_eff_CHP_engine
#         + m.grid_power_req ==
#         m.gasifier_elec_power_req
#     )

# m.Cons_26 = Constraint(rule=Gasifier_Elec_rule)

# # ---------------------------------------------------------------------------------
# # Constraint 27  •  Log‑linear Arrhenius fit for Water‑Gas‑Shift (WGS) rate
# # ---------------------------------------------------------------------------------
# def rate_constant_k_wgs(m):
#     """Natural‑log correlation for the intrinsic WGS rate constant k_WGS."""
#     T = m.T_wgs  # fixed (Param) in Kelvin
#     # Right‑hand side expression (dimensionless)
#     rhs = (-(J_wg) / (R * T)                          +
#            delA_wg * log(T)                      +
#            (delB_wg * T) / 2                         +
#            (delC_wg * T**2) / 6                      +
#            (delD_wg / (2 * T**2))                    +
#            I_wg)
#     # Constrain ln(k_WGS) to equal the fitted correlation
#     return log(bb3(m.x_WGS['H2'],m.x_WGS['CO2'],m.x['CO'],m.x['H2O'],m.x_total))  == rhs

# # Create the indexed constraint
# m.Cons_27 = Constraint(rule=rate_constant_k_wgs)

# # # ──────────────────────────────────────────────────────────────────────────────
# # # Constraint 28  •  WGS Equilibrium Relationship (molar basis, dimensionless)
# # # ──────────────────────────────────────────────────────────────────────────────
# # def equilibrium_constant_K_WGS(m):
# #     """Equilibrium constraint for the Water‑Gas‑Shift (WGS) reaction."""
# #     # Left-hand side: K_eq × [CO] × [H2O]  (approx. mol fractions)
# #     lhs = m.k_wgs * \
# #           (m.x['CO']   / m.x_total) * \
# #           (m.x['H2O']  / m.x_total)
# #     # Right-hand side: [H2] × [CO2] (WGS outlet molar flows)
# #     rhs = m.x_WGS['H2'] * m.x_WGS['CO2']
    
# #     return lhs == rhs

# # m.Cons_28 = Constraint(rule=equilibrium_constant_K_WGS)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 29 • Molar CO₂ balance in WGS outlet [mol/s]
# # ────────────────────────────────────────────────────────────────────────────────
# def carbon_balance_wgs(m):
#     CO_in_mol  = ( m.sng_flow_rate_WGS
#                    * (m.x['CO'] / m.x_total)
#                    * 1000 / m.MM_gases['CO'] )  # mol/s
#     CO2_in_mol = ( m.sng_flow_rate_WGS
#                    * (m.x['CO2'] / m.x_total)
#                    * 1000 / m.MM_gases['CO2'] )  # mol/s
#     CO_conv = m.WGS_C1 * CO_in_mol  # mol/s of CO converted → CO₂

#     return m.x_WGS['CO2'] == CO2_in_mol + CO_conv

# m.Cons_29 = Constraint(rule=carbon_balance_wgs)


# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 30  •  Hydrogen balance in the Water‑Gas‑Shift reactor
# # ────────────────────────────────────────────────────────────────────────────────
# def hydrogen_balance_wgs(m):
#     # Raw syngas H2O in mol/s → scaled to WGS stream
#     H2O_in_mol = m.sng_flow_rate_WGS * (m.x['H2O'] / m.x_total) * 1000 / m.MM_gases['H2O']

#     # Total H2 produced = 2*(converted H2O + injected steam)
#     H2_produced = 2 * (m.WGS_C2 * H2O_in_mol + m.steam_mol_wgs)

#     return 2 * m.x_WGS['H2'] == H2_produced

# # Register the corrected constraint
# m.Cons_30 = Constraint(rule=hydrogen_balance_wgs)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 31  •  Oxygen balance in the Water‑Gas‑Shift reactor
# # ────────────────────────────────────────────────────────────────────────────────
# def oxygen_balance_wgs(m):
#     # ----- Inlet molar flows (scaled to WGS stream) ----------------------------
#     CO_in_mol  = ( m.sng_flow_rate_WGS                     # kg s⁻¹
#                    * (m.x['CO'] / m.x_total)     # kg CO per kg syngas
#                    * 1000 / m.MM_gases['CO'] )                    # → mol s⁻¹

#     H2O_in_mol = ( m.sng_flow_rate_WGS
#                    * (m.x['H2O'] / m.x_total)
#                    * 1000 / m.MM_gases['H2O'] )

#     # ----- Oxygen atoms in reactants that disappear from the inlet ------------
#     O_from_CO_conv   = m.WGS_C1 * CO_in_mol         # 1 O atom per CO converted
#     O_from_H2O_conv  = m.WGS_C2 * H2O_in_mol        # 1 O atom per H₂O converted
#     O_from_steam     = m.steam_mol_wgs              # each injected H₂O brings 1 O

#     # ----- Oxygen atoms leaving as CO₂ (2 O per molecule) ---------------------
#     return 2 * m.x_WGS['CO2'] == (
#         O_from_CO_conv + O_from_H2O_conv + O_from_steam
#     )

# m.Cons_31 = Constraint(rule=oxygen_balance_wgs)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 32 • Residual molar flow of CO in WGS outlet [mol/s]
# # ────────────────────────────────────────────────────────────────────────────────
# def CO_wgs_mol_rule(m):
#     mol_frac_CO = m.x['CO'] / m.x_total  # dimensionless
#     return m.x_WGS['CO'] == (
#         mol_frac_CO * (1 - m.WGS_C1) *        # unconverted CO fraction
#         m.sng_flow_rate_WGS *                 # kg/s of syngas to WGS
#         1000 / m.MM_gases['CO']                      # kg → mol conversion
#     )

# m.Cons_32 = Constraint(rule=CO_wgs_mol_rule)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 33 • Residual molar flow of H₂O in WGS outlet [mol/s]
# # ────────────────────────────────────────────────────────────────────────────────
# def H2O_wgs_mol_rule(m):
#     mol_frac_H2O = m.x['H2O'] / m.x_total  # dimensionless
#     return m.x_WGS['H2O'] == (
#         mol_frac_H2O * (1 - m.WGS_C2) *          # unconverted H2O fraction
#         m.sng_flow_rate_WGS *                   # kg/s of syngas to WGS
#         1000 / m.MM_gases['H2O']                       # kg → mol conversion
#     )

# m.Cons_33 = Constraint(rule=H2O_wgs_mol_rule)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 34 • Total molar flow rate of WGS outlet [mol/s]
# # ────────────────────────────────────────────────────────────────────────────────
# def wgs_mol(m):
#     return m.x_total_WGS == sum(
#         m.x_WGS[comp] for comp in ("H2", "CO2", "CO", "H2O")
#     )

# m.Cons_34 = Constraint(rule=wgs_mol)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 35  •  Mass flow of H₂ from only WGS outlet  [kg s⁻¹]
# # ────────────────────────────────────────────────────────────────────────────────
# def H2_wgs_kg_rule(m):
#     mol_frac_H2 = m.x_WGS["H2"] / m.x_total_WGS  # dimensionless
#     return m.x_WGS_kg["H2"] == (
#         m.sng_flow_rate_WGS *
#         mol_frac_H2 *
#         m.MM_gases["H2"] / 1000  # g/mol → kg/mol
#     )

# # Register the constraint
# m.Cons_35 = Constraint(rule=H2_wgs_kg_rule)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 36 • Mass flow of CO₂ from combined gasifier and WGS outlet [kg/s]
# # ────────────────────────────────────────────────────────────────────────────────
# def CO2_wgs_gasifier_kg_rule(m):
#     mol_frac_CO2 = m.x_WGS["CO2"] / m.x_total_WGS  # dimensionless
#     return m.x_WGS_kg["CO2"] == (
#         m.sng_flow_rate_WGS *
#         mol_frac_CO2 *
#         m.MM_gases["CO2"] / 1000  # g/mol → kg/mol
#     )

# m.Cons_36 = Constraint(rule=CO2_wgs_gasifier_kg_rule)


# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 37  •  Mass‑flow of residual CO from WGS outlet  [kg s⁻¹]
# # ────────────────────────────────────────────────────────────────────────────────
# def CO_wgs_kg_rule(m):

#     # Inlet mol‑fraction of CO to WGS (from raw syngas)
#     mol_frac_CO_in = m.x["CO"] / m.x_total

#     # Fraction that *remains* un‑converted
#     mol_frac_CO_residual = (1 - m.WGS_C1) * mol_frac_CO_in

#     # Convert to kg s⁻¹
#     return (
#         m.x_WGS_kg["CO"]
#         == m.sng_flow_rate_WGS        # kg s⁻¹ syngas to WGS
#         * mol_frac_CO_residual               # residual mol‑fraction
#         * m.MM_gases["CO"] / m.MM_gases["CO"]  # g mol⁻¹ ratio (=1, kept for form)
#     )

# # Register the constraint
# m.Cons_37 = Constraint(rule=CO_wgs_kg_rule)

# # ────────────────────────────────────────────────────────────────────────────────
# # Constraint 38 • Mass flow of H₂O from WGS outlet [kg/s]
# # ────────────────────────────────────────────────────────────────────────────────
# def H2O_kg_rule(m):
#     mol_frac_H2O_in = m.x['H2O'] / m.x_total
#     unconverted_H2O_frac = mol_frac_H2O_in * (1 - m.WGS_C2)
#     return m.x_WGS_kg['H2O'] == (
#         m.sng_flow_rate_WGS *
#         unconverted_H2O_frac *
#         m.MM_gases['H2O'] / 1000  # convert g/mol → kg/mol for consistent units
#     )

# m.Cons_38 = Constraint(rule=H2O_kg_rule)


# # ------------------------------------------------------------------------------------------
# # Constraint 39 • Mass flow rate of steam supplied to the WGS reactor (kg s⁻¹)
# # ------------------------------------------------------------------------------------------
# def steam_kg_rule(m):
#     return m.steam_kg_wgs == m.steam_mol_wgs * m.MM_gases['H2O'] / 1000

# m.Cons_39 = Constraint(rule=steam_kg_rule)

# # ------------------------------------------------------------------------------------------
# # Constraint 40 • Reaction heat recovered from the Water-Gas Shift (WGS) reactor (kW)
# # ------------------------------------------------------------------------------------------
# def heat_recovery_wgs_rule(m):
#     return m.heat_recovered_wgs == (
#         41.1 *                      # ΔH of WGS (kJ/mol)
#         m.WGS_C1 *          # CO conversion fraction
#         (m.x['CO'] / m.x_total) *   # Molar fraction of CO
#         m.sng_flow_rate_WGS *              # Syngas flow to WGS (kg/s)
#         1000 / m.MM_gases['CO'] *  # Convert kg/s → mol/s
#         m.wgs_heat_recovery_eff    # Efficiency of heat recovery
#     )

# m.Cons_40 = Constraint(rule=heat_recovery_wgs_rule)

# # --------------------------------------------------------------------------------------------
# # Constraint 41 • Heat extracted from syngas during cooling prior to WGS reactor (kW)
# # --------------------------------------------------------------------------------------------
# def syngas_cooling_rule(m):
#     return m.heat_syngas_to_wgs == (
#         m.x_kg_total *       # Syngas mass flow rate (kg/s)
#         m.Cp_syngas *               # Specific heat (kJ/kg·K)
#         (m.T_opt - m.T_wgs) *# Temperature drop (K)
#         m.syngas_cooler_eff         # Heat exchanger effectiveness
#     )

# m.Cons_41 = Constraint(rule=syngas_cooling_rule)

# # ------------------------------------------------------------------
# # Constraint 42  •  Hydrogen Demand Satisfaction (kg/h)
# # ------------------------------------------------------------------
# def hydrogen_demand_constraint(m):
#     return (
#         (m.x_WGS_kg['H2'] + m.x_kg['H2']) * 3600
#         >= m.demand_H2
#     )

# m.Cons_42 = Constraint(rule=hydrogen_demand_constraint)

# # ------------------------------------------------------------------
# # Constraint 43  •  Gasifier Capacity (t/h)
# # ------------------------------------------------------------------
# def gasifier_capacity_rule(m):
#     return m.gasifier_capacity == (m.Max_Feed * 3600) / 1000

# m.Cons_43 = Constraint(rule=gasifier_capacity_rule)

# # ------------------------------------------------------------------
# # Constraint 44  •  Gasifier Capital Cost (USD)
# # ------------------------------------------------------------------
# def gasifier_cost_rule(m):
#     return m.gasifier_cost == m.base_cost_gasifier * \
#            (m.gasifier_capacity / 27.500) ** m.scaling_exponent * \
#            (m.CEPCI_current / m.CEPCI_base)

# m.Cons_44 = Constraint(rule=gasifier_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 45  •  Syngas Cleaning Capacity Tracking (kg h⁻¹)
# # ------------------------------------------------------------------
# def syngas_capacity_tracking(m):
#     return m.syngas_max_flow >= m.x_kg_total * 3600  # [kg/h]

# m.Cons_45 = Constraint(rule=syngas_capacity_tracking)

# # ------------------------------------------------------------------
# # Constraint 46  •  Gas Cleaning Capacity (t h⁻¹)
# # ------------------------------------------------------------------
# def gas_cleaning_capacity_rule(m):
#     return m.gas_cleaning_capacity >= m.syngas_max_flow / 1000  # [t/h]

# m.Cons_46 = Constraint(rule=gas_cleaning_capacity_rule)

# # ------------------------------------------------------------------
# # Constraint 47  •  Gas Cleaning Capital Cost (USD)
# # ------------------------------------------------------------------
# def cleaning_cost_rule(m):
#     return m.sng_cleaning_cost == m.base_cost_gas_cleaner * \
#            (m.gas_cleaning_capacity / 28.930) ** m.scaling_exponent * \
#            (m.CEPCI_current / m.CEPCI_base)

# m.Cons_47 = Constraint(rule=cleaning_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 48  •  CHP Engine Sizing (Nominal Capacity in kW)
# # ------------------------------------------------------------------
# def chp_capacity_rule(m):
#     return m.chp_capacity == m.power_CHP_engine

# m.Cons_48 = Constraint(rule=chp_capacity_rule)

# # ------------------------------------------------------------------
# # Constraint 49  •  CHP Engine Capital Cost (USD)
# # ------------------------------------------------------------------
# def chp_cost_rule(m):
#     return m.chp_cost == m.base_cost_chp * \
#            (m.chp_capacity / 65) ** m.scaling_exponent * \
#            (m.CEPCI_current / m.CEPCI_base)

# m.Cons_49 = Constraint(rule=chp_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 50  •  WGS Reactor Sizing (Capacity in t/h)
# # ------------------------------------------------------------------
# def wgs_capacity_rule(m):
#     return m.wgs_capacity == m.sng_flow_rate_WGS * (m.x['CO'] / m.x_total) * 3600 / 1000

# m.Cons_50 = Constraint(rule=wgs_capacity_rule)

# # ------------------------------------------------------------------
# # Constraint 51  •  WGS Reactor Capital Cost (USD)
# # ------------------------------------------------------------------
# def wgs_cost_rule(m):
#     return m.wgs_cost == m.base_cost_wgs * \
#            (m.wgs_capacity / 32.970) ** m.scaling_exponent * \
#            (m.CEPCI_current / m.CEPCI_base)

# m.Cons_51 = Constraint(rule=wgs_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 52 • Capacity = sep_max_flow converted to tons/hour
# # ------------------------------------------------------------------
# def separator_capacity_rule(m):
#     return m.separator_capacity == Max_Demand / 1000  # t/h

# m.Cons_52 = Constraint(rule=separator_capacity_rule)

# # ------------------------------------------------------------------
# # Constraint 53  •  Separator Capital Cost (USD)
# # ------------------------------------------------------------------
# def separator_cost_rule(m):
#     return m.separator_cost == m.base_cost_separator * \
#            (m.separator_capacity / 5) ** m.scaling_exponent * \
#            (m.CEPCI_current / m.CEPCI_base)

# m.Cons_53 = Constraint(rule=separator_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 54  •  Total Capital Investment Cost (USD)
# # ------------------------------------------------------------------
# def total_capital_cost_rule(m):
#     return m.total_capital_cost == (
#         m.gasifier_cost +
#         m.sng_cleaning_cost +
#         m.chp_cost +
#         m.wgs_cost +
#         m.separator_cost -
#         m.investment
#     )

# m.Cons_54 = Constraint(rule=total_capital_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 55  •  Annual Revenue (USD/year)
# # ------------------------------------------------------------------
# def annual_revenue_rule(m):
#     revenue_H2 = m.hydrogen_price * (m.x_WGS_kg['H2'] + m.x_kg['H2']) * 3600 * 24 * 365

#     revenue_CO = m.CO_price * m.x_WGS_kg['CO'] * 3600 * 24 * 365

#     revenue_CO2 = m.CO2_price * m.x_WGS_kg['CO2'] * 3600 * 24 * 365
    
#     revenue_electricity = m.electricity_price * ((m.Elec_power_total * m.elec_eff_CHP_engine) - m.gasifier_elec_power_req) * 1 * 24 * 365

#     return m.annual_revenue == revenue_H2 + revenue_CO + revenue_CO2 + revenue_electricity

# m.Cons_55 = Constraint(rule=annual_revenue_rule)

# # ------------------------------------------------------------------
# # Constraint 56  •  Annual Biomass Feedstock Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_biomass_cost_rule(m):
#     return m.annual_biomass_cost == m.biomass_cost * m.biomass_feed * 3600 * 24 * 365

# m.Cons_56 = Constraint(rule=annual_biomass_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 57  •  Annual Grid Electricity Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_power_cost_rule(m):
#     return m.annual_power_cost == m.grid_electricity_price * m.grid_power_req * 24 * 365

# m.Cons_57 = Constraint(rule=annual_power_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 58  •  Annual Steam Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_steam_cost_rule(m):
#     return m.annual_steam_cost == m.steam_cost * m.steam_kg_wgs * 3600 * 24 * 365

# m.Cons_58 = Constraint(rule=annual_steam_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 59  •  Annual Labor Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_labor_cost_rule(m):
#     return m.annual_labor_cost == m.labor_cost_factor * (
#         m.annual_biomass_cost + m.annual_steam_cost
#     )

# m.Cons_59 = Constraint(rule=annual_labor_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 60  •  Annual Maintenance Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_maintenance_cost_rule(m):
#     return m.annual_maintenance_cost == m.maintenance_factor * m.total_capital_cost

# m.Cons_60 = Constraint(rule=annual_maintenance_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 61  •  Annual Insurance and Tax Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_insurance_cost_rule(m):
#     return m.annual_insurance_cost == m.insurance_tax_factor * m.total_capital_cost

# m.Cons_61 = Constraint(rule=annual_insurance_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 62  •  Annual Waste Treatment Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_waste_cost_rule(m):
#     return m.annual_waste_cost == m.waste_factor * m.annual_biomass_cost

# m.Cons_62 = Constraint(rule=annual_waste_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 63  •  Annual Catalyst Replacement Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_catalyst_cost_rule(m):
#     return m.annual_catalyst_cost == m.catalyst_unit_cost * m.wgs_capacity * 1000 * m.catalyst_replacement_rate

# m.Cons_63 = Constraint(rule=annual_catalyst_cost_rule)


# # ------------------------------------------------------------------
# # Constraint 64  •  Total Annual Operating Cost (USD/year)
# # ------------------------------------------------------------------
# def annual_operating_cost_rule(m):
#     return m.annual_operating_cost == (
#         m.annual_biomass_cost +
#         m.annual_steam_cost +
#         m.annual_labor_cost +
#         m.annual_maintenance_cost +
#         m.annual_insurance_cost +
#         m.annual_waste_cost +
#         m.annual_catalyst_cost
#     )

# m.Cons_64 = Constraint(rule=annual_operating_cost_rule)

# # ------------------------------------------------------------------
# # Constraint 65  •  Annual Cash Flow (USD/year)
# # ------------------------------------------------------------------
# def annual_cash_flow_rule(m):
#     return m.annual_cash_flow == m.annual_revenue - m.annual_operating_cost

# m.Cons_65 = Constraint(rule=annual_cash_flow_rule)

# # ------------------------------------------------------------------
# # Constraint 66  •  Net Present Value (NPV) of Hydrogen Production (kg)
# # ------------------------------------------------------------------
# def NPV_hydrogen_rule(m):
#     annual_h2 = (m.x_WGS_kg['H2'] + m.x_kg['H2']) * 3600 * 24 * 365
    
#     discounted_h2 = sum(
#         annual_h2 / ((1 + m.discount_rate_h2) ** year)
#         for year in range(1, m.project_life + 1)
#     )
#     return m.NPV_hydrogen_prod == discounted_h2


# m.Cons_66 = Constraint(rule=NPV_hydrogen_rule)

# # ------------------------------------------------------------------
# # Constraint 67  •  Net Present Value (NPV) of Total Cash Flow ($)
# # ------------------------------------------------------------------
# def NPV_project_rule(m):
#     discounted_cash_flows = sum(
#         m.annual_cash_flow / ((1 + m.discount_rate_project) ** year)
#         for year in range(1, m.project_life + 1)
#     )
#     return m.NPV == discounted_cash_flows - m.total_capital_cost

# m.Cons_67 = Constraint(rule=NPV_project_rule)

# # ------------------------------------------------------------------
# # Constraint 68  •  Levelised Cost of Hydrogen (LCOH) Calculation ($/kg)
# # ------------------------------------------------------------------
# def Levelised_cost_hydrogen_rule(m):
#     # NPV of Total Costs = sum of discounted annual (CAPEX + OPEX)
#     annualized_capex = m.total_capital_cost / m.project_life

#     npv_total_costs = sum(
#         (annualized_capex + m.annual_operating_cost) / ((1 + m.discount_rate_project) ** year)
#         for year in range(1, m.project_life + 1)
#     )
#     return m.LCOH == npv_total_costs / m.NPV_hydrogen_prod  # LCOH = (NPV of Costs) / (NPV of H2 Production)

# m.Cons_68 = Constraint(rule=Levelised_cost_hydrogen_rule)


# # ----------------------------------------------------------------------
# # 6.  OBJECTIVE  ##########################################
# # ----------------------------------------------------------------------

# m.obj = Objective(expr = m.annual_operating_cost + (m.total_capital_cost / m.project_life), sense=minimize)

# # m.obj = Objective(expr = m.LCOH, sense=minimize)
# # m.obj = Objective(expr = m.NPV, sense=maximize)

# # ----------------------------------------------------------------------
# # 7.  SOLVER CALL  ##########################################
# # ---------------------------------------------------------------------- 

# # Initialize solver using diffeerent available settings
# # solver = TrustRegionSolver(solver ='ipopt', max_it=10000, trust_radius=12000, sample_radius=1200, algorithm_type=0, reduced_model_type=4, globalization_strategy=0, gamma_e=10, delta_min=1e-4, ep_delta=1e-3) 
# solver = TrustRegionSolver(solver ='ipopt', max_it=10000, trust_radius=11500, sample_radius=1150, algorithm_type=0, reduced_model_type=4, globalization_strategy=1, gamma_e=10, delta_min=1e-4, ep_delta=1e-3, eta=1e-4) 

# # Define an external function list (eflist) as needed
# eflist = [bb2,bb3,bb4,bb5]

# # If using default solver settings, you need to give a solver name available in gams
# # solver.config['solver_options']['solver'] = 'conopt'

# # # Solve the model using TrustRegionSolver
# # solver.solve(m, eflist)

# # # Display the solution
# # m.display()

# # Open the file and redirect stdout safely
# filename = f"Model_BHP_PFS_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}_GS{solver.config['globalization strategy']}.txt"

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




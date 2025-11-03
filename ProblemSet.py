# -*- coding: utf-8 -*-
"""

Problem Set: Models 1-25
Engineering Problems: Models 26-30

@author: gh00616
"""

#################################### Model 1 ###############################################
# Model 1 (Bieglers Example)
# Define the optimization model
model = ConcreteModel()
model.x = Var(range(2), domain=Reals, initialize=0.)

# Define an external function
def blackbox(a):
    return (a**3) + (a**2) - a

bb = ExternalFunction(blackbox)

# Define the objective and constraints
model.obj = Objective(
    expr=(model.x[0])**2 + (model.x[1])**2)

model.c1 = Constraint(expr=bb(model.x[0]) + model.x[0] + 1 == model.x[1])


# Initialize the TrustRegionSolver with necessary configurations (hints: ep_s=1e-3, delta_min=1e-5, config.ep_i, delta_min=1e-4, ep_i=1e-4, delta_min=0.1e-4, ep_i=1e-5, ... delta_min=1e-3, ep_delta=1e-2, ep_i=1e-3, ep_s=4.6e-3) 
solver = TrustRegionSolver(solver ='ipopt', max_it=500, trust_radius=10, sample_radius=1, algorithm_type=1, reduced_model_type=0, gamma_e=10, gamma_c=0.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_1_Biegler_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

######################################## Model 2 ###############################################
# Model 2 (Eason's Example), it is solved using default values of tuning parameter, see notes for the default values

# Define the optimization model
model = ConcreteModel()
model.x = Var(range(5), domain=Reals, initialize=2.0)
model.x[4] = 1.0

# Define an external function
def blackbox(a, b):
    return sin(a - b)

bb = ExternalFunction(blackbox)

# Define the objective and constraints
model.obj = Objective(
    expr=(model.x[0] - 1.0)**2 + (model.x[0] - model.x[1])**2 + (model.x[2] - 1.0)**2 +
          (model.x[3] - 1.0)**4 + (model.x[4] - 1.0)**6
)

model.c1 = Constraint(expr=model.x[3] * model.x[0]**2 + bb(model.x[3], model.x[4]) == 2 * sqrt(2.0))
model.c2 = Constraint(expr=model.x[2]**4 * model.x[1]**2 + model.x[1] == 8 + sqrt(2.0))

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', algorithm_type=4, reduced_model_type=0)

# Define an external function list (eflist) as needed
eflist = [bb]
    
# Open the file and redirect stdout safely
filename = f"Model_2_Eason_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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


####################################### Model 3 ###############################################
# Model 3 (Yoshio Example)

# Define the optimization model
m = ConcreteModel()
m.x1 = Var(initialize=0)
m.x2 = Var(bounds=(-2.0, None), initialize=0)

def blackbox(a,b):
    return a**2 + b**2
bb = ExternalFunction(blackbox)

m.obj = Objective(expr=(m.x1 - 1) ** 2 + (m.x2 - 3) ** 2 + bb(m.x1, m.x2) ** 2)

m.c1 = Constraint(expr = 2 * m.x1 + m.x2 + 10.0 == bb(m.x1, m.x2))

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=1)

# Define an external function list (eflist) as needed
eflist = [bb]
    
# Open the file and redirect stdout safely
filename = f"Model_3_Yoshio_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

######################################## Model 4 ###############################################
# Model 4 (Rastrigin function)

# Define the optimization model

from math import pi

m = ConcreteModel()

# Define constants and also add the terms in the objective function accordingly
A = 10  # Constant in Rastrigin function
n = 3   # Number of dimensions (you can increase this value for higher dimensions)

# Define decision variables (you can add bounds if necessary)
m.x = Var(range(n), initialize=0, bounds=(-5.12,5.12))

def blackbox(a):
    return a**2 - A*cos(2*pi*a)
bb = ExternalFunction(blackbox)

# Set the objective to minimize the Rastrigin function
m.obj = Objective(expr = A * n + m.x[0]**2 - A*cos(2*pi*m.x[0]) + m.x[1]**2 - A*cos(2*pi*m.x[1]) + bb(m.x[2]))  # Minimize

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-4, ep_delta=1e-3)
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=10, sample_radius=1, algorithm_type=0, reduced_model_type=4, gamma_e=10, criticality_check=0.1)

# Define an external function list (eflist) as needed
eflist = [bb]
    
# Open the file and redirect stdout safely
filename = f"Model_4_Rastrigin_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

######################################## Model 5 ###############################################
# Model 5 (Rosen-Suzuki function)

# Define the optimization model
m = ConcreteModel()

# Define decision variables (you can add bounds if necessary)
m.x = Var(range(4), initialize=0., bounds=(-2,2))

def blackbox1(a,b):
    return 2*a**2 - 21*a + 7*b
bb1 = ExternalFunction(blackbox1)

def blackbox2(a,b):
    return a**2 + 2*b**2
bb2 = ExternalFunction(blackbox2)

# Constraint

m.c1 = Constraint(expr = -(8 - m.x[0]**2 - m.x[1]**2 - m.x[2]**2 - m.x[3]**2 - m.x[0] + m.x[1] - m.x[2] + m.x[3]) <= 0)
m.c2 = Constraint(expr = -(10 - m.x[0]**2 - 2*m.x[1]**2 - bb2(m.x[2], m.x[3]) + m.x[0] + m.x[3]) <= 0)
m.c3 = Constraint(expr = -(5 - 2*m.x[0]**2 - m.x[1]**2 - m.x[2]**2 - 2*m.x[0] + m.x[1] + m.x[3]) <= 0)

# Set the objective to minimize the Rosen-Suzuki function
m.obj = Objective(expr = m.x[0]**2 + m.x[1]**2 + m.x[3]**2 - 5*m.x[0] - 5*m.x[1] + bb1(m.x[2], m.x[3]))  # Minimize

# Initialize the TrustRegionSolver with necessary configurations (hints: ep_s=3.3e-2 for A1S0,A2S0, , ep_s=0.015, ep_i=0.0004, , ep_s=0.04, ep_i=0.09)
solver = TrustRegionSolver(solver ='ipopt', max_it=450, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=4, gamma_e=15, criticality_check=0.2)

# Define an external function list (eflist) as needed
eflist = [bb1, bb2]
    
# Open the file and redirect stdout safely
filename = f"Model_5_Rosen-Suzuki_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

######################################## Model 6 ###############################################
# Model 6 (Toy Hydrology function)

# Define the optimization model
from math import pi

m = ConcreteModel()

# Define decision variables (you can add bounds if necessary)
m.x = Var(range(2), initialize=0, bounds=(0,1))

def blackbox(a):
    return 2*pi*a**2
bb = ExternalFunction(blackbox)

# Constraint
m.c1 = Constraint(expr = 1.5 - m.x[0] - 2*m.x[1] - 0.5*sin(-4*pi*m.x[1] + bb(m.x[0])) <= 0)
m.c2 = Constraint(expr = m.x[0]**2 + m.x[1]**2 - 1.5 <= 0)

m.obj = Objective(expr = sum(m.x[i] for i in range(2)))

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=9, gamma_e=10, criticality_check=0.1)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_6_Toy_Hydrology_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

######################################## Model 7 ###############################################
# Model 7 (Goldstein-Price function)

# Define the optimization model
m = ConcreteModel()

# Variables: x1 and x2 are real numbers with no bounds
m.x1 = Var(initialize=0, bounds=(-2,2))
m.x2 = Var(initialize=-1, bounds=(-2,2))

def blackbox1(a,b):
    return - 14*b + 6*a*b + 3*b**2
bb1 = ExternalFunction(blackbox1)

def blackbox2(a,b):
    return (2*a - 3*b)**2
bb2 = ExternalFunction(blackbox2)

# Objective: Minimize the Goldstein-Price function
m.obj = Objective(expr = (1 + (m.x1 + m.x2 + 1)**2 * (19 - 14*m.x1 + 3*m.x1**2 + bb1(m.x1,m.x2))) * (30 + bb2(m.x1,m.x2) * (18 - 32*m.x1 + 12*m.x1**2 + 48*m.x2 - 36*m.x1*m.x2 + 27*m.x2**2)))

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=4, sample_radius=1, algorithm_type=4, reduced_model_type=4, gamma_e=10, criticality_check=0.2)

# Define an external function list (eflist) as needed
eflist = [bb1, bb2]

# Open the file and redirect stdout safely
filename = f"Model_7_Goldstein-Price_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

######################################## Model 8 ###############################################
# Model 8 (Colville Function, 3 surrogates)

# Define the optimization model
m = ConcreteModel()

# Define decision variables (you can add bounds if necessary)
m.x1 = Var(initialize=80, bounds=(78,102))
m.x2 = Var(initialize=35, bounds=(33,45))
m.x3 = Var(initialize=30, bounds=(27,45))
m.x4 = Var(initialize=30, bounds=(27,45))
m.x5 = Var(initialize=30, bounds=(27,45))


def blackbox2(a,b,c):
    return 0.00002584*a*b - 0.00006663*c*b
bb2 = ExternalFunction(blackbox2)

def blackbox3(a,b,c):
    return 2275.1327*((a*b)**(-1)) - 0.2668*c*((b)**(-1))
bb3 = ExternalFunction(blackbox3)

def blackbox4(a,b,c):
    return 1330.3294*((a*b)**(-1)) - 0.42*c*((b)**(-1))
bb4 = ExternalFunction(blackbox4)


# Constraint
m.c1 = Constraint(expr = bb2(m.x3, m.x5, m.x2) - 0.0000734*m.x1*m.x4 - 1 <= 0)
m.c2 = Constraint(expr = 0.000853007*m.x2*m.x5 + 0.00009395*m.x1*m.x4 - 0.00033085*m.x3*m.x5 - 1 <= 0)
m.c3 = Constraint(expr = bb4(m.x2, m.x5, m.x1) - 0.30586*((m.x2*m.x5)**(-1))*m.x3**2 - 1 <= 0)
m.c4 = Constraint(expr = 0.00024186*m.x2*m.x5 + 0.00010159*m.x1*m.x2 + 0.00007379*m.x3**2 - 1 <= 0)
m.c5 = Constraint(expr = bb3(m.x3, m.x5, m.x1) - 0.40584*((m.x5)**(-1))*m.x4 - 1 <= 0)
m.c6 = Constraint(expr = 0.00029955*m.x3*m.x5 + 0.00007992*m.x1*m.x2 + 0.00012157*m.x3*m.x4 - 1 <= 0)


m.obj = Objective(expr = 5.3578*m.x3**2 + 0.8357*m.x1*m.x5 + 37.2392*m.x1)  # Minimize

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=100, sample_radius=10, algorithm_type=4, reduced_model_type=0, gamma_e=15, criticality_check=0.1)

# Define an external function list (eflist) as needed
eflist = [bb2, bb3, bb4]

# Open the file and redirect stdout safely
filename = f"Model_8_Colville_Function_1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

######################################## Model 9 ###############################################
# Model 9 (Colville Function, 4 surrogates)

# Define the optimization model
m = ConcreteModel()
# Define decision variables (you can add bounds if necessary)
m.x1 = Var(initialize=78, bounds=(78,102))
m.x2 = Var(initialize=33, bounds=(33,45))
m.x3 = Var(initialize=30, bounds=(27,45))
m.x4 = Var(initialize=45, bounds=(27,45))
m.x5 = Var(initialize=37, bounds=(27,45))

def blackbox1(a,b):
    return 0.8357*a*b 
bb1 = ExternalFunction(blackbox1)

def blackbox2(a,b,c):
    return 0.00002584*a*b - 0.00006663*c*b
bb2 = ExternalFunction(blackbox2)

def blackbox3(a,b,c):
    return 2275.1327*((a*b)**(-1)) - 0.2668*c*((b)**(-1))
bb3 = ExternalFunction(blackbox3)

def blackbox4(a,b,c):
    return 1330.3294*((a*b)**(-1)) - 0.42*c*((b)**(-1))
bb4 = ExternalFunction(blackbox4)


# Constraint
m.c1 = Constraint(expr = bb2(m.x3, m.x5, m.x2) - 0.0000734*m.x1*m.x4 - 1 <= 0)
m.c2 = Constraint(expr = 0.000853007*m.x2*m.x5 + 0.00009395*m.x1*m.x4 - 0.00033085*m.x3*m.x5 - 1 <= 0)
m.c3 = Constraint(expr = bb4(m.x2, m.x5, m.x1) - 0.30586*((m.x2*m.x5)**(-1))*m.x3**2 - 1 <= 0)
m.c4 = Constraint(expr = 0.00024186*m.x2*m.x5 + 0.00010159*m.x1*m.x2 + 0.00007379*m.x3**2 - 1 <= 0)
m.c5 = Constraint(expr = bb3(m.x3, m.x5, m.x1) - 0.40584*((m.x5)**(-1))*m.x4 - 1 <= 0)
m.c6 = Constraint(expr = 0.00029955*m.x3*m.x5 + 0.00007992*m.x1*m.x2 + 0.00012157*m.x3*m.x4 - 1 <= 0)

# Set the objective to minimize the colville function
m.obj = Objective(expr = 5.3578*m.x3**2 + bb1(m.x1, m.x5) + 37.2392*m.x1)  # Minimize

# Initialize the TrustRegionSolver with necessary configurations (hints: ep_i=1e-4, ep_s=1)
solver = TrustRegionSolver(solver ='ipopt', max_it=2500, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=0, gamma_e=10, criticality_check=0.1)

# Define an external function list (eflist) as needed
eflist = [bb1, bb2, bb3, bb4]

# Open the file and redirect stdout safely
filename = f"Model_9_Colville_Function_2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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


####################################### Model 10 ###############################################
# Model 10 (Powell's Function)

# Define the optimization model
m = ConcreteModel()

# Define variables x1, x2, x3, x4 with no bounds (they are free variables)
m.x1 = Var(initialize=3, bounds=(-4,5))  # Starting point for x1
m.x2 = Var(initialize=-0.256837, bounds=(-4,5)) # Starting point for x2
m.x3 = Var(initialize=0.517729, bounds=(-4,5))  # Starting point for x3
m.x4 = Var(initialize=2.244261, bounds=(-4,5))  # Starting point for x4

def blackbox1(a,b):
    return (a - 2*b)**4
bb1 = ExternalFunction(blackbox1)

def blackbox2(a,b):
    return 10*(a - b)**4
bb2 = ExternalFunction(blackbox2)

m.obj = Objective(expr=(m.x1 + 10 * m.x2)**2 + 5 * (m.x3 - m.x4)**2 + bb1(m.x2,m.x3) + bb2(m.x1,m.x4))

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=2, gamma_e=10, criticality_check=0.2)

# Define an external function list (eflist) as needed
eflist = [bb1,bb2]

# Open the file and redirect stdout safely
filename = f"Model_10_Powell_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

################################ Model 11 ###############################################
# Model 11 (Gramacy & Lee (2012) Function)

# Define the optimization model
model = ConcreteModel()
model.x = Var(bounds=(0.5,2.5), initialize=1)

# Define an external function
def blackbox(a):
    return sin(10*(22/7)*a)/(2*a)

bb = ExternalFunction(blackbox)


# Define the objective
model.obj = Objective(
    expr=bb(model.x) + (model.x - 1)**4)

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1, sample_radius=0.1, algorithm_type=0, reduced_model_type=1, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_11_Gramacy&Lee_Example_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

################################ Model 12 ###############################################
# Model 12 (Easom Function)

# Define the optimization model
model = ConcreteModel()
model.x = Var(range(2), bounds=(-100,100), initialize=0.)

# Define an external function
def blackbox(a):
    return cos(a)

bb = ExternalFunction(blackbox)


# Define the objective
model.obj = Objective(
    expr= -bb(model.x[0]) * bb(model.x[1]) * exp(-(model.x[0]-(22/7))**2 - (model.x[1]-(22/7))**2)
    )

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=2500, algorithm_type=0, reduced_model_type=4, gamma_e=2.5, ep_compatibility=1e-20, delta_min = 1e-4, ep_delta=1e-3)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_12_Easom_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

################################ Model 13 ###############################################
# Model 13 (Six-Hump Camel Function)

# Define the optimization model
model = ConcreteModel()
model.x1 = Var(bounds=(-3,2), initialize=0.1)
model.x2 = Var(bounds=(-2,2), initialize=0.1)

# Define an external function
def blackbox(a):
    return 4*(a**2) - 2.1*(a**4) + ((a**6)/3)

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(
    expr= bb(model.x1) + (model.x1*model.x2) + (-4*(model.x2**2) + 4*(model.x2**4))
    )

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-1, ep_delta=1, ep_compatibility=1e-6)
solver = TrustRegionSolver(solver ='ipopt', max_it=1500, algorithm_type=4, reduced_model_type=4, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_13_Six-Hump_Camel_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

################################ Model 14 ###############################################
# Model 14 (Three-Hump Camel Function)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-5,5), initialize=0.)

# Define an external function
def blackbox(a):
    return 2*(a**2) - 1.05*(a**4) + ((a**6)/6)

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(
    expr= bb(model.x[0]) + (model.x[0]*model.x[1]) + (model.x[1]**2)
    )

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=1, sample_radius=0.1, algorithm_type=2, reduced_model_type=3, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_14_Three-Hump_Camel_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

################################ Model 15 ###############################################
# Model 15 (McCormick Function)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds={ 
    0: (-1.5, 4), 
    1: (-3, 4)},
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return sin(a+b)

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(
    expr= bb(model.x[0],model.x[1]) + (model.x[0]-model.x[1])**2 - 1.5*model.x[0] + 2.5*model.x[1] + 1
    )

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1...........ep_compatibility=1e-6, criticality_check=0.75, ep_i=1e-2, ep_s=1e-2)
solver = TrustRegionSolver(solver ='ipopt', max_it=300, trust_radius=1100, sample_radius=110, algorithm_type=2, reduced_model_type=3)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_15_McCormick_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

############################## Model 16 ###############################################
# Model 16 (Matyas Function)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-10,10),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return 0.26*(a**2 + b**2)

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(
    expr= bb(model.x[0], model.x[1]) - 0.48*model.x[0]*model.x[1]
    )

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, A4S1, ep_i=1.5e-3, delta_min=1e-3, ep_delta=1e-2, A4S0 ep_i=1.5e-3, delta_min=1e-3, ep_delta=1.5e-3, A3S1,ep_i=1e-3, ep_s=0.011, delta_min=1e-4, ep_delta=1e-3, A3S2, ep_i=1e-2, ep_s=1e-1, delta_min=1e-4, ep_delta=1e-3)
solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=1, gamma_e=10)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_16_Matyas_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

############################## Model 17 ###############################################
# Model 17 (Bohachevsky Formulation 1)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-100,100),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a):
    return 0.3*cos(3*(22/7)*a)

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(
    expr= model.x[0]**2 + 2*model.x[1]**2 - bb(model.x[0]) - 0.4*cos(4*(22/7)*model.x[1]) + 0.7
    )

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4)
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=100, sample_radius=10, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_17_Bohachevsky_Function_1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

############################## Model 18 ###############################################
# Model 18 (Bohachevsky Formulation 2)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-100,100),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return 0.3*cos(3*(22/7)*a)*cos(4*(22/7)*b)

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(
    expr= model.x[0]**2 + 2*model.x[1]**2 - bb(model.x[0],model.x[1]) + 0.3
    )

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1000, sample_radius=100, algorithm_type=4, reduced_model_type=9, gamma_e=2.5, ep_s=1e-3)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_18_Bohachevsky_Function_2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

############################## Model 19 ###############################################
# Model 19 (Bohachevsky Formulation 3)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-100,100),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return 0.3*cos((3*(22/7)*a) + (4*(22/7)*b))

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(
    expr= model.x[0]**2 + 2*model.x[1]**2 - bb(model.x[0],model.x[1]) + 0.3
    )

# Initialize the TrustRegionSolver with necessary configurations (hints: ep_s=1e-4, ep_i=1e-4)
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=1000, sample_radius=100, algorithm_type=4, reduced_model_type=9, gamma_e=2.5, ep_s=1e-4)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_19_Bohachevsky_Function_3_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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


############################## Model 20 ###############################################
# Model 20 (Rosenbrock_Function)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-2,2),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return a**2 + b**2

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(expr=(1 - model.x[0])**2 + 100 * (model.x[1] - model.x[0]**2)**2, sense=minimize)

# Constraints
model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 1)
model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 1)

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=250, trust_radius=2, sample_radius=0.2, algorithm_type=4, reduced_model_type=4, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_20_Rosenbrock_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

############################## Model 21 ###############################################
# Model 21 (Beale's_Function)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-4.5,4.5),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return a**2 + b**2

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(expr=(1.5 - model.x[0] + model.x[0] * model.x[1])**2 +
                        (2.25 - model.x[0] + model.x[0] * model.x[1]**2)**2 +
                        (2.625 - model.x[0] + model.x[0] * model.x[1]**3)**2, sense=minimize)

# Constraints
model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 3)
model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 5)

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=750, trust_radius=45, sample_radius=4.5, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_21_Beale's_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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
    
############################## Model 22 ###############################################
# Model 22 (Booth's_Function)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-10,10),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return a**2 + b**2

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(expr=(model.x[0] + 2 * model.x[1] - 7)**2 + (2 * model.x[0] + model.x[1] - 5)**2, sense=minimize)

# Constraints
model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 50)

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-3, ep_delta=1e-2)
solver = TrustRegionSolver(solver ='ipopt', max_it=1500, trust_radius=10, sample_radius=1, algorithm_type=4, reduced_model_type=0, gamma_e=10)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_22_Booth's_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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


############################## Model 23 ###############################################
# Model 23 (Styblinski-Tang_Function)

# Define the optimization model
model = ConcreteModel()

model.x = Var(range(2), bounds=(-5,5),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return a**2 + b**2

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(expr=(model.x[0]**4 - 16 * model.x[0]**2 + 5 * model.x[0] + model.x[1]**4 - 16 * model.x[1]**2 + 5 * model.x[1]) / 2, sense=minimize)

# Constraints
model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
model.con2 = Constraint(expr=bb(model.x[0],model.x[1]) <= 50)

# Initialize the TrustRegionSolver with necessary configurations
solver = TrustRegionSolver(solver ='ipopt', max_it=2500, trust_radius=1, sample_radius=0.1, algorithm_type=4, reduced_model_type=2, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_23_Styblinski-Tang_Function_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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
    
    
############################## Model 24 ###############################################
# Model 24 (P1)

# Define the optimization model
model = ConcreteModel()

# Variables
model.x = Var(range(2),
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return (a - 1)**2 + b**2

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(expr=bb(model.x[0], model.x[1]), sense=maximize)

# Constraints
model.con1 = Constraint(expr=model.x[0]**2 + 6*model.x[1] - 36 <= 0)
model.con2 = Constraint(expr=-4*model.x[0] + model.x[1]**2 - 2*model.x[1] <= 0)

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4)
solver = TrustRegionSolver(solver ='ipopt', max_it=1000, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_24_P1_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

    
############################## Model 25 ###############################################
# Model 25 (P2)

# Define the optimization model
model = ConcreteModel()

# Variables
model.x = Var(range(2),
    bounds={ 
    0: (0, None), 
    1: (0, 1.8)},
    initialize={
    0: 0,
    1: 0})

# Define an external function
def blackbox(a,b):
    return a**2 + b**2

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(expr=2*model.x[0] - model.x[0]**2 + model.x[1], sense=maximize)

# Constraints
model.con1 = Constraint(expr=bb(model.x[0],model.x[1]) <= 4)

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4)
solver = TrustRegionSolver(solver ='ipopt', max_it=250, algorithm_type=4, reduced_model_type=0, gamma_e=10)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_25_P2_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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




#### Engineering Case Studies ####

############################## Model 26 ###############################################
# Model 26 (Pressure Vessel Design)

# Define the optimization model
model = ConcreteModel()

a = 0.0625

# Variables
model.x1 = Var(bounds=(a, 99*a), initialize=0.7780270626316168)   
model.x2 = Var(bounds=(a, 99*a), initialize=15*a)   
model.x3 = Var(bounds=(10, 200), initialize=40.312283554875876)   
model.x4 = Var(bounds=(10, 200), initialize=200) 

# Define an external function

def blackbox(a,b,c,d):
    return 0.6224*(a*c*d) + 1.7781*b*(c**2) + 3.1661*(a**2)*d + 19.84*(a**2)*c

bb = ExternalFunction(blackbox)

# Define the objective
model.obj = Objective(expr=bb(model.x1,model.x2,model.x3,model.x4), sense=minimize)
# model.obj = Objective(expr=0.6224*(model.x1*model.x3*model.x4) + 1.7781*model.x2*(model.x3**2) + 3.1661*(model.x1**2)*model.x4 + 19.84*(model.x1**2)*model.x3, sense=minimize)

# Constraints
model.c1 = Constraint(expr= - model.x1 + 0.0193*model.x3 <= 0)
model.c2 = Constraint(expr= - model.x2 + 0.0095*model.x3 <= 0)
model.c3 = Constraint(expr= - (22/7)*(model.x3**2)*model.x4 - (4/3)*(22/7)*(model.x3**3) + 1296000 <= 0)
model.c4 = Constraint(expr= model.x4 - 240 <= 0)

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4, , delta_min=1, ep_delta=2)
solver = TrustRegionSolver(solver ='ipopt', max_it=15000, trust_radius=1, sample_radius=0.1, algorithm_type=4, reduced_model_type=0, gamma_e=10)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_26_Pressure_Vessel_Design_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

############################## Model 27 ###############################################
# Model 27 (Liquid_Liquid_Extraction_Column)

# Define the optimization model
model = ConcreteModel()

# PARAMETERS
model.m = Param(default=1.5)       # distribution coefficient

# VARIABLES
model.vx = Var(bounds=(0.05, 0.25), initialize=0.08)  # vx lower bound >0 to avoid division by zero # Raffinate velocity
model.vy = Var(bounds=(0.05, 0.30), initialize=0.10)  # vy lower bound >0 # Extract velocity
model.F = Var(bounds=(0, None), initialize=1.2) # Extraction factor F
model.Y0 = Var(bounds=(0, None), initialize=1.3862070832433895) # Extract outlet concentration
# model.Nox = Var(bounds=(0, None), initialize=4.559179311599174) # Experimental number of transfer units

# Define an external function
# number of transfer units # A black-box because it is calculated from experimental data
def blackbox(a,b):
    return 4.81*((a/b)**0.24)

bb = ExternalFunction(blackbox)

# OBJECTIVE: maximize vy * Y0
model.obj = Objective(expr = model.vy * model.Y0, sense=maximize)

# CONSTRAINTS
# Define F = m * vx / vy
model.F_definition = Constraint(expr = model.F == model.m * model.vx / model.vy)


# Define Y0 using nonlinear constraint
model.Y0_definition = Constraint(
    expr = model.Y0 * (1 - model.F * exp(bb(model.vx, model.vy) * (1 - model.F))) ==
          model.F * (1 - exp(bb(model.vx, model.vy) * (1 - model.F)))
)

# (Optional) Constraints
# Example flooding constraint (you can modify or add based on detailed flooding curve)
model.flooding = Constraint(expr = model.vx + model.vy <= 0.20)


# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4,gamma_e=8.35,9.99)
solver = TrustRegionSolver(solver ='ipopt', max_it=100, algorithm_type=4, reduced_model_type=0, gamma_e=8.35)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_27_Liquid_Liquid_Extraction_Column_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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
    
    
############################## Model 28 ###############################################
# Model 28 (Himmelblau’s Problem)

# Define the optimization model
model = ConcreteModel()

# === VARIABLES ===
model.x1 = Var(bounds=(78, 102), initialize=100)   
model.x2 = Var(bounds=(33, 45), initialize=40)   
model.x3 = Var(bounds=(27, 45), initialize=40)   
model.x4 = Var(bounds=(27, 45), initialize=35)   
model.x5 = Var(bounds=(27, 45), initialize=30)

model.g1 = Var(bounds=(0, 92), initialize=30)
model.g2 = Var(bounds=(90, 110), initialize=100)
model.g3 = Var(bounds=(20, 25), initialize=20)

# Define an external function
def blackbox1(a):
    return a**2

bb1 = ExternalFunction(blackbox1)

# Define an external function
def blackbox2(a,b):
    return a*b

bb2 = ExternalFunction(blackbox2)

# === OBJECTIVE ===
model.obj = Objective(expr=5.3578547*bb1(model.x3) + 0.8356891*model.x1*model.x5 + 37.2932239*model.x1 - 40792.141, sense=minimize)

# === CONSTRAINTS ===
model.c1 = Constraint(expr= model.g1 == 85.334407 + 0.0056858*bb2(model.x2,model.x5) + 0.00026*model.x1*model.x4 - 0.0022053*model.x3*model.x5)
model.c2 = Constraint(expr= model.g2 == 80.51249 + 0.0071317*bb2(model.x2,model.x5) + 0.0029955*model.x1*model.x2 - 0.0021813*(model.x3**2))
model.c3 = Constraint(expr= model.g3 == 9.300961 + 0.0047026*model.x3*model.x5 + 0.0012547*model.x1*model.x3 - 0.0019085*model.x3*model.x4)



# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-6, ep_s=1e-2, ep_i=1e-4,gamma_e=8.35,9.99, , ep_s=1e-2, ep_i=1e-3)
solver = TrustRegionSolver(solver ='ipopt', max_it=1000, trust_radius=10000, sample_radius=1000, algorithm_type=0, reduced_model_type=4, gamma_e=15)

# More hints 
# for A3,S3, GP(2)
# A5,S4 and S3 = trust_radius=10, sample_radius=2.4, ep_s=1e0
# 2/1 ratio in tr radius and sampling radius
# Define an external function list (eflist) as needed

eflist = [bb1, bb2]

# Open the file and redirect stdout safely
filename = f"Model_28_Himmelblau_Problem_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

############################## Model 29 ###############################################
# Model 29 (Spring Design)

# Define the optimization model
model = ConcreteModel()

# Decision variables
model.x1 = Var(bounds=(0.05, 2.0), initialize=1.0)     # Wire diameter
model.x2 = Var(bounds=(0.25, 1.3), initialize=0.5)     # Mean coil diameter
model.x3 = Var(bounds=(2.0, 15.0), initialize=5.0)     # Number of active coils

# Define an external function

def blackbox(a,b,c):
    return (c + 2) * b * a**2

bb = ExternalFunction(blackbox)

# Objective function: Minimize weight (x3 + 2)x2*x1^2
model.obj = Objective(expr=bb(model.x1,model.x2,model.x3), sense=minimize)

# Constraints
model.g1 = Constraint(expr=1 - (model.x2**3 * model.x3) / (71785 * model.x1**4) <= 0)

model.g2 = Constraint(
    expr=((4 * model.x2**2 - model.x1 * model.x2) / (12566 * ((model.x2 * model.x1**3) - (model.x1**4)))) + (1 / (5108 * model.x1**2)) - 1 <= 0)

model.g3 = Constraint(expr=1 - (140.45 * model.x1 / (model.x2**2 * model.x3)) <= 0)

model.g4 = Constraint(expr=((model.x1 + model.x2)/1.5) - 1 <= 0)

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1, ep_compatibility=1e-15)
solver = TrustRegionSolver(solver ='ipopt', max_it=15000, algorithm_type=4, reduced_model_type=0, gamma_e=2.5)

# Define an external function list (eflist) as needed
eflist = [bb]

# Open the file and redirect stdout safely
filename = f"Model_29_Spring_Design_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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
    
    
############################## Model 30 ###############################################
# Model 30 (Alkylation Process, 1 surrogates)

# Define the optimization model
m = ConcreteModel()

# Variables
m.x = Var(RangeSet(0,9), bounds={ 
    0: (0, 2000), 
    1: (0, 16000),
    2: (0, 120),
    3: (0, 5000),
    4: (0, 2000),
    5: (85, 93),
    6: (90, 95),
    7: (5.69, 12),
    8: (1.2, 4),
    9: (145, 162)
}, initialize={
    0: 1309.276241883202,
    1: 6210.726557172016,
    2: 120.0,
    3: 2088.796312316891,
    4: 1239.055259143405,
    5: 92.99796451282054,
    6: 92.66666666666667,
    7: 5.69,
    8: 3.090000000000005,
    9: 145.0
})

def blackbox1(a):
    return (1.12 + (0.12167*a) - (0.0067 * (a**2)))

bb1 = ExternalFunction(blackbox1)

# Constraints
m.c1 = Constraint(expr = m.x[3] == m.x[0] * bb1(m.x[7]))
m.c2 = Constraint(expr = m.x[6] == 86.35 + (1.098 * m.x[7]) - (0.038 * (m.x[7]**2)) + (0.325 * (m.x[5] - 89)))
m.c3 = Constraint(expr = m.x[8] == 35.28 - (0.222 * m.x[9]))
m.c4 = Constraint(expr = m.x[9] == (3 * m.x[6]) - 133)
m.c5 = Constraint(expr = m.x[7] * m.x[0] == m.x[1] + m.x[4])
m.c6 = Constraint(expr = m.x[4] == (1.22 * m.x[3]) - m.x[0])
m.c7 = Constraint(expr = (m.x[5] * m.x[3] * m.x[8]) + (m.x[5] * (1000 * m.x[2])) == 98000 * m.x[2])


# Objective
m.obj = Objective(expr = ((0.063 * m.x[3] * m.x[6]) - (5.04 * m.x[0]) - (0.035 * m.x[1]) - (10 * m.x[2]) - (3.36 * m.x[4])), sense=maximize)

# Initialize the TrustRegionSolver with necessary configurations (hints: delta_min=1e-2, ep_delta=1e-1)
solver = TrustRegionSolver(solver ='ipopt', max_it=10000, algorithm_type=4, reduced_model_type=1, gamma_e=13) 

# Define an external function list (eflist) as needed
eflist = [bb1]

# Open the file and redirect stdout safely
filename = f"Model_30_Alkylation_Process_A{solver.config['algorithm type']}_S{solver.config['reduced model type']}.txt"

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

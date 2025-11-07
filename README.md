# TRF Solver
The Trust-Region Filter (TRF) solver was developed using concepts from nonlinear optimisation, derivative-free optimisation, and surrogate modelling. It allows users to solve grey-box optimisation problems in which glass-box parts of the problem are modelled with fully specified equation-based models (with available derivative information) and black-box parts of the problem lack analytic derivative information (i.e., only function evaluations are available via external calls). This trust-region-based method optimises the grey-box problem using the accurate surrogate models constructed with the help of external black-box evaluations, thus avoiding the direct implementation of the computationally expensive black-box models. The filter serves as the globalisation strategy in the method. The optimisation is performed iteratively using the TRF solver, resulting in fewer calls to the computationally costly external black box functions.

Please refer to the manuscript (https://doi.org/10.48550/arXiv.2509.01651) for more details.

The user is required to provide black-box external functions and the glass-box model. The default values of tuning parameters for the TRF algorithm are implemented; however, the user is sometimes required to change them (via the solver's options) based on the sensitivity analysis. 

The solver is implemented in Python using the Pyomo modelling language. The required Python packages are specified in both requirements.txt (for pip users) and environment.yml (for conda users). A nonlinear programming (NLP) solver (such as IPOPT) is required to run the TRF solver.

# Installation
- Download all files in this repository.
- Download/install and open Anaconda Navigator.
- Create a new environment using either:
  a. environment.yml (recommended for conda), or
  b. requirements.txt (for pip users in a virtualenv).
- Select Python 3.8.20 when setting up the environment.
- Activate the environment.
- Open Spyder IDE, then open the RunFile.py file.
- Scroll to the indicated section (at the end: line 269 and onwards) and add your grey box optimisation code.

# Running
- A sample grey box optimisation setup is included in the RunFile.py file between lines 269 and 311.
- Additional problems (used in the manuscript: https://doi.org/10.48550/arXiv.2509.01651) are provided in the ProblemSet.py file and can be copied into the RunFile.py file.
- Different solver options (such as surrogate forms, algorithmic variants introduced in the manuscript: https://doi.org/10.48550/arXiv.2509.01651, initial trust radius, maximum iterations, tolerances, etc.) can be set; a few are set as examples in the sample grey-box optimisation setup.
- While optimising, the solver will display progress and results on Spyder's console via logger.
- The iteration-wise solution (text file) will also be saved in the working directory.

# Benchmarking
- Download the Benchmarking.zip file, and extract it to a directory.
- There will be five ".py" files for five different black-box/derivative-free optimisation solvers in the extracted folder. Each ".py" file contains code for 25 problems.
- Create a new environment using either:
  a. Benchmarking.yml (recommended for conda), or
  b. requirements.txt (for pip users in a virtualenv).
- Activate the environment.
- Open Spyder IDE, then open one of the ".py" files. For example, open Benchmarking-COBYLA.py and uncomment the formulation/code of problem 1 (i.e., P1). Run the code. Repeat the process for all 25 problems to reproduce benchmarking results for each DFO solver.

Have feedback or questions? Please email: gulhameed361@gmail.com.
Thanks for using the TRF Solver — enjoy optimising!
Stay tuned for more updates.

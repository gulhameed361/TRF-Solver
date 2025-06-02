# TRF Solver
The Trust-Region Filter (TRF) solver was developed using concepts from nonlinear optimisation, derivative-free optimisation, and surrogate modelling. It allows users to solve grey box optimisation problems in which glass box parts of the problem are modelled with open, equation-based models (with available derivative information) and black box parts of the problem lack derivative information (i.e., only function evaluations are possible). This trust-region-based method utilises surrogate models constructed using external black box evaluations, thus avoiding the direct implementation of the computationally expensive black box models. This is done iteratively using the TRF solver, resulting in fewer calls to the computationally costly external black box functions.

Please refer to the manuscript (--------) for more details.

The user is required to provide black-box external functions and the glass-box model. The default values of tuning parameters for the TRF algorithm are implemented, however, the user is sometimes required to change them (via the solver's options) based on sensitivity analysis. 

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
- A sample grey box optimisation setup is included in RunFile.py file between lines 269 and 311.
- Additional problems (used in the manuscript) are provided in ProblemSet.py file and can be copied into RunFile.py file.
- While optimising, the solver will display progress and results on Spyder's console.
- The iteration-wise solution (text file) will also be saved in the working directory.

Have feedback or questions? Please email: gulhameed361@gmail.com.
Thanks for using the TRF Solver — enjoy optimising!


# TRF-Solver
The Trust-Region Filter (TRF) solver was developed using concepts from nonlinear optimisation, derivative-free optimisation, and surrogate modelling. It allows users to solve grey box optimisation problems in which glass box parts of the problem are modelled with open, equation-based models (with available derivative information) and black box parts of the problem lack derivative information (i.e., only function evaluations are possible). This trust-region-based method utilises surrogate models that are constructed using external black box evaluations, thus avoiding the direct implementation of the computationally expensive black box models. This is done iteratively, resulting in fewer calls to the computationally expensive external black box functions.
The original grey box problem can be formulated as follows:
minimize     f(x)
subject to   h(x) = 0
             g(x) ≤ 0
             y = d(w)
where:
x is a set of all inputs and surrogate outputs
w are the inputs to the external black box function
d(w) are the outputs of the external functions as a function of w
f, h, g, and d are all assumed to be twice continuously differentiable

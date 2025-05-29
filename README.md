# TRF-Solver
The Trust-Region Filter (TRF) solver was developed using concepts from nonlinear optimisation, derivative-free optimisation, and surrogate modelling. It allows users to solve grey box optimisation problems in which glass box parts of the problem are modelled with open, equation-based models (with available derivative information) and black box parts of the problem lack derivative information (i.e., only function evaluations are possible). This trust-region-based method utilises surrogate models that are constructed using external black box evaluations, thus avoiding the direct implementation of the computationally expensive black box models. This is done iteratively, resulting in fewer calls to the computationally expensive external black box functions.
\documentclass{article}
\usepackage{amsmath, amssymb}

\begin{document}

\section*{Grey-box Optimization Problem}

The grey-box optimisation problem is formulated as:
\begin{equation}
\begin{aligned}
    \min_{z, w} \quad & f(z, w, d(w)) \\
    \text{subject to} \quad & h(z, w, d(w)) = 0, \\
                            & g(z, w, d(w)) \leq 0,
\end{aligned}
\tag{1}
\end{equation}

where \( z \in \mathbb{R}^n \) and \( w \in \mathbb{R}^m \) are decision variables. The vector \( w \) represents inputs to the black-box function \( d(w) : \mathbb{R}^m \rightarrow \mathbb{R}^p \), and \( z \) includes the remaining decision variables. The objective function and constraints—\( f \), \( h \), \( g \), and \( d \)—are assumed to be continuously differentiable on the domain \( \mathbb{R}^{m+n+p} \), although derivative information for the black-box function \( d(w) \) is unavailable.

To simplify formulation \eqref{1}, the glass-box and black-box components are decoupled by introducing \( y \in \mathbb{R}^p \) as an explicit constraint, reformulating the problem as:

\begin{equation}
\begin{aligned}
    \min_{x} \quad & f(x) \\
    \text{subject to} \quad & h(x) = 0, \\
                            & g(x) \leq 0, \\
                            & y = d(w),
\end{aligned}
\tag{2}
\end{equation}

where \( x^\top = [w^\top, y^\top, z^\top] \).

The Trust Region Framework (TRF) method iteratively generates a sequence of points \( \{x\} \) that converges to a first-order Karush-Kuhn-Tucker (KKT) point of the original grey-box optimisation problem \eqref{1}, ensuring feasibility of both the glass-box constraints (\( h(x) = 0 \), \( g(x) \leq 0 \)) and the black-box constraint (\( y = d(w) \)), while minimising the objective \( f(x) \) with minimal calls to the external black-box function evaluations.

\end{document}


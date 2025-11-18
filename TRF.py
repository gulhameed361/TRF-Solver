#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from math import pow

import time

from pyomo.common.dependencies import numpy as np

from filterMethod import (
    FilterElement, Filter)
from funnelMethod import Funnel
from helper import (cloneXYZ, packXYZ)
from Logger import Logger
from PyomoInterface import (
    PyomoInterface, ROMType, ALGType)

def TRF(m, eflist, config):
    """The main function of the Trust Region Filter algorithm

    m is a PyomoModel containing ExternalFunction() objects Model
    requirements: m is a nonlinear program, with exactly one active
    objective function.

    eflist is a list of ExternalFunction objects that should be
    treated with the trust region

    config is the persistent set of variables defined 
    in the ConfigBlock class object

    Return: 
    model is solved, variables are at optimal solution or
    other exit condition.  model is left in reformulated form, with
    some new variables introduced in a block named "tR" TODO: reverse
    the transformation.
    """

    logger = Logger()
    filteR = Filter()
    problem = PyomoInterface(m, eflist, config)
    x, y, z = problem.getInitialValue()
    iteration = -1

    romParam, yr, FEs = problem.buildROM(x, config.sample_radius)
    #y = yr
    rebuildROM = False
    
    ROMAccuracy = False
    
    xk, yk, zk = cloneXYZ(x, y, z)
    chik = 1e8
    EV = None
    FEs = None
    thetak = np.linalg.norm(yr - yk,1)
    IT = None
    objk = problem.evaluateObj(x, y, z)
    
    stepNorm = 1e10
    
    funnel = Funnel(phi_init=thetak,
                f_best_init=objk,
                phi_min=config.phi_min,
                kappa_f=config.kappa_f,
                kappa_r=config.kappa_r,
                alpha=config.alpha,
                beta=config.beta,
                mu_s=config.mu_s,
                eta=config.eta)
    
    while True:
        if iteration >= 0:
            logger.printIteration(iteration)
            #print(xk)
        start_time = time.time()
        # increment iteration counter
        iteration = iteration + 1
        if iteration > config.max_it:
            print("EXIT: Maxmium iterations\n")
            break

        ######  Why is this here ###########
        if iteration == 1:
            config.sample_region = False
        ################################

        # Keep Sample Region within Trust Region
        if config.trust_radius < config.sample_radius:
            config.sample_radius = max(
                config.sample_radius_adjust*config.trust_radius,
                config.delta_min)
            rebuildROM = True

        #Generate a RM r_k (x) that is kappa-fully linear on sigma k
        if(rebuildROM):
            #TODO: Ask Jonathan what variable 1e-3 should be
            if config.reduced_model_type == 5 or config.reduced_model_type == 6 or config.reduced_model_type == 7:
                if config.trust_radius < 1e-1:
                    problem.romtype = ROMType.ts
                # Added this switching / Gul
                else:
                    problem.romtype = config.reduced_model_type
            elif config.reduced_model_type == 0 or config.reduced_model_type == 1 or config.reduced_model_type == 2:
                if config.trust_radius < 1e-3:
                    problem.romtype = ROMType.linear
                # Added this switching / Gul
                elif config.trust_radius >= 1e-3 and config.trust_radius < 1e-1:
                    problem.romtype = ROMType.quadratic_simp
                else:
                    problem.romtype = config.reduced_model_type
            elif config.reduced_model_type == 3 or config.reduced_model_type == 4 or config.reduced_model_type == 8 or config.reduced_model_type == 9:
                problem.romtype = config.reduced_model_type

            romParam, yr, FEs = problem.buildROM(x, config.sample_radius)
            
            FEs = FEs + (1*len(romParam))
        else:
            FEs = 0 + (1*len(romParam))
        
        g, J, H, varlist, conlist, HM, EV, H_abs, EV_abs, H_clamped, EV_clamped = problem.grad_hess_calc(x, y, z, romParam)
        
        if config.algorithm_type == 0:
            problem.algtype == ALGType.Normal_TR
        elif config.algorithm_type == 1:
            problem.algtype == ALGType.Simple_Diagonal_Loading
            x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(HM)
            x_eig, y_eig, z_eig = problem.create_ordered_eigenvalue_lists(EV)
        elif config.algorithm_type == 2:
            problem.algtype == ALGType.Hessian_Clamped_Eigenvalues
            x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_clamped)
        elif config.algorithm_type == 3:
            problem.algtype == ALGType.Hessian_Absolute_Eigenvalues
            x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_abs)
        elif config.algorithm_type == 4:
            if ROMAccuracy == False:
                problem.algtype == ALGType.Hessian_Absolute_Eigenvalues
                x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_abs)
            elif ROMAccuracy == True:
                problem.algtype == ALGType.Hessian_Clamped_Eigenvalues
                x_hess_rows, y_hess_rows, z_hess_rows = problem.create_ordered_hessian_rows_lists(H_clamped)
        else:
            raise('The selected variant of the algorithm is not supported!')
            
        
        # print(x_hess_rows)
        # print(y_hess_rows)
        # print(z_hess_rows)
        
        # print(x_eig)
        # print(y_eig)
        # print(z_eig)
        
        # Criticality Check
        if iteration > 0:
            flag, chik = problem.criticalityCheck(x, y, z, romParam, g, J, varlist, conlist)
            # print(flag)
            # print(chik)
            # print(EV)
            if (not flag):
                raise Exception("Criticality Check fails!\n")

        # Save the iteration information to the logger
        logger.newIter(iteration,xk,yk,zk,thetak,objk,chik,list(EV.values()),FEs,IT,
                       config.print_variables)

        # Check for Termination
        if (thetak < config.ep_i and
            chik < config.ep_chi and
            config.sample_radius < config.ep_delta):
            print("EXIT: OPTIMAL SOLUTION FOUND")
            break

        # Possibly a local optimum, added this termication condition / Gul
        if (thetak < config.ep_i and
            stepNorm < config.ep_s):
            print("EXIT: POSSIBLY AN OPTIMAL SOLUTION IS FOUND")
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if (config.trust_radius <= config.delta_min and thetak < config.ep_i):
            if subopt_flag:
                print("EXIT: FEASIBLE SOLUTION FOUND")
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare
            # the boolean subopt_flag
            subopt_flag = False

        # New criticality phase
        if not config.sample_region:
            config.sample_radius = config.trust_radius/2.0
            if config.sample_radius > chik * config.criticality_check:
                config.sample_radius = config.sample_radius/10.0
            config.trust_radius = config.sample_radius*2
        else:
            config.sample_radius = max(min(config.sample_radius,
                                   chik*config.criticality_check),
                               config.delta_min)

        logger.setCurIter(trustRadius=config.trust_radius,
                          sampleRadius=config.sample_radius)

        # Compatibility Check (Definition 2)
        # radius=max(kappa_delta*config.trust_radius*min(1,kappa_mu*config.trust_radius**mu),
        #            delta_min)
        radius = max(config.kappa_delta*config.trust_radius *
                     min(1,
                         config.kappa_mu*pow(config.trust_radius,config.mu)),
                     config.delta_min)

        try:
            if config.algorithm_type == 0:
                flag, obj = problem.compatibilityCheck(
                    x, y, z, xk, yk, zk, None, None, None, None, None, None, 
                    romParam, radius, config.compatibility_penalty, config.algorithm_type)
            elif config.algorithm_type == 1:
                flag, obj = problem.compatibilityCheck(
                    x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, x_eig, y_eig, z_eig, 
                    romParam, radius, config.compatibility_penalty, config.algorithm_type)
            elif config.algorithm_type == 2 or config.algorithm_type == 3 or config.algorithm_type == 4:
                flag, obj = problem.compatibilityCheck(
                    x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, None, None, None, 
                    romParam, radius, config.compatibility_penalty, config.algorithm_type)
            else:
                raise('Please choose one of the available variants (0-4)!')
        except:
            print("Compatibility check failed, unknown error")
            raise

        if not flag:
            raise Exception("Compatibility check fails!\n")


        theNorm = np.linalg.norm(x - xk, 2)**2 + np.linalg.norm(z - zk, 2)**2
        if (obj - config.compatibility_penalty*theNorm >
            config.ep_compatibility):
            # Restoration stepNorm
            yr = problem.evaluateDx(x)
            
            theta = np.linalg.norm(yr - y, 1)

            logger.iterlog.restoration = True

            if config.globalization_strategy == 0: 
                fe = FilterElement(
                     objk - config.gamma_f*thetak,
                     (1 - config.gamma_theta)*thetak)
                filteR.addToFilter(fe)
            elif config.globalization_strategy == 1:
                pass

            rhok = 1 - ((theta - config.ep_i)/max(thetak, config.ep_i))
            if rhok < config.eta1:
                config.trust_radius = max(config.gamma_c*config.trust_radius,
                                  config.delta_min)
                ROMAccuracy = False
            elif rhok >= config.eta2:
                config.trust_radius = min(config.gamma_e*config.trust_radius,
                    config.radius_max)
                ROMAccuracy = True
            elif rhok >= config.eta1 and rhok < config.eta2:
                ROMAccuracy = False

            obj = problem.evaluateObj(x, y, z)

            # stepNorm = min(np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf), config.step_max)
            stepNorm = np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf)
            logger.setCurIter(stepNorm=stepNorm)
            
            

        else:
            # Solve TRSP_k
            if config.algorithm_type == 0:
                flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, None, None, None, None, None, None, 
                                          romParam, config.trust_radius, config.algorithm_type)
            elif config.algorithm_type == 1:
                 flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, x_eig, y_eig, z_eig, 
                                           romParam, config.trust_radius, config.algorithm_type)  
            elif config.algorithm_type == 2 or config.algorithm_type == 3 or config.algorithm_type == 4:
                flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, x_hess_rows, y_hess_rows, z_hess_rows, None, None, None,
                                          romParam, config.trust_radius, config.algorithm_type)             
            else:
                raise('Please add other variants of the algorithm')
            
            if not flag:
                raise Exception("TRSPk fails!\n")

            # Filter
            yr = problem.evaluateDx(x)
            
            # stepNorm = min(np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf), config.step_max)
            stepNorm = np.linalg.norm(packXYZ(x-xk, y-yk, z-zk), np.inf)
            logger.setCurIter(stepNorm=stepNorm)

            # theta = Trial theta, thetak = current theta
            theta = np.linalg.norm(yr - y, 1)
            
            # Calculate rho for theta step trust region update
            rhok = 1 - ((theta - config.ep_i) /
                    max(thetak, config.ep_i))
            
            # Filter method
            if config.globalization_strategy == 0:
            
                fe = FilterElement(obj, theta)

                if not filteR.checkAcceptable(fe, config.theta_max) and iteration > 0:
                    logger.iterlog.rejected = True
                    config.trust_radius = max(config.gamma_c*stepNorm,
                                  config.delta_min)
                    # config.trust_radius = max(config.gamma_c*config.trust_radius,
                    #                   config.delta_min)
                    rebuildROM = False
                    ROMAccuracy = False
                
                    x, y, z = cloneXYZ(xk, yk, zk)
                    continue

                # Switching Condition and Trust Region update
                if (((objk - obj) >= config.kappa_theta*
                     pow(thetak, config.gamma_s))
                     and
                     (thetak < config.theta_min)):
                    logger.iterlog.fStep = True

                    config.trust_radius = min(
                         max(config.gamma_e*stepNorm, config.trust_radius),
                         config.radius_max)
                    # config.trust_radius = min(config.gamma_e*config.trust_radius,
                    #     config.radius_max)
                    ROMAccuracy = True

                else:
                    logger.iterlog.thetaStep = True

                    fe = FilterElement(
                       obj - config.gamma_f*theta,
                       (1 - config.gamma_theta)*theta)
                    
                    filteR.addToFilter(fe)

                    #trust region update
                    if rhok < config.eta1:
                        config.trust_radius = max(config.gamma_c*stepNorm,
                                      config.delta_min)
                        # config.trust_radius = max(config.gamma_c*config.trust_radius,
                        #                   config.delta_min)
                        ROMAccuracy = False
                    elif rhok >= config.eta2:
                        config.trust_radius = min(
                            max(config.gamma_e*stepNorm, config.trust_radius),
                            config.radius_max)
                        # config.trust_radius = min(config.gamma_e*config.trust_radius,
                        #     config.radius_max)
                        ROMAccuracy = True
                    elif rhok >= config.eta1 and rhok < config.eta2:
                        ROMAccuracy = False
                    
            # Funnel method
            elif config.globalization_strategy == 1:
                status = funnel.classify_step(thetak, theta, objk, obj, config.trust_radius)

                if status == 'f':
                    funnel.accept_f(theta, obj)
                    logger.iterlog.fStep = True
                    config.trust_radius = min(
                         max(config.gamma_e*stepNorm, config.trust_radius),
                         config.radius_max)
                    # config.trust_radius = min(config.gamma_e*config.trust_radius,
                    #     config.radius_max)
                    ROMAccuracy = True
                elif status in ('theta', 'theta-relax'):
                    if status == 'theta':
                        funnel.accept_theta(theta)
                        logger.iterlog.thetaStep = True
                    else:
                        funnel.relax_theta(theta)
                        logger.iterlog.relaxthetaStep = True
                    
                    #trust region update
                    if rhok < config.eta1:
                        config.trust_radius = max(config.gamma_c*stepNorm,
                                      config.delta_min)
                        # config.trust_radius = max(config.gamma_c*config.trust_radius,
                        #                   config.delta_min)
                        ROMAccuracy = False
                    elif rhok >= config.eta2:
                        config.trust_radius = min(
                            max(config.gamma_e*stepNorm, config.trust_radius),
                            config.radius_max)
                        # config.trust_radius = min(config.gamma_e*config.trust_radius,
                        #     config.radius_max)
                        ROMAccuracy = True
                    elif rhok >= config.eta1 and rhok < config.eta2:
                        ROMAccuracy = False
                else:      # 'reject'
                    logger.iterlog.rejected = True
                    config.trust_radius = max(config.gamma_c*stepNorm,
                                  config.delta_min)
                    # config.trust_radius = max(config.gamma_c*config.trust_radius,
                    #                   config.delta_min)
                    rebuildROM = False
                    ROMAccuracy = False
                
                    x, y, z = cloneXYZ(xk, yk, zk)
                    continue

            

            # # Switching Condition and Trust Region update
            # if (((objk - obj) >= config.kappa_theta*
            #      pow(thetak, config.gamma_s))
            #     and
            #     (thetak < config.theta_min)):
            #     logger.iterlog.fStep = True

            #     config.trust_radius = min(
            #         max(config.gamma_e*stepNorm, config.trust_radius),
            #         config.radius_max)
            #     # config.trust_radius = min(config.gamma_e*config.trust_radius,
            #     #     config.radius_max)

            # else:
            #     if not filteR.checkAcceptable(fe, config.theta_max) and iteration > 0:
            #         logger.iterlog.rejected = True
            #         config.trust_radius = max(config.gamma_c*stepNorm,
            #                           config.delta_min)
            #         # config.trust_radius = max(config.gamma_c*config.trust_radius,
            #         #                   config.delta_min)
            #         rebuildROM = False
            #         x, y, z = cloneXYZ(xk, yk, zk)
            #         continue
            #     else:
            #         logger.iterlog.thetaStep = True

            #         fe = FilterElement(
            #             obj - config.gamma_f*theta,
            #             (1 - config.gamma_theta)*theta)
            #         filteR.addToFilter(fe)

            #         # Calculate rho for theta step trust region update
            #         rhok = 1 - ((theta - config.ep_i) /
            #                     max(thetak, config.ep_i))
            #         if rhok < config.eta1:
            #             config.trust_radius = max(config.gamma_c*stepNorm,
            #                               config.delta_min)
            #             # config.trust_radius = max(config.gamma_c*config.trust_radius,
            #             #                   config.delta_min)
            #         elif rhok >= config.eta2:
            #             config.trust_radius = min(
            #                 max(config.gamma_e*stepNorm, config.trust_radius),
            #                 config.radius_max)
            #             # config.trust_radius = min(config.gamma_e*config.trust_radius,
            #             #     config.radius_max)

        # Accept step
        
        # print('xk')
        # print(xk)
        # print('yk')
        # print(yk)
        # print('zk')
        # print(zk)
        # print(theta)
        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        thetak = theta
        objk = obj
        end_time = time.time()
        IT = end_time - start_time


    logger.printVectors()
#    problem.reverseTransform()


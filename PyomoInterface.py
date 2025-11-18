#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.dependencies import numpy as np

from math import inf
from pyomo.common.collections import ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Block, ScalarBlock, Var, Param, VarList, ConstraintList, Constraint, Objective,
    RangeSet, value, ConcreteModel, Reals, exp, sqrt, minimize, maximize, inequality
)
from pyomo.environ import exp, sqrt
from pyomo.core.expr import current as EXPR
from pyomo.core.base.external import PythonCallbackFunction
from pyomo.core.base.numvalue import nonpyomo_leaf_types
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from GeometryGenerator import (
    generate_quadratic_rom_geometry
)
from GP_Regressor import generate_pyomo_gp_expression, GPR, kernel_matrix
from helper import maxIgnoreNone, minIgnoreNone

from pyomo.core.expr.numvalue import native_types

from pyomo.core.expr.calculus.derivatives import differentiate

from pyomo.core.expr.numvalue import NumericValue

from EigenvalueCalculator import compute_eigenvalues_from_sparse

logger = logging.getLogger('TRF Algorithm')

class GlobalizationStratgy:
    Filter = 0
    Funnel = 1

class ROMType:
    linear = 0
    quadratic = 1
    quadratic_simp = 2
    GP = 3
    ts = 4
    ts_linear = 5
    ts_quadratic = 6
    ts_quadratic_simp = 7
    ts_GP = 8
    hybrid_ts_GP = 9

kernel_list = ['RBF', 'matern12', 'matern52', 'matern72']
kernel_type = kernel_list[1]

class ALGType:
    Normal_TR = 0
    Simple_Diagonal_Loading = 1
    Hessian_Clamped_Eigenvalues = 2
    Hessian_Absolute_Eigenvalues = 3
    Hessian_Eigenvalues_Filtering = 4

class ReplaceEFVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self, trf_block, efSet):
        super(ReplaceEFVisitor, self).__init__(
            descend_into_named_expressions=True,
            remove_named_expressions=False)
        self.trf = trf_block
        self.efSet = efSet
        
    def beforeChild(self, node, child, child_idx):
        # We want to capture all of the variables on the model.
        # If we reject a step, we need to know all the vars to reset.
        descend, result = super().beforeChild(node, child, child_idx)
        if (
            not descend
            and result.__class__ not in native_types
            and result.is_variable_type()
        ):
            self.trf.all_variables.add(result)
        return descend, result

    def exitNode(self, node, values):
        new_node = super().exitNode(node, values)
        if new_node.__class__ is not EXPR.ExternalFunctionExpression:
            return new_node
        if id(new_node._fcn) not in self.efSet:
            return new_node
        # At this point we know this is an ExternalFunctionExpression
        # node that we want to replace with an auliliary variable (y)
        new_args = []
        seen = ComponentSet()
        # TODO: support more than PythonCallbackFunctions
        assert isinstance(new_node._fcn, PythonCallbackFunction)
        #
        # Note: the first argument to PythonCallbackFunction is the
        # function ID.  Since we are going to complain about constant
        # parameters, we need to skip the first argument when processing
        # the argument list.  This is really not good: we should allow
        # for constant arguments to the functions, and we should relax
        # the restriction that the external functions implement the
        # PythonCallbackFunction API (that restriction leads unfortunate
        # things later; i.e., accessing the private _fcn attribute
        # below).
        for arg in values[1][1:]:
            if type(arg) in nonpyomo_leaf_types or arg.is_fixed():
                # We currently do not allow constants or parameters for
                # the external functions.
                raise RuntimeError( 
                    "TrustRegion does not support black boxes with "
                    "constant or parameter inputs\n\tExpression: %s"
                    % (new_node,) )
            if arg.is_expression_type():
                # All expressions (including simple linear expressions)
                # are replaced with a single auxiliary variable (and
                # eventually an additional constraint equating the
                # auxiliary variable to the original expression)
                _x = self.trf.x.add()
                _x.set_value( value(arg) )
                self.trf.conset.add(_x == arg)
                new_args.append(_x)
            else:
                # The only thing left is bare variables: check for duplicates.
                if arg in seen:
                    raise RuntimeError(
                        "TrustRegion does not support black boxes with "
                        "duplicate input arguments\n\tExpression: %s"
                        % (new_node,) )
                seen.add(arg)
                new_args.append(arg)
        _y = self.trf.y.add()
        self.trf.external_fcns.append(new_node)
        self.trf.exfn_xvars.append(new_args)
        return _y

class PyomoInterface(object):
    '''
    Initialize with a pyomo model m.
    This is used in TRF.py, same requirements for m apply

    m is reformulated into form for use in TRF algorithm

    Specified ExternalFunction() objects are replaced with new variables
    All new attributes (including these variables) are stored on block
    "tR"


    Note: quadratic ROM is messy, uses full dimension of x variables. clean up later.

    '''

    stream_solver = False # True prints solver output to screen
    keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
    countDx = -1
    romtype = ROMType.linear
    algtype = ALGType.Normal_TR

    def __init__(self, m, eflist, config):

        self.config = config
        self.model = m;
        self.TRF = self.transformForTrustRegion(self.model,eflist)

        self.lx = len(self.TRF.xvars)
        self.lz = len(self.TRF.zvars)
        self.ly = len(self.TRF.y)

        self.createParam()
        self.createRomConstraint()
        self.createCompCheckObjective()
        self.cacheBound()

        self.geoM = None
        self.pset = None


    def substituteEF(self, expr, trf, efSet):
        """Substitute out an External Function

        Arguments:
            expr : a pyomo expression. We will search this expression tree
            trf : a pyomo block. We will add tear variables y on this block
            efSet: the (pyomo) set of external functions for which we will use TRF method

        This function returns an expression after removing any
        ExternalFunction in the set efSet from the expression tree
        expr. New variables are declared on the trf block and replace
        the external function.

        """
        return ReplaceEFVisitor(trf, efSet).dfs_postorder_stack(expr)


    def transformForTrustRegion(self,model,eflist):
        # transform and model into suitable form for TRF method
        #
        # Arguments:
        # model : pyomo model containing ExternalFunctions
        # eflist : a list of the external functions that will be
        #   handled with TRF method rather than calls to compiled code

        efSet = set([id(x) for x in eflist])

        TRF = Block()
        # model.TRF = TRF  # Explicitly assign the block to the model attribute # Gul
        TRF.all_variables = ComponentSet()

        # Get all varibles
        seenVar = set()
        allVariables = []
        for var in model.component_data_objects(Var):
            if id(var) not in seenVar:
                seenVar.add(id(var))
                allVariables.append(var)


        # This assumes that an external funtion call is present, required!
        model.add_component(unique_component_name(model,'tR'), TRF)
        TRF.y = VarList()
        TRF.x = VarList()
        TRF.conset = ConstraintList()
        TRF.external_fcns = []
        TRF.exfn_xvars = []

        # TODO: Copy constraints onto block so that transformation can be reversed.

        for con in model.component_data_objects(Constraint,active=True):
            con.set_value((con.lower, self.substituteEF(con.body,TRF,efSet), con.upper))
        for obj in model.component_data_objects(Objective,active=True):
            obj.set_value(self.substituteEF(obj.expr,TRF,efSet))
            ## Assume only one active objective function here
            self.objective=obj

        if self.objective.sense == maximize:
            self.objective.expr = -1* self.objective.expr
            self.objective.sense = minimize



        # xvars and zvars are lists of x and z varibles as in the paper
        TRF.xvars = []
        TRF.zvars = []
        seenVar = set()
        for varss in TRF.exfn_xvars:
            for var in varss:
                if id(var) not in seenVar:
                    seenVar.add(id(var))
                    TRF.xvars.append(var)

        for var in allVariables:
            if id(var) not in seenVar:
                seenVar.add(id(var))
                TRF.zvars.append(var)

        # TODO: build dict for exfn_xvars
        # assume it is not bottleneck of the code
        self.exfn_xvars_ind = []
        for varss in TRF.exfn_xvars:
            listtmp = []
            for var in varss:
                for i in range(len(TRF.xvars)):
                    if(id(var)==id(TRF.xvars[i])):
                        listtmp.append(i)
                        break

            self.exfn_xvars_ind.append(listtmp)
        
        return TRF

    # TODO:
    # def reverseTransform(self):
    #     # After solving the problem, return the
    #     # model back to the original form, and delete
    #     # all add-on structures
    #     for conobj in self.TRF.changed_objects:
    #         conobj.activate()

    #     self.model.del_component(self.model.tR)


    def getInitialValue(self):
        x = np.zeros(self.lx, dtype=float)
        y = np.zeros(self.ly, dtype=float)
        z = np.zeros(self.lz, dtype=float)
        for i in range(0, self.lx):
            x[i] = value(self.TRF.xvars[i])
        for i in range(0, self.ly):
            #initialization of y?
            y[i] = 1
        for i in range(0, self.lz):
            z[i] = value(self.TRF.zvars[i])
        return x, y, z

    def createParam(self):
        self.TRF.ind_lx=RangeSet(0,self.lx-1)
        self.TRF.ind_ly=RangeSet(0,self.ly-1)
        self.TRF.ind_lz=RangeSet(0,self.lz-1)
        self.TRF.px0 = Param(self.TRF.ind_lx,mutable=True,default=0)
        self.TRF.py0 = Param(self.TRF.ind_ly,mutable=True,default=0)
        self.TRF.pz0 = Param(self.TRF.ind_lz,mutable=True,default=0)
        self.TRF.plrom = Param(self.TRF.ind_ly,range(self.lx+1),mutable=True,default=0)
        self.TRF.pqrom = Param(self.TRF.ind_ly,range(int((self.lx*self.lx+self.lx*3)/2. + 1)),mutable=True,default=0)
        self.TRF.ppenaltyComp = Param(mutable=True,default=0)
        self.TRF.grad_e1 = Param(self.TRF.ind_lx,mutable=True,default=0)
        self.TRF.grad_true_e1 = Param(self.TRF.ind_lx,mutable=True,default=0)
        self.TRF.ind_gpr_samples = RangeSet(0, int((2 * self.lx) + 1) - 1) # GPR samples
        self.TRF.gprom_x = Param(self.TRF.ind_ly, self.TRF.ind_gpr_samples, self.TRF.ind_lx, mutable=True, default=1)
        self.TRF.gprom_y = Param(self.TRF.ind_ly, self.TRF.ind_gpr_samples, mutable=True, default=1) 
        self.TRF.gprom_GPM = Param(self.TRF.ind_ly, self.TRF.ind_gpr_samples, mutable=True, default=1) 
        self.TRF.gprom_ls = Param(self.TRF.ind_ly, mutable=True, default=1)
        self.TRF.gprom_sigma = Param(self.TRF.ind_ly, mutable=True, default=1)
        self.TRF.base_radius = Param(self.TRF.ind_ly, mutable=True, default=1)

    def ROMlinear(self,model,i):
        ind = self.exfn_xvars_ind[i]
        y1 = (model.plrom[i,0] + sum(model.plrom[i,j+1] * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(0, len(ind))))
        return y1

    def ROMQuad(self,model,i):
        y1 = model.pqrom[i,0] + sum(model.pqrom[i,j+1] * (model.xvars[j] - model.px0[j]) for j in range(0,self.lx))
        count = self.lx+1
        for j1 in range(self.lx):
            for j2 in range(j1,self.lx):
                y1 += (model.xvars[j2] - model.px0[j2]) * (model.xvars[j1] - model.px0[j1])*model.pqrom[i,count]
                count = count + 1
        return y1
    
    def ROMQuadSimp(self,model,i):
        y1 = model.pqrom[i,0] + sum(model.pqrom[i,j+1] * (model.xvars[j] - model.px0[j]) for j in range(0,self.lx))
        count = self.lx+1
        for j1 in range(self.lx):
            y1 += (model.xvars[j1] - model.px0[j1]) * (model.xvars[j1] - model.px0[j1])*model.pqrom[i,count]
            count = count + 1
        return y1
    
    def ROMGP(self, model, i):
        ind = self.exfn_xvars_ind[i]
        true_fcn = self.TRF.external_fcns[i]._fcn
        values = [model.xvars[ind[j]] for j in range(len(ind))]
        true_e1 = true_fcn._fcn(*values)
        if not hasattr(self.TRF, "gprom_x") or not hasattr(self.TRF, "gprom_y"):
            raise RuntimeError("TRF.gprom_x or TRF.gprom_y is not initialized.")
        x_points = [[(self.TRF.gprom_x[i, j, k]) for k in range(self.lx)] for j in range(int((2 * self.lx) + 1))]
        y_points = [(self.TRF.gprom_y[i, j]) for j in range(int((2 * self.lx) + 1))]
        x_unknown = [(model.xvars[ind[j]] - model.px0[ind[j]].value) for j in range(0, len(ind))]
        UKM = kernel_matrix(x_points, y_points, x_unknown, kernel_type, model.gprom_sigma[i], model.gprom_ls[i])
        y1 = true_e1 + sum(UKM[j] * model.gprom_GPM[i,j] for j in range(int((2 * self.lx) + 1)))
        return y1
    
    def ROM_TS(self,model,i):
        ind = self.exfn_xvars_ind[i]
        true_fcn = self.TRF.external_fcns[i]._fcn
        values = [model.xvars[ind[j]] for j in range(len(ind))]
        true_e1 = true_fcn._fcn(*values)
        grad_true_e1 = [differentiate(true_e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        y1 = true_e1 + sum((model.grad_true_e1[ind[j]]) * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(len(ind)))
        return y1
    
    def ROM_TS_linear(self,model,i):
        ind = self.exfn_xvars_ind[i]
        e1 = (model.plrom[i,0] + sum(model.plrom[i,j+1] * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(0, len(ind))))
        grad_e1 = [differentiate(e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        true_fcn = self.TRF.external_fcns[i]._fcn
        values = [model.xvars[ind[j]] for j in range(len(ind))]
        true_e1 = true_fcn._fcn(*values)
        grad_true_e1 = [differentiate(true_e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        y1 = (e1 
              + (true_e1 - e1)
              + sum((model.grad_true_e1[ind[j]] - model.grad_e1[ind[j]]) * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(len(ind)))
              )
        return y1
    
    def ROM_TS_Quad(self,model,i):
        ind = self.exfn_xvars_ind[i]
        e1 = model.pqrom[i,0] + sum(model.pqrom[i,j+1] * (model.xvars[j] - model.px0[j]) for j in range(0,self.lx))
        count = self.lx+1
        for j1 in range(self.lx):
            for j2 in range(j1,self.lx):
                e1 += (model.xvars[j2] - model.px0[j2]) * (model.xvars[j1] - model.px0[j1])*model.pqrom[i,count]
                count = count + 1
        grad_e1 = [differentiate(e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        true_fcn = self.TRF.external_fcns[i]._fcn
        values = [model.xvars[ind[j]] for j in range(len(ind))]
        true_e1 = true_fcn._fcn(*values)
        grad_true_e1 = [differentiate(true_e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        y1 = (e1 
              + (true_e1 - e1)
              + sum((model.grad_true_e1[ind[j]] - model.grad_e1[ind[j]]) * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(len(ind)))
              )
        return y1
    
    def ROM_TS_QuadSimp(self,model,i):
        ind = self.exfn_xvars_ind[i]
        e1 = model.pqrom[i,0] + sum(model.pqrom[i,j+1] * (model.xvars[j] - model.px0[j]) for j in range(0,self.lx))
        count = self.lx+1
        for j1 in range(self.lx):
            e1 += (model.xvars[j1] - model.px0[j1]) * (model.xvars[j1] - model.px0[j1])*model.pqrom[i,count]
            count = count + 1
        grad_e1 = [differentiate(e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        true_fcn = self.TRF.external_fcns[i]._fcn
        values = [model.xvars[ind[j]] for j in range(len(ind))]
        true_e1 = true_fcn._fcn(*values)
        grad_true_e1 = [differentiate(true_e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        y1 = (e1 
              + (true_e1 - e1)
              + sum((model.grad_true_e1[ind[j]] - model.grad_e1[ind[j]]) * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(len(ind)))
              )
        return y1
    
    def ROM_TS_GP(self, model, i):
        ind = self.exfn_xvars_ind[i]
        true_fcn = self.TRF.external_fcns[i]._fcn
        values = [model.xvars[ind[j]] for j in range(len(ind))]
        true_e1 = true_fcn._fcn(*values)
        grad_true_e1 = [differentiate(true_e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        if not hasattr(self.TRF, "gprom_x") or not hasattr(self.TRF, "gprom_y"):
            raise RuntimeError("TRF.gprom_x or TRF.gprom_y is not initialized.")
        x_points = [[(self.TRF.gprom_x[i, j, k]) for k in range(self.lx)] for j in range(int((2 * self.lx) + 1))]
        y_points = [(self.TRF.gprom_y[i, j]) for j in range(int((2 * self.lx) + 1))]
        x_unknown = [(model.xvars[ind[j]] - model.px0[ind[j]].value) for j in range(0, len(ind))]
        UKM = kernel_matrix(x_points, y_points, x_unknown, kernel_type, model.gprom_sigma[i], model.gprom_ls[i])
        e1 = true_e1 + sum(UKM[j] * model.gprom_GPM[i,j] for j in range(int((2 * self.lx) + 1)))
        grad_e1 = [differentiate(e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        y1 = (e1 
              + (true_e1 - e1)
              + sum((model.grad_true_e1[ind[j]] - model.grad_e1[ind[j]]) * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(len(ind))))
        return y1
    
    def ROM_H_TS_GP(self,model,i):
        ind = self.exfn_xvars_ind[i]
        true_fcn = self.TRF.external_fcns[i]._fcn
        values = [model.xvars[ind[j]].value for j in range(len(ind))]
        true_e1 = true_fcn._fcn(*values)
        grad_true_e1 = [differentiate(true_e1, wrt_list=[model.xvars[ind[j]]]) for j in range(len(ind))]
        e1 = true_e1 + sum((model.grad_true_e1[ind[j]]) * (model.xvars[ind[j]] - model.px0[ind[j]]) for j in range(len(ind)))
        e1_value = true_e1 + sum((model.grad_true_e1[ind[j]].value) * (model.xvars[ind[j]].value - model.px0[ind[j]].value) for j in range(len(ind)))
        C = 3 # Scaling factor
        radius = self.TRF.base_radius[i].value
        x_points = []
        y_points = []
        x_points.append(values)  # Center point
        y_points.append([true_e1 - e1_value])
        for scale in [0.5, C-2, C - 1, C]:
            new_point = [xi + scale * radius for xi in values]  # Create new input point
            true_value = true_fcn._fcn(*new_point)  # Evaluate true function
            grad_true_value = [differentiate(true_value, wrt_list=[xi + scale * radius]) for xi in values]
            grad_true_value = [g[0] if isinstance(g, (list, tuple)) else g for g in grad_true_value]
            taylor_value = true_value + sum(grad_true_value[j] * (new_point[j] - model.px0[ind[j]].value) for j in range(len(ind)))  # Taylor approx
            x_points.append(new_point)  # Store input
            y_points.append([true_value - taylor_value])  # Store error
        sigma, length_scale, m, covM, M = GPR(x_points, y_points, x_points, kernel_type)
        KM = list(covM@M)
        x_unknown = [(model.xvars[ind[j]] - model.px0[ind[j]].value) for j in range(0, len(ind))]
        UKM = kernel_matrix(x_points, y_points, x_unknown, kernel_type, sigma, length_scale)
        e2 = sum(UKM[j] * KM[j] for j in range(len(KM)))
        y1 = e1 + e2
        return y1

    def createRomConstraint(self):
        def consROMl(model, i):
            return  model.y[i+1] == self.ROMlinear(model,i)
        self.TRF.romL = Constraint(self.TRF.ind_ly, rule=consROMl)

        def consROMq(model, i):
            return  model.y[i+1] == self.ROMQuad(model,i)
        self.TRF.romQ = Constraint(self.TRF.ind_ly, rule=consROMq)
        
        def consROMqs(model, i):
            return  model.y[i+1] == self.ROMQuadSimp(model,i)
        self.TRF.romQS = Constraint(self.TRF.ind_ly, rule=consROMqs)
        
        def consROMGP(model, i):
            return  model.y[i+1] == self.ROMGP(model, i)
        self.TRF.romGP = Constraint(self.TRF.ind_ly, rule=consROMGP)
        
        def consROMts(model, i):
            return  model.y[i+1] == self.ROM_TS(model,i)
        self.TRF.romTS = Constraint(self.TRF.ind_ly, rule=consROMts)
        
        def consROMtsl(model, i):
            return  model.y[i+1] == self.ROM_TS_linear(model,i)
        self.TRF.romTSL = Constraint(self.TRF.ind_ly, rule=consROMtsl)
        
        def consROMtsq(model, i):
            return  model.y[i+1] == self.ROM_TS_Quad(model,i)
        self.TRF.romTSQ = Constraint(self.TRF.ind_ly, rule=consROMtsq)
        
        def consROMtsqs(model, i):
            return  model.y[i+1] == self.ROM_TS_QuadSimp(model,i)
        self.TRF.romTSQS = Constraint(self.TRF.ind_ly, rule=consROMtsqs)
        
        def consROMtsGP(model, i):
            return  model.y[i+1] == self.ROM_TS_GP(model, i)
        self.TRF.romTSGP = Constraint(self.TRF.ind_ly, rule=consROMtsGP)
        
        def consROMhtsGP(model, i):
            return  model.y[i+1] == self.ROM_H_TS_GP(model, i)
        self.TRF.romHTSGP = Constraint(self.TRF.ind_ly, rule=consROMhtsGP)


    def createCompCheckObjective(self):
        obj = 0
        model = self.TRF
        for i in range(0, self.ly):
            obj += (self.ROMlinear(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckL = Objective(expr=obj)

        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROMQuad(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckQ = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROMQuadSimp(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckQS = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROMGP(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckGP = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROM_TS(model,i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckTS = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROM_TS_linear(model,i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckTSL = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROM_TS_Quad(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckTSQ = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROM_TS_QuadSimp(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckTSQS = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROM_TS_GP(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckTSGP = Objective(expr=obj)
        
        obj = 0
        for i in range(0, self.ly):
            obj += (self.ROM_H_TS_GP(model, i) - model.y[i+1]) ** 2
        # for i in range(0, self.lx):
        #     obj += model.ppenaltyComp * (model.xvars[i] - model.px0[i]) ** 2
        # for i in range(0, self.lz):
        #     obj += model.ppenaltyComp * (model.zvars[i] - model.pz0[i]) ** 2
        model.objCompCheckHTSGP = Objective(expr=obj)
        

    def cacheBound(self):
        self.TRF.xvarlo = []
        self.TRF.xvarup = []
        self.TRF.zvarlo = []
        self.TRF.zvarup = []
        for x in self.TRF.xvars:
            self.TRF.xvarlo.append(x.lb)
            self.TRF.xvarup.append(x.ub)
        for z in self.TRF.zvars:
            self.TRF.zvarlo.append(z.lb)
            self.TRF.zvarup.append(z.ub)


    def setParam(self,x0=None,y0=None,z0=None,rom_params=None, penaltyComp = None):
        if x0 is not None:
            for i in range(self.lx):
                self.TRF.px0[i] = x0[i]

        # if y0 is not None:
        #     for i in range(self.ly):
        #         self.TRF.py0[i] = y0[i]

        # if z0 is not None:
        #     for i in range(self.lz):
        #         self.TRF.pz0[i] = z0[i]

        if rom_params is not None:
            if(self.romtype==ROMType.linear):
                for i in range(self.ly):
                    for j in range(len(rom_params[i])):
                        self.TRF.plrom[i,j] = rom_params[i][j]
            elif(self.romtype==ROMType.quadratic):
                for i in range(self.ly):
                    for j in range(len(rom_params[i])):
                        self.TRF.pqrom[i,j] = rom_params[i][j]
            elif(self.romtype==ROMType.quadratic_simp):
                for i in range(self.ly):
                    for j in range(len(rom_params[i])):
                        self.TRF.pqrom[i,j] = rom_params[i][j]
            elif(self.romtype==ROMType.ts):
                pass  # Skip execution when romtype is ts because basis is 0 in this case
            elif(self.romtype==ROMType.ts_linear):
                for i in range(self.ly):
                    for j in range(len(rom_params[i])):
                        self.TRF.plrom[i,j] = rom_params[i][j]
            elif(self.romtype==ROMType.ts_quadratic):
                for i in range(self.ly):
                    for j in range(len(rom_params[i])):
                        self.TRF.pqrom[i,j] = rom_params[i][j]
            elif(self.romtype==ROMType.ts_quadratic_simp):
                for i in range(self.ly):
                    for j in range(len(rom_params[i])):
                        self.TRF.pqrom[i,j] = rom_params[i][j]
            elif(self.romtype in [ROMType.GP, ROMType.ts_GP]):
                for i in range(self.ly):
                    x_points, y_points, GP_param, known_GPM = rom_params[i]
                    self.TRF.gprom_sigma[i] = GP_param[0]
                    self.TRF.gprom_ls[i] = GP_param[1]
                    for j, point in enumerate(x_points):
                        for k, value in enumerate(point):
                            self.TRF.gprom_x[i, j, k].set_value(value)
                    for j, y_value in enumerate(y_points):
                        self.TRF.gprom_y[i, j].set_value(y_value[0])
                    for j, value in enumerate(known_GPM):
                        self.TRF.gprom_GPM[i, j].set_value(value[0])
            elif(self.romtype==ROMType.hybrid_ts_GP):
                for i in range(self.ly):
                    self.TRF.base_radius[i].set_value(rom_params[i])
                    
                    
        # if penaltyComp is not None:
        #     self.TRF.ppenaltyComp.set_value(penaltyComp)

    def setVarValue(self, x=None, y=None, z=None):
        if x is not None:
            if(len(x) != self.lx):
                raise Exception(
                    "setValue: The dimension of x is not consistant!\n")
            for i in range(0, self.lx):
                self.TRF.xvars[i].set_value(x[i])
        if y is not None:
            if(len(y) != self.ly):
                raise Exception(
                    "setValue: The dimension of y is not consistant!\n")
            for i in range(0, self.ly):
                self.TRF.y[i+1].set_value(y[i])

        if z is not None:
            if(len(z) != self.lz):
                raise Exception(
                    "setValue: The dimension of z is not consistant!\n")
            for i in range(0, self.lz):
                self.TRF.zvars[i].set_value(z[i])

    def setBound(self, x0, y0, z0, radius):
        for i in range(0,self.lx):
            self.TRF.xvars[i].setlb(maxIgnoreNone(x0[i] - radius,self.TRF.xvarlo[i]))
            self.TRF.xvars[i].setub(minIgnoreNone(x0[i] + radius,self.TRF.xvarup[i]))
        for i in range(0,self.ly):
            self.TRF.y[i+1].setlb(y0[i] - radius)
            self.TRF.y[i+1].setub(y0[i] + radius)
        for i in range(0,self.lz):
            self.TRF.zvars[i].setlb(maxIgnoreNone(z0[i] - radius,self.TRF.zvarlo[i]))
            self.TRF.zvars[i].setub(minIgnoreNone(z0[i] + radius,self.TRF.zvarup[i]))
     
            
    def setBoundTRSPk1(self, x0, y0, z0, radius):
        for i in range(0,self.lx):
            self.TRF.xvars[i].setlb(self.TRF.xvarlo[i])
            self.TRF.xvars[i].setub(self.TRF.xvarup[i])
        for i in range(0,self.ly):
            self.TRF.y[i+1].setlb(None)
            self.TRF.y[i+1].setub(None)
        for i in range(0,self.lz):
            self.TRF.zvars[i].setlb(self.TRF.zvarlo[i])
            self.TRF.zvars[i].setub(self.TRF.zvarup[i])

    def TRSPk1_CONS(self, model, x0, y0, z0, x_hess_rows, y_hess_rows, z_hess_rows, x_eig, y_eig, z_eig, radius, algtype):
        if (algtype in [ALGType.Normal_TR]):
            pass
        
        elif (algtype in [ALGType.Simple_Diagonal_Loading]):
            step_matrix = (
                       [self.TRF.xvars[i] - x0[i] for i in range(self.lx)] +
                       [self.TRF.y[i+1] - y0[i] for i in range(self.ly)] +
                       [self.TRF.zvars[i] - z0[i] for i in range(self.lz)]
                        )
            hessian_matrix = x_hess_rows + y_hess_rows + z_hess_rows
            
            eigenvalue_list = x_eig + y_eig + z_eig
            
            # Minimum eigenvalue
            lambda_min = min(eigenvalue_list)
            
            # Positive shift to ensure PD: shift diagonal by (ε - λ_min) if λ_min < ε
            epsilon = 1e-4
            alpha = max(0.0, epsilon - lambda_min)
            
            # Apply diagonal loading: H_pd = H + alpha * I
            for i in range(len(hessian_matrix)):
                hessian_matrix[i][i] += alpha
            
            # # Debugging print statements
            # print("\n--- DEBUG INFO ---")
            # print("Step Matrix:", step_matrix)
            # print("Hessian Matrix:", hessian_matrix)
            
            def consTR(model):
                trust_region_expr = sum(
                    sum(hessian_matrix[i][j] * step_matrix[j] for j in range(len(step_matrix))) * step_matrix[i]
                    for i in range(len(step_matrix))
                )
                # # Debugging print statement
                # print("Trust Region Expression:", trust_region_expr)
                # if isinstance(trust_region_expr, bool):
                #     print("ERROR: trust_region_expr resolved to a Boolean value!")
                #     return Constraint.Feasible  # Prevents Pyomo from raising an error
                return trust_region_expr <= radius + 1e-5
            constraint_name = "trust_region_constraint"
            if hasattr(self.TRF, constraint_name):
                self.TRF.del_component(constraint_name)
                self.TRF.add_component(constraint_name, Constraint(rule=consTR))
        
        elif (algtype in [ALGType.Hessian_Clamped_Eigenvalues, ALGType.Hessian_Absolute_Eigenvalues]):
            step_matrix = (
                       [self.TRF.xvars[i] - x0[i] for i in range(self.lx)] +
                       [self.TRF.y[i+1] - y0[i] for i in range(self.ly)] +
                       [self.TRF.zvars[i] - z0[i] for i in range(self.lz)]
                        )
            hessian_matrix = x_hess_rows + y_hess_rows + z_hess_rows
            
            # # Debugging print statements
            # print("\n--- DEBUG INFO ---")
            # print("Step Matrix:", step_matrix)
            # print("Hessian Matrix:", hessian_matrix)
            
            def consTR(model):
                trust_region_expr = sum(
                                sum(hessian_matrix[i][j] * step_matrix[j] for j in range(len(step_matrix))) * step_matrix[i]
                                for i in range(len(step_matrix))
                                   )
                # # # Debugging print statement
                # print("Trust Region Expression:", trust_region_expr)
                # if isinstance(trust_region_expr, bool):
                #     print("ERROR: trust_region_expr resolved to a Boolean value!")
                #     return Constraint.Feasible  # Prevents Pyomo from raising an error
                return trust_region_expr <= radius + 1e-5
            constraint_name = "trust_region_constraint"
            if hasattr(self.TRF, constraint_name):
                self.TRF.del_component(constraint_name)
            self.TRF.add_component(constraint_name, Constraint(rule=consTR))
            

    def evaluateDx(self,x):
        # This is messy, currently redundant with
        # some lines in buildROM()
        self.countDx += 1
        ans = []
        for i in range(0,self.ly):
            fcn = self.TRF.external_fcns[i]._fcn
            values = []
            for j in self.exfn_xvars_ind[i]:
                values.append(x[j])

            ans.append(fcn._fcn(*values))
        return np.array(ans)

    def evaluateObj(self, x, y, z):
        if(len(x) != self.lx or len(y) != self.ly or len(z) != self.lz):
            raise Exception("evaluateObj: The dimension is not consistent with the initialization \n")
        self.setVarValue(x=x,y=y,z=z)
        return self.objective()

    def deactiveExtraConObj(self):
        self.TRF.objCompCheckL.deactivate()
        self.TRF.romL.deactivate()
        self.TRF.objCompCheckQ.deactivate()
        self.TRF.romQ.deactivate()
        self.TRF.objCompCheckQS.deactivate()
        self.TRF.romQS.deactivate()
        self.TRF.objCompCheckGP.deactivate()
        self.TRF.romGP.deactivate()
        self.TRF.objCompCheckTS.deactivate()
        self.TRF.romTS.deactivate()
        self.TRF.objCompCheckTSL.deactivate()
        self.TRF.romTSL.deactivate()
        self.TRF.objCompCheckTSQ.deactivate()
        self.TRF.romTSQ.deactivate()
        self.TRF.objCompCheckTSQS.deactivate()
        self.TRF.romTSQS.deactivate()
        self.TRF.objCompCheckTSGP.deactivate()
        self.TRF.romTSGP.deactivate()
        self.TRF.objCompCheckHTSGP.deactivate()
        self.TRF.romHTSGP.deactivate()
        self.objective.activate()

    def activateRomCons(self,x0, rom_params):
        self.setParam(x0=x0,rom_params=rom_params)
        if(self.romtype==ROMType.linear):
            self.TRF.romL.activate()
        elif(self.romtype==ROMType.quadratic):
            self.TRF.romQ.activate()
        elif(self.romtype==ROMType.quadratic_simp):
            self.TRF.romQS.activate()
        elif(self.romtype==ROMType.GP):
            self.TRF.romGP.activate()
        elif(self.romtype==ROMType.ts):
            self.TRF.romTS.activate()
        elif(self.romtype==ROMType.ts_linear):
            self.TRF.romTSL.activate()
        elif(self.romtype==ROMType.ts_quadratic):
            self.TRF.romTSQ.activate()
        elif(self.romtype==ROMType.ts_quadratic_simp):
            self.TRF.romTSQS.activate()
        elif(self.romtype==ROMType.ts_GP):
            self.TRF.romTSGP.activate()
        elif(self.romtype==ROMType.hybrid_ts_GP):
            self.TRF.romHTSGP.activate()

    def activateCompCheckObjective(self, x0, z0, rom_params, penalty):
        self.setParam(x0=x0,z0=z0,rom_params=rom_params,penaltyComp = penalty)
        if(self.romtype==ROMType.linear):
            self.TRF.objCompCheckL.activate()
        elif(self.romtype==ROMType.quadratic):
            self.TRF.objCompCheckQ.activate()
        elif(self.romtype==ROMType.quadratic_simp):
            self.TRF.objCompCheckQS.activate()
        elif(self.romtype==ROMType.GP):
            self.TRF.objCompCheckGP.activate()
        elif(self.romtype==ROMType.ts):
            self.TRF.objCompCheckTS.activate()
        elif(self.romtype==ROMType.ts_linear):
            self.TRF.objCompCheckTSL.activate()
        elif(self.romtype==ROMType.ts_quadratic):
            self.TRF.objCompCheckTSQ.activate()
        elif(self.romtype==ROMType.ts_quadratic_simp):
            self.TRF.objCompCheckTSQS.activate()
        elif(self.romtype==ROMType.ts_GP):
            self.TRF.objCompCheckTSGP.activate()
        elif(self.romtype==ROMType.hybrid_ts_GP):
            self.TRF.objCompCheckHTSGP.activate()
        self.objective.deactivate()


    def solveModel(self, x, y, z):
        model = self.model
        opt = SolverFactory(self.config.solver)
        opt.options.update(self.config.solver_options)
        ##
        # opt.options['halt_on_ampl_error'] = 'yes'
        # # opt.options['solver_options']['linear_solver'] = 'mumps'  # Other options: 'ma27', 'ma57', 'pardiso'
        # opt.options['max_iter'] = 20000
        
        opt.options = {
            # 'nlp_scaling_method': 'gradient-based',
            'max_iter': 20000,
            'halt_on_ampl_error': 'yes',
            # 'print_level': 7,
            # 'tol': 1e-7, #1e-6
            # 'constr_viol_tol': 1e-7, #1e-6
            # 'dual_inf_tol': 1e-4, #1e-6
            # 'acceptable_constr_viol_tol': 1e-4,
            # 'acceptable_tol': 1e-4, #1e-3
            # 'mu_strategy' : 'adaptive'
        }
        
        ##
        results = opt.solve(
            model, keepfiles=self.keepfiles, tee=self.stream_solver)

        if ((results.solver.status == SolverStatus.ok)
                and (results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible])):
        # if (results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible]):
            model.solutions.load_from(results)
            for i in range(0, self.lx):
                x[i] = value(self.TRF.xvars[i])
            for i in range(0, self.ly):
                y[i] = value(self.TRF.y[i+1])
            for i in range(0, self.lz):
                z[i] = value(self.TRF.zvars[i])

            for obj in model.component_data_objects(Objective,active=True):
                return True, obj()

        else:
            print("Waring: solver Status: " + str(results.solver.status))
            print("And Termination Conditions: " + str(results.solver.termination_condition))
            return False, 0

    # def TRSPk(self, x, y, z, x0, y0, z0, x_eig, y_eig, z_eig, rom_params, radius):
    # def TRSPk(self, x, y, z, x0, y0, z0, rom_params, radius):
    def TRSPk(self, x, y, z, x0, y0, z0, a, b, c, d, e, f, rom_params, radius, algtype):
        
        if(len(x) != self.lx or len(y) != self.ly or len(z) != self.lz or
                len(x0) != self.lx or len(y0) != self.ly or len(z0) != self.lz):
            raise Exception(
                "TRSP_k: The dimension is not consistant with the initialization!\n")

        if (algtype in [ALGType.Normal_TR]):
            self.setBound(x0, y0, z0, radius)
        elif (algtype in [ALGType.Simple_Diagonal_Loading, ALGType.Hessian_Clamped_Eigenvalues, ALGType.Hessian_Absolute_Eigenvalues, ALGType.Hessian_Eigenvalues_Filtering]):
            self.setBoundTRSPk1(x0, y0, z0, radius)
            self.TRSPk1_CONS(self.model, x0, y0, z0, a, b, c, d, e, f, radius, algtype)      
        else:
            raise('Please select the trust-region constraint for Trust-region Subproblem')
        
        self.setVarValue(x, y, z)
        self.deactiveExtraConObj()
        self.activateRomCons(x0, rom_params)

        return self.solveModel(x, y, z)

    def compatibilityCheck(self, x, y, z, x0, y0, z0, a, b, c, d, e, f, rom_params, radius, penalty, algtype):
        if(len(x) != self.lx or len(y) != self.ly or len(z) != self.lz or
                len(x0) != self.lx or len(y0) != self.ly or len(z0) != self.lz):
            raise Exception(
                "Compatibility_Check: The dimension is not consistant with the initialization!\n")
        
        if (algtype in [ALGType.Normal_TR]):
            self.setBound(x0, y0, z0, radius)
        elif (algtype in [ALGType.Simple_Diagonal_Loading, ALGType.Hessian_Clamped_Eigenvalues, ALGType.Hessian_Absolute_Eigenvalues, ALGType.Hessian_Eigenvalues_Filtering]):
            self.setBoundTRSPk1(x0, y0, z0, radius)
            self.TRSPk1_CONS(self.model, x0, y0, z0, a, b, c, d, e, f, radius, algtype)   
        else:
            raise('Please select the trust-region constraint for Trust-region Subproblem')

        # self.setBound(x0, y0, z0, radius)
        # self.setBoundTRSPk2(x0, y0, z0, radius)
        # self.TRSPk_CONS(self.model, x0, y0, z0, a, b, c, radius, 3)
        self.setVarValue(x, y, z)
        self.deactiveExtraConObj()
        self.activateCompCheckObjective(x0, z0, rom_params, penalty)
        #self.deactiveExtraConObj()
        #self.model.pprint()
        return self.solveModel(x, y, z)
    
    def grad_hess_calc(self, x, y, z, rom_params):
        
        model = self.model
 
        self.setBound(x, y, z, 1e10)
        self.setVarValue(x=x,y=y,z=z)
        self.deactiveExtraConObj()
        self.activateRomCons(x, rom_params)
        
        optGJH = SolverFactory('local.gjh')
        optGJH.solve(model, tee=False, symbolic_solver_labels=True)
        
        g, J, H, varlist, conlist = model._gjh_info 
        
        HM, EV, abs_eigenvalues, H_abs, clamped_eigenvalues, H_clamped = compute_eigenvalues_from_sparse(H, len(varlist))
        
        cleaned_varlist = [model.find_component(var) for var in varlist] 
        eigenvalue_dict = {cleaned_varlist[i].name: EV[i] for i in range(len(cleaned_varlist))}
        abs_eigenvalue_dict = {cleaned_varlist[i].name: abs_eigenvalues[i] for i in range(len(cleaned_varlist))}
        clamped_eigenvalue_dict = {cleaned_varlist[i].name: clamped_eigenvalues[i] for i in range(len(cleaned_varlist))} 
        HM_row_dict = {cleaned_varlist[i].name: HM[i] for i in range(len(cleaned_varlist))}
        H_abs_row_dict = {cleaned_varlist[i].name: H_abs[i] for i in range(len(cleaned_varlist))}
        H_clamped_row_dict = {cleaned_varlist[i].name: H_clamped[i] for i in range(len(cleaned_varlist))}
        
        return g, J, H, varlist, conlist, HM_row_dict, eigenvalue_dict, H_abs_row_dict, abs_eigenvalue_dict, H_clamped_row_dict, clamped_eigenvalue_dict
    
    def create_ordered_hessian_rows_lists(self, hessian_row_dict):
        x_hess_rows = []
        y_hess_rows = []
        z_hess_rows = []
    
        for i in range(self.lx):
            var_name = self.TRF.xvars[i].name 
            if var_name in hessian_row_dict:
                x_hess_rows.append(hessian_row_dict[var_name])  
    
        for i in range(self.ly):
            var_name = self.TRF.y[i+1].name
            if var_name in hessian_row_dict:
                y_hess_rows.append(hessian_row_dict[var_name])
            
        for i in range(self.lz):
            var_name = self.TRF.zvars[i].name 
            if var_name in hessian_row_dict:
                z_hess_rows.append(hessian_row_dict[var_name]) 

        return x_hess_rows, y_hess_rows, z_hess_rows
    
    def create_ordered_eigenvalue_lists(self, eigenvalue_dict):
        x_eig = []
        y_eig = []
        z_eig = []
    
        for i in range(self.lx):
            var_name = self.TRF.xvars[i].name 
            x_eig.append(abs(eigenvalue_dict.get(var_name, 1e-8)))
    
        for i in range(self.ly):
            var_name = self.TRF.y[i+1].name
            y_eig.append(abs(eigenvalue_dict.get(var_name, 1e-8)))
    
        for i in range(self.lz):
            var_name = self.TRF.zvars[i].name 
            z_eig.append(abs(eigenvalue_dict.get(var_name, 1e-8)))

        return x_eig, y_eig, z_eig

    def criticalityCheck(self, x, y, z, rom_params, g, J, varlist, conlist, worstcase=False, M=[0.0]):

        model = self.model
        
        l = ConcreteModel()
        l.v = Var(varlist, domain=Reals)
        for i in varlist:
            #dummy = model.find_component(i)
            l.v[i] = 0.0
            l.v[i].setlb(-1.0)
            l.v[i].setub(1.0)
        if worstcase:
            if M.all() == 0.0:
                print('WARNING: worstcase criticality was requested but Jacobian error bound is zero')
            l.t = Var(range(0, self.ly), domain=Reals)
            for i in range(0, self.ly):
                l.t[i].setlb(-M[i])
                l.t[i].setub(M[i])

        # def linConMaker(l, i):
        #     # i should be range(len(conlist) - 1)
        #     # because last element of conlist is the objective
        #     con_i = model.find_component(conlist[i])

        #     isEquality = con_i.equality

        #     isROM = False

        #     if conlist[i][:7] == '.' + self.TRF.name + '.rom':
        #         isROM = True
        #         romIndex = int(filter(str.isdigit, conlist[i]))

        #     # This is very inefficient
        #     # Fix this later if these problems get big
        #     # This is the ith row of J times v
        #     Jv = sum(x[2] * l.v[varlist[x[1]]] for x in J if x[0] == i)

        #     if isEquality:
        #         if worstcase and isROM:
        #             return Jv + l.t[romIndex] == 0
        #         else:
        #             return Jv == 0
        #     else:
        #         lo = con_i.lower
        #         up = con_i.upper
        #         # if lo is not None:
        #         #     level = lo.value - con_i.lslack()
        #         #     if up is not None:
        #         #         return (lo.value <= level + Jv <= up.value)
        #         #     else:
        #         #         return (lo.value <= level + Jv)
        #         # elif up is not None:
        #         #     level = up.value - con_i.uslack()
        #         #     return (level + Jv <= up.value)
        #         # else:
        #         #     raise Exception(
        #         #         "This constraint seems to be neither equality or inequality: " + conlist(i))
                
        #         if lo is not None and up is not None:
        #             # Both lower and upper bounds are defined
        #             level = lo.value - con_i.lslack() if hasattr(con_i, 'lslack') else lo.value
        #             return (lo.value <= level + Jv <= up.value)
        #         elif lo is not None:
        #             # Only lower bound is defined
        #             level = lo.value - con_i.lslack() if hasattr(con_i, 'lslack') else lo.value
        #             return (lo.value <= level + Jv)
        #         elif up is not None:
        #             # Only upper bound is defined
        #             level = up.value - con_i.uslack() if hasattr(con_i, 'uslack') else up.value
        #             return (level + Jv <= up.value)
        #         else:
        #             # Neither lower nor upper bound is defined
        #             raise ValueError(
        #                        f"Constraint {conlist[i]} is neither equality nor inequality. It has no bounds."
        #                         )


        # def linConMaker(l, i):
        #     print(f"Processing constraint {i}...")

        #     con_i = model.find_component(conlist[i])
        #     if con_i is None:
        #         print(f"Skipping constraint {i}: Not found in model.")
        #         return Constraint.Skip  # Skip if constraint not found

        #     isEquality = con_i.equality
        #     isROM = False
        #     romIndex = None

        #     if conlist[i][:7] == '.' + self.TRF.name + '.rom':
        #         isROM = True
        #         romIndex = ''.join(filter(str.isdigit, conlist[i]))
        #         romIndex = int(romIndex) if romIndex else None
        #         if romIndex is None:
        #             print(f"Skipping ROM constraint {i}: Invalid index.")
        #             return Constraint.Skip

        #     # Compute Jv safely
        #     try:
        #         Jv = sum(x[2] * l.v[varlist[x[1]]] for x in J if x[0] == i)
        #         print(f"Jv for constraint {i}: {Jv}")
        #     except Exception as e:
        #         print(f"Error computing Jv for constraint {i}: {e}")
        #         return Constraint.Skip

        #     ### Handling Equality Constraints ###
        #     if isEquality:
        #         print(f'Equality Constraint {i}: Jv = {Jv}')
        #         return Jv == 0

        #     ### Handling Inequality Constraints ###
        #     lo = con_i.lower
        #     up = con_i.upper

        #     if lo is not None:
        #         level = value(lo) - con_i.lslack() if hasattr(con_i, 'lslack') else value(lo)
        #         if up is not None:
        #             print(f'Bound Constraint {i}: {lo} <= Jv <= {up}')
        #             return inequality(level, Jv, value(up))  
        #         else:
        #             print(f'Lower Bound Constraint {i}: Jv >= {lo}')
        #             return Jv >= level

        #     elif up is not None:
        #         level = value(up) - con_i.uslack() if hasattr(con_i, 'uslack') else value(up)
        #         print(f'Upper Bound Constraint {i}: Jv <= {up}')
        #         return Jv <= level

        #     print(f'Constraint {i} is skipped')
        #     return Constraint.Skip

      
        
        def linConMaker(l, i):
    
            # Retrieve the constraint component
            con_i = model.find_component(conlist[i])
            
            if con_i is None:
                print(f"⚠️ Constraint {i} not found, skipping!")
                return Constraint.Skip
            
            # Check if the constraint is an equality
            isEquality = con_i.equality

            # Check if the constraint is a ROM constraint
            isROM = False
            romIndex = None  # Initialize properly

            if conlist[i][:7] == '.' + self.TRF.name + '.rom':
                isROM = True
                romIndex_str = ''.join(filter(str.isdigit, conlist[i]))
                if romIndex_str: # Check if romIndex_str is not empty
                    romIndex = int(romIndex_str)
                else:
                    print(f"⚠️ Skipping ROM constraint {i}: Invalid index!")
                    return Constraint.Skip  # Skip if romIndex is invalid

            # Compute the linear expression Jv
            try:
                Jv = sum(x[2] * l.v[varlist[x[1]]] for x in J if x[0] == i)
            except Exception as e:
                print(f"❌ Error computing Jv for constraint {i}: {e}")
                return Constraint.Skip  

            # Handle equality constraints
            if isEquality:
                if worstcase and isROM and romIndex is not None:
                    return Jv + l.t[romIndex] == 0  # Return a valid equality constraint
                else:
                    return Jv == 0  # Return a valid equality constraint

            # Handle inequality constraints
            else:
                lo = con_i.lower
                up = con_i.upper
                
                if lo is not None and up is not None:
                    # Both lower and upper bounds are defined
                    level = lo.value - con_i.lslack() if hasattr(con_i, 'lslack') else lo.value
                    return (lo.value, level + Jv, up.value) # Return a valid range constraint
                elif lo is not None:
                    # Only lower bound is defined
                    level = lo.value - con_i.lslack() if hasattr(con_i, 'lslack') else lo.value
                    return inequality(lo.value, level + Jv, None)  # Return a valid inequality constraint
                elif up is not None:
                    # Only upper bound is defined
                    level = up.value - con_i.uslack() if hasattr(con_i, 'uslack') else up.value
                    return inequality(None, level + Jv, up.value) # Return a valid inequality constraint
                else:
                    # Neither lower nor upper bound is defined
                    raise ValueError(
                         f"Constraint {conlist[i]} is neither equality nor inequality. It has no bounds."
                            )

        try:
            l.lincons = Constraint(range(len(conlist)-1), rule=linConMaker)
        except Exception as e:
            ValueError(f"❌ Error defining constraints: {e}")

        try:
            l.obj = Objective(expr=sum(x[1] * l.v[varlist[x[0]]] for x in g if varlist[x[0]] in l.v))
        except Exception as e:
            ValueError(f"❌ Error defining objective function: {e}")

        # l.lincons = Constraint(range(len(conlist)-1), rule=linConMaker)        
        # l.obj = Objective(expr=sum(x[1] * l.v[varlist[x[0]]] for x in g if varlist[x[0]] in l.v))

        # Calculate gradient norm for scaling purposes
        gfnorm = sqrt(sum(x[1]**2 for x in g))


        opt = SolverFactory(self.config.solver)
        opt.options.update(self.config.solver_options)
        ##
        opt.options['halt_on_ampl_error'] = 'yes'
        # opt.options['solver_options']['linear_solver'] = 'mumps'  # Other options: 'ma27', 'ma57', 'pardiso'
        opt.options['max_iter'] = 20000
        ##
        results = opt.solve(
            l, keepfiles=self.keepfiles, tee=self.stream_solver)

        if ((results.solver.status == SolverStatus.ok)
                and (results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible])):
        # if (results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible]):
            l.solutions.load_from(results)
            if gfnorm > 1:
                return True, abs(l.obj())/gfnorm
            else:
                return True, abs(l.obj())
        else:
            print("Waring: Crticality check fails with solver Status: " + str(results.solver.status))
            print("And Termination Conditions: " + str(results.solver.termination_condition))
            return False, inf




    ####################### Build ROM ####################

    def initialQuad(self, lx):
        _, self.pset, self.geoM = generate_quadratic_rom_geometry(lx)

    def buildROM(self, x, radius_base):
        """
        This function builds a linear ROM near x based on the perturbation.
        The ROM is returned by a format of params array.
        I think the evaluate count is broken here!
        """

        y1 = self.evaluateDx(x)
        rom_params = []
        
        if(self.romtype==ROMType.linear):
            
            dim = int(self.lx* + 1)
            FEs = dim * int(self.ly)
            
            for i in range(0, self.ly):
                rom_params.append([])
                rom_params[i].append(y1[i])

                # Check if it works with Ampl
                fcn  =  self.TRF.external_fcns[i]._fcn
                values = [];
                for j in self.exfn_xvars_ind[i]:
                    values.append(x[j])

                for j in range(0, len(values)):
                    radius = radius_base # * scale[j]
                    values[j] = values[j] + radius
                    y2 = fcn._fcn(*values)
                    rom_params[i].append((y2 - y1[i]) / radius)
                    values[j] = values[j] - radius

        elif(self.romtype==ROMType.quadratic):
            #Quad ROM
            # basis = [1, x1, x2,..., xn, x1x1, x1x2,x1x3,...,x1xn,x2x2,x2x3,...,xnxn]
            if self.geoM is None:
                self.initialQuad(self.lx)

            dim = int((self.lx*self.lx+self.lx*3)/2. + 1)
            FEs = dim * int(self.ly)
            
            rhs=[]
            radius = radius_base #*np.array(scale)
            for p in self.pset[:-1]:
                y = self.evaluateDx(x+radius*p)
                rhs.append(y)
            rhs.append(y1)

            coefs = np.linalg.solve(self.geoM,np.matrix(rhs))
            for i in range(0, self.ly):
                rom_params.append([])
                for j in range(0, dim):
                    rom_params[i].append(coefs[j,i])
                for j in range(1, self.lx+1):
                    rom_params[i][j]=rom_params[i][j]/radius#/radius[j-1]
                count = self.lx+1
                for ii in range(0, self.lx):
                    for j in range(ii, self.lx):
                        rom_params[i][count]=rom_params[i][count]/radius#/radius[ii]/radius[j]
                        count = count + 1

        elif(self.romtype==ROMType.quadratic_simp):
            #Quad ROM
            # basis = [1, x1, x2,..., xn, x1x1, x1x2,x1x3,...,x1xn,x2x2,x2x3,...,xnxn]
            if self.geoM is None:
                self.initialQuad(self.lx)

            dim = int((2*self.lx) + 1)
            FEs = dim * int(self.ly)
            
            rhs=[]
            radius = radius_base #*np.array(scale)
            for p in self.pset[:-1]:
                y = self.evaluateDx(x+radius*p)
                rhs.append(y)
            rhs.append(y1)

            coefs = np.linalg.solve(self.geoM,np.matrix(rhs))
            for i in range(0, self.ly):
                rom_params.append([])
                for j in range(0, dim):
                    rom_params[i].append(coefs[j,i])
                for j in range(1, self.lx+1):
                    rom_params[i][j]=rom_params[i][j]/radius#/radius[j-1]
                count = self.lx+1
                for ii in range(0, self.lx):
                    rom_params[i][count]=rom_params[i][count]/radius#/radius[ii]/radius[j]
                    count = count + 1
        
        elif self.romtype == ROMType.ts:
            dim = 1
            FEs = dim*int(self.ly)
        
        elif(self.romtype==ROMType.ts_linear):
              
            dim = int(self.lx* + 1)
            FEs = dim * int(self.ly)
              
            for i in range(0, self.ly):
                rom_params.append([])
                rom_params[i].append(y1[i])

                # Check if it works with Ampl
                fcn  =  self.TRF.external_fcns[i]._fcn
                values = [];
                for j in self.exfn_xvars_ind[i]:
                    values.append(x[j])

                for j in range(0, len(values)):
                    radius = radius_base # * scale[j]
                    values[j] = values[j] + radius
                    y2 = fcn._fcn(*values)
                    rom_params[i].append((y2 - y1[i]) / radius)
                    values[j] = values[j] - radius
        
        elif(self.romtype==ROMType.ts_quadratic):
            #Quad ROM
            # basis = [1, x1, x2,..., xn, x1x1, x1x2,x1x3,...,x1xn,x2x2,x2x3,...,xnxn]
            if self.geoM is None:
                self.initialQuad(self.lx)

            dim = int((self.lx*self.lx+self.lx*3)/2. + 1)
            FEs = dim * int(self.ly)
            
            rhs=[]
            radius = radius_base #*np.array(scale)
            for p in self.pset[:-1]:
                y = self.evaluateDx(x+radius*p)
                rhs.append(y)
            rhs.append(y1)

            coefs = np.linalg.solve(self.geoM,np.matrix(rhs))
            for i in range(0, self.ly):
                rom_params.append([])
                for j in range(0, dim):
                    rom_params[i].append(coefs[j,i])
                for j in range(1, self.lx+1):
                    rom_params[i][j]=rom_params[i][j]/radius#/radius[j-1]
                count = self.lx+1
                for ii in range(0, self.lx):
                    for j in range(ii, self.lx):
                        rom_params[i][count]=rom_params[i][count]/radius#/radius[ii]/radius[j]
                        count = count + 1
                        
        elif(self.romtype==ROMType.ts_quadratic_simp):
            #Quad ROM
            # basis = [1, x1, x2,..., xn, x1x1, x1x2,x1x3,...,x1xn,x2x2,x2x3,...,xnxn]
            if self.geoM is None:
                self.initialQuad(self.lx)

            dim = int((2*self.lx) + 1)
            FEs = dim * int(self.ly)
            
            rhs=[]
            radius = radius_base #*np.array(scale)
            for p in self.pset[:-1]:
                y = self.evaluateDx(x+radius*p)
                rhs.append(y)
            rhs.append(y1)

            coefs = np.linalg.solve(self.geoM,np.matrix(rhs))
            for i in range(0, self.ly):
                rom_params.append([])
                for j in range(0, dim):
                    rom_params[i].append(coefs[j,i])
                for j in range(1, self.lx+1):
                    rom_params[i][j]=rom_params[i][j]/radius#/radius[j-1]
                count = self.lx+1
                for ii in range(0, self.lx):
                    rom_params[i][count]=rom_params[i][count]/radius#/radius[ii]/radius[j]
                    count = count + 1
                
        elif(self.romtype in [ROMType.GP, ROMType.ts_GP]):
        
            C = 2
            dim = int((2 * self.lx) + 1)
            FEs = dim * int(self.ly)
            
            radius = radius_base
            
            for i in range(0, self.ly):
                rom_params.append([])
            
                fcn  =  self.TRF.external_fcns[i]._fcn
                ind = self.exfn_xvars_ind[i]
                
                x_points = []
                y_points = []
                GP_param = []
                
                
                x_init = [x[j] for j in self.exfn_xvars_ind[i]]
                
                x_points.append(x_init)  # Center point
                y_points.append([fcn._fcn(*x_init)])
                
                x_points.append([xi + radius for xi in x_init])  # Point at radius_base
                y_points.append([fcn._fcn(*x_points[-1])])
                
                x_points.append([xi + C * radius for xi in x_init])  # Point at C * radius_base
                y_points.append([fcn._fcn(*x_points[-1])])
                
                for scale in np.linspace(1.0 / (dim - 2), (dim - 3) / (dim - 2), dim - 3):  
                    new_point = [xi + scale * C * radius for xi in x_init]
                    x_points.append(new_point)
                    y_points.append([fcn._fcn(*new_point)])
            
                rom_params[i] = [x_points, y_points] 
                
                sigma, length_scale, m, covM, M = GPR(x_points, y_points, x_points, kernel_type)
                
                KM = list(covM@M)
                
                GP_param.append(sigma)
                GP_param.append(length_scale)

                rom_params[i] = [x_points, y_points, GP_param, KM]
                
        elif self.romtype == ROMType.hybrid_ts_GP:
            dim = 4
            FEs = dim * int(self.ly)
            
            rom_params = [float(radius_base) for _ in range(self.ly)]   
                
        return rom_params, y1, FEs
        
            

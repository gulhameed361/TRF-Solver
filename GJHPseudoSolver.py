# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:56:50 2024

@author: gh00616
"""

import logging

from pyomo.common.tempfiles import TempfileManager

from pyomo.common.download import DownloadFactory
from pyomo.opt.base import SolverFactory, OptSolver
from pyomo.solvers.plugins.solvers.ASL import ASL

from pyomo.common.config import ( 
    ConfigBlock, ConfigValue, 
    PositiveInt, PositiveFloat, 
    NonNegativeFloat, In)
from pyomo.core import Var, value

from readgjh import readgjh
import getGJH

logger = logging.getLogger('TRF Algorithm')
#fh = logging.FileHandler('debug_vars.log')
#logger.setLevel(logging.DEBUG)
#logger.addHandler(fh)

def load():
    pass

DownloadFactory.register('gjh')(getGJH.get_gjh)


@SolverFactory.register('contrib.gjh', doc='Interface to the AMPL GJH "solver"')
class GJHSolver(ASL):
    """An interface to the AMPL GJH "solver" for evaluating a model at a
    point."""

    def __init__(self, **kwds):
        kwds['type'] = 'gjh'
        kwds['symbolic_solver_labels'] = True
        super(GJHSolver, self).__init__(**kwds)
        self.options.solver = 'gjh'
        self._metasolver = False

    # A hackish way to hold on to the model so that we can parse the
    # results.
    def _initialize_callbacks(self, model):
        self._model = model
        self._model._gjh_info = None
        super(GJHSolver, self)._initialize_callbacks(model)

    def _presolve(self, *args, **kwds):
        super(GJHSolver, self)._presolve(*args, **kwds)
        self._gjh_file = self._soln_file[:-3]+'gjh'
        TempfileManager.add_tempfile(self._gjh_file, exists=False)

    def _postsolve(self):
        #
        # TODO: We should return the information using a better data
        # structure (ComponentMap? so that the GJH solver does not need
        # to be called with symbolic_solver_labels=True
        #
        self._model._gjh_info = readgjh(self._gjh_file)
        self._model = None
        return super(GJHSolver, self)._postsolve()


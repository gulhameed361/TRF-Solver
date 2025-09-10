# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:56:50 2024

@author: gh00616
"""

# local_gjh_fallback.py
import logging
import os
import shutil
from pyomo.opt.base import SolverFactory
from pyomo.solvers.plugins.solvers.ASL import ASL
from pyomo.common.tempfiles import TempfileManager
from readgjh import readgjh
from distutils.spawn import find_executable

logger = logging.getLogger('local.gjh')

# Name of gjh executable
GJH_FILENAME = "gjh.exe" if os.name == "nt" else "gjh"
GJH_CWD_PATH = os.path.join(os.getcwd(), GJH_FILENAME)  # current working directory fallback


@SolverFactory.register('local.gjh', doc='Local GJH solver (system PATH or cwd fallback)')
class LocalGJHSolver(ASL):
    def __init__(self, **kwds):
        kwds['type'] = 'gjh'
        kwds['symbolic_solver_labels'] = True
        super().__init__(**kwds)
        self._metasolver = False

    def available(self, exception_flag=True):
        """Find gjh in PATH first, fallback to cwd."""
        # Try to find gjh in system PATH
        path_gjh = find_executable("gjh")
        if path_gjh:
            self.options.solver = path_gjh
            return super().available(exception_flag=exception_flag)

        # Fallback to current working directory
        if os.path.isfile(GJH_CWD_PATH):
            self.options.solver = GJH_CWD_PATH
            return super().available(exception_flag=exception_flag)

        # Not found anywhere
        if exception_flag:
            raise RuntimeError(
                f"GJH solver not found! Tried system PATH and current directory ({GJH_CWD_PATH})."
            )
        return False

    def _initialize_callbacks(self, model):
        self._model = model
        self._model._gjh_info = None
        super()._initialize_callbacks(model)

    def _presolve(self, *args, **kwds):
        super()._presolve(*args, **kwds)
        self._gjh_file = self._soln_file[:-3] + 'gjh'
        TempfileManager.add_tempfile(self._gjh_file, exists=False)

    def _postsolve(self):
        if not os.path.exists(self._gjh_file) or \
           not os.path.exists(self._gjh_file[:-3] + 'col') or \
           not os.path.exists(self._gjh_file[:-3] + 'row'):
            raise RuntimeError(
                f"GJH failed to produce .gjh, .col, or .row files. "
                f"Check the solver log."
            )
        self._model._gjh_info = readgjh(self._gjh_file)
        self._model = None
        return super()._postsolve()





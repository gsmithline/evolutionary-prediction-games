import numpy as np
import pandas as pd

import functools
import itertools

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts,amssymb,bm}',
})

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

import scipy.stats

from tqdm.auto import tqdm

import evoml
import evoml.analysis

param_tracker = evoml.ParamTracker()
param = param_tracker.store

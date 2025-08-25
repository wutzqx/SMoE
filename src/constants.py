from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'GEC': [(0.99995, 1.04), (0.99995, 1.06)],
		'PSM': [(0.99995, 1.04), (0.99995, 1.06)],
		'Genesis': [(0.99995, 1.04), (0.99995, 1.06)],
		'PUMP': [(0.99995, 1.04), (0.99995, 1.06)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

lr_d = {
		'SMD': 0.0001,
		'PSM': 0.01,
		'GEC': 0.09,
		'GHL': 0.0001,
		'PUMP': 0.0001,
		'Genesis': 0.08,
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
		'MSDS': 0.001,
		'MBA': 0.001, 
	}
lr = lr_d[args.dataset]

percentiles = {
		'SMD': (98, 2000),
		'PSM': (98, 2000),
		'GEC': (98, 2000),
		'GHL': (98, 2000),
		'PUMP': (98, 2000),
		'Genesis': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9
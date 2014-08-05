from __future__ import division
import numpy as np
import sys, os, gzip, warnings
import cPickle as pickle

import pymouse.models.svi
from pymouse.scripts.consistency import get_changepoints, \
        get_experiments, get_behavior, mask_data, diff_and_renorm

warnings.simplefilter("ignore")

CACHING = True

frames_to_load_per_mouse=5000
n_points_for_init=10000
prefit_portion=0.25

Nmax = 80
Nsubmax = 8
Ndim = 32

train_experiments = [
        'C57-10_blank_25percenttmtindpg_6-3-13_pt2',
        'C57-10_tmttopright_25percenttmtindpg_6-3-13_pt2',
        'C57-1_blank_25percenttmtindpg_6-3-13',
        'C57-1_tmttopright_25percenttmtindpg_6-3-13',
        'C57-2_blank_25percenttmtindpg_6-3-13',
        'C57-2_tmttopright_25percenttmtindpg_6-3-13',
        'C57-3_blank_25percenttmtindpg_6-3-13',
        'C57-3_tmttopright_25percenttmtindpg_6-3-13',
        ]

def load_data():
    if CACHING and os.path.isfile('data_loader_cache.pkl.gz'):
        with gzip.open('data_loader_cache.pkl.gz','r') as infile:
            data, changepoints, group_ids = pickle.load(infile)
    else:
        ### load the data files
        store_name = '/scratch/6-3-13 25% TMT in DPG.h5'
        experiments = [s for s in get_experiments(store_name) if 'C57-5' not in s]

        ### load the data
        df, arrays = get_behavior(store_name=store_name,
                                experiments=train_experiments,
                                query="index<%d"%frames_to_load_per_mouse,
                                array_names=["data"],
                                median_kernel=[(i*2+1,1) for i in range(1,10)],
                                normalize_data=True)

        data = arrays['data']
        data = diff_and_renorm(data)

        ### chunk it up
        data, changepoints, group_ids = _gen_data(data, df, 'gibbs')

        ### reduce dimensionality
        for d in data:
            assert d.ndim == 2 and d.shape[1] == 600
        reducer = np.random.normal(size=(600,Ndim))
        data = [d.dot(reducer) for d in data]

    return data, changepoints, group_ids

def _gen_data(data, df, method):
    splits = []
    for mouse_name in df.mouse_name.unique():
        splits.append(np.argwhere(df.mouse_name.str.contains(mouse_name))[0][0])
    splits = np.sort(splits)[1:]

    data = np.array_split(data,splits)
    changepoints = [get_changepoints(d, 6,0.1,1.5)[1] for d in data]
    group_ids = np.array(["_tmt" in s
        for s in df.mouse_name.values[np.r_[splits-1,splits[-1]+1]]], dtype='int32')

    return data, changepoints, group_ids

def get_model(data,changepoints,group_ids):
    pymousemodel = pymouse.models.svi.GMMHSMM(
            Nmax=Nmax, Nsubmax=Nsubmax,
            method='gibbs',
            n_iter=0, max_r=8, n_points_for_init=n_points_for_init,
            prefit_portion=prefit_portion, n_cpu=0)

    pymousemodel.fit(data,changepoints,group_ids)

    return pymousemodel.posteriormodel_


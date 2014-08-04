#!/usr/bin/env python
from __future__ import division
import numpy as np
from numpy.random import rand
import gzip, cPickle, glob, re, logging
from operator import xor
from mpi4py import MPI

import data_loader

np.random.seed(0)

niter = 250
nsamples_between_swaps = 2
save_every = 1
basetemp = 1.005

log_options = dict(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        )
savedir = '/data'

# TODO more than one model per mpi process
# TODO monitoring script

def swap_samples(comm,model,swapcounts,itr):
    rank = comm.rank
    parity = itr % 2

    E1, T1, x1 = model.energy, model.temperature, model.get_sample()

    if rank % 2 == parity and rank < comm.size-1:
        comm.send((E1,T1,x1), dest=rank+1)
        x2 = comm.recv(source=rank+1)
        if x2 is not None:
            model.set_sample(x2)
            swapcounts[(comm.rank,comm.rank+1)] += 1
            logging.info('SWAP with higher temperature')
        else:
            logging.info('no swap with higher temperature')
    elif rank % 2 != parity and rank > 0:
        E2, T2, x2 = comm.recv(source=rank-1)
        swap_logprob = min(0.,(E1-E2)*(1./T1 - 1./T2))
        if np.log(rand()) < swap_logprob:
            comm.send(x1, dest=rank-1)
            model.set_sample(x2)
            swapcounts[(comm.rank-1,comm.rank)] += 1
            logging.info('SWAP with lower temperature')
        else:
            comm.send(None, dest=rank-1)
            logging.info('no swap with lower temperature')

def save_sample(comm,model,swapcounts,itr):
    filename = os.path.join(savedir,'sample_%03d_%05d.pkl.gz' % (comm.rank,itr))
    sample = model.get_sample()
    rngstate = np.random.get_state()
    with gzip.open(filename,'w') as outfile:
        cPickle.dump((sample,swapcounts,rngstate),outfile,protocol=-1)
    logging.info('saved sample in %s' % filename)

def load_latest_sample(comm):
    model = data_loader.get_model(*data_loader.load_data())
    model.temperature = basetemp**comm.rank

    filenames = glob.glob(os.path.join(savedir,'sample_%03d_*.pkl.gz' % comm.rank))
    if len(filenames) == 0:
        swapcounts = {(comm.rank,comm.rank+1):0,(comm.rank-1,comm.rank):0}
        niter_complete = 0
        logging.info('starting fresh chain')
    else:
        filename = sorted(filenames)[-1]
        niter_complete = int(re.findall('\d+',filename)[-1])
        with gzip.open(filename,'r') as infile:
            sample, swapcounts, rngstate = cPickle.load(infile)
        model.set_sample(sample)
        np.random.set_state(rngstate)
        logging.info('loaded chain state from %s' % filename)

    return model, swapcounts, niter_complete

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    logging.basicConfig(filename='%03d.log' % comm.rank, **log_options)

    model, swapcounts, start_iter = load_latest_sample(comm)

    for itr in xrange(start_iter,niter):
        for itr2 in xrange(nsamples_between_swaps):
            model.resample_model()
        swap_samples(comm,model,swapcounts,itr)
        if itr % save_every == 0:
            save_sample(comm,model,swapcounts,itr)

    comm.Barrier()


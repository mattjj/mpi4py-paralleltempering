from __future__ import division
import numpy as np
from numpy.random import rand
from operator import xor
import gzip, cPickle, glob, re
from mpi4py import MPI

import data_loader

niter = 25
nsamples_between_swaps = 1
save_every = 1
basetemp = 1.05

np.random.seed(0)

# TODO logging
# TODO more than one model per mpi process? that decouples mpi processes from
# temperatures and saves memory. but it makes the logic harder since we have to
# think about local vs cross-machine swaps.

def swap_samples(comm,model,itr):
    rank = comm.rank
    parity = itr % 2

    E1, T1, x1 = model.energy, model.temperature, model.get_sample()

    if rank % 2 == parity and rank < comm.size-1:
        comm.send((E1,T1,x1), dest=rank+1)
        x2 = comm.recv(source=rank+1)
        if x2 is not None:
            model.set_sample(x2)
    elif rank % 2 != parity and rank > 0:
        E2, T2, x2 = comm.recv(source=rank-1)
        swap_logprob = min(0.,(E1-E2)*(1./T1 - 1./T2))
        if np.log(rand()) < swap_logprob:
            comm.send(x1, dest=rank-1)
            model.set_sample(x2)
        else:
            comm.send(None, dest=rank-1)

def save_sample(comm,model,itr):
    filename = 'sample_%03d_%05d.pkl.gz' % (comm.rank,itr)
    sample = model.get_sample()
    rngstate = np.random.get_state()
    with gzip.open(filename,'w') as outfile:
        cPickle.dump((sample,rngstate),outfile,protocol=-1)

def load_latest_sample(comm):
    model = data_loader.get_model(*data_loader.load_data())
    model.temperature = basetemp**comm.rank

    filenames = glob.glob('sample_%03d_*.pkl.gz' % comm.rank)
    if len(filenames) == 0:
        niter_complete = 0
    else:
        filename = sorted(filenames)[-1]
        niter_complete = int(re.findall('\d+',filename)[-1])
        with gzip.open(filename,'r') as infile:
            sample, rngstate = cPickle.load(infile)
        model.set_sample(sample)
        np.random.set_state(rngstate)

    return model, niter_complete

if __name__ == '__main__':
    comm = MPI.COMM_WORLD

    model, start_iter = load_latest_sample(comm)

    for itr in xrange(start_iter,niter):
        for itr2 in xrange(nsamples_between_swaps):
            model.resample_model()
        swap_samples(comm,model,itr)
        if itr % save_every == 0:
            save_sample(comm,model,itr)

    comm.Barrier()


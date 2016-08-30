"""
Gaussian-Bernoulli Restricted Boltzmann Machine
"""

# Authors: Taku Yoshioka <taku.yoshioka.4096@gmail.com>
# License: BSD 3 clause

import copy
import numpy as np
import scipy
import time

from theano import function, shared
from theano import config as cnf
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class GBRBMTrainingParams:
    def __init__(
        self, n_components=4, n_rate=1, algorithm='cd', str_cost='energy', 
        seed=1, verbose=True, n_mod=100, batch_size=100, n_particles=100, 
        lmd_w=0.0, lmd_h=0.0, alpha=0.0, eps=1e-2, decay=1.0,  
        is_update={'w': True, 'z': False, 'b': True, 'c': True},  
        init_w={'scale': 0.01,         'distribution': 'normal'}, 
        init_z={'scale': np.log(0.01), 'distribution': 'const'}, 
        init_b={'scale': 0.01,         'distribution': 'normal'}, 
        init_c={'scale': 0.01,         'distribution': 'normal'}):
        # Model parameters
        self.n_components = n_components
        self.n_rate       = n_rate

        # Learning coefficient parameters
        self.alpha = alpha
        self.eps   = eps
        self.decay = decay

        # Algorithm parameters
        self.algorithm  = algorithm
        self.seed       = seed
        self.batch_size = batch_size

        # Sampling parameters
        self.n_particles = n_particles

        # Cost function parameters
        self.lmd_w    = lmd_w
        self.lmd_h    = lmd_h
        self.str_cost = str_cost

        # Display parameters
        self.verbose = verbose
        self.n_mod   = n_mod

        # Initialization and update of parameters
        self.init_w    = init_w
        self.init_z    = init_z
        self.init_b    = init_b
        self.init_c    = init_c
        self.is_update = is_update

class GBRBM:
    def __init__(self, params_tr, params_model, t_start, t_end):
        self.params_tr    = params_tr
        self.params_model = params_model
        self.t_start      = t_start
        self.t_end        = t_end

    def fit(self, xs, xs_=None):
        """
        xs : numpy array n_tr x D, training data
        xs_: numpy array n_te x D, test data 
        """
        # Aliases
        params_tr    = self.params_tr
        params_model = self.params_model
        batch_size   = self.params_tr.batch_size
        n_samples    = xs.shape[0]
        d_vis        = xs.shape[1]
        t_start      = self.t_start
        t_end        = self.t_end

        # Initialization
        params_np, hist = _init_params_all(d_vis, params_tr, params_model)
        ixss, bsize     = _get_minibatches(batch_size, n_samples)

        # Create Theano objects
        funcs_th, params_th, _ = _init_theano(xs, xs_, params_np, params_tr)

        # Training loop
        if params_tr.algorithm == 'cd':
            hist = _cd_learning(
                hist, funcs_th, params_th, params_tr, t_start, t_end, ixss)
        elif params_tr.algorithm == 'pcd':
            _init_particles(funcs_th, n_samples, params_tr)
            hist = _pcd_learning(
                hist, funcs_th, params_th, params_tr, t_start, t_end, ixss)
        else:
            raise ValueError(
                'Invalid string of algorithm: ' + params_tr.algorithm)

        # Copy theano objects to numpy array
        self.params_np = _copy_params(params_th)

        # Keep history of training
        self.hist = hist

        # Model parameter set
        self.params_model = self.params_np
        self.params_model.update({'hist': hist})

        return self
        
    def reconstruct(self, xs):
        """
        Reconstruct given data
        """
        w, z, b, _ = _get_params(self.params_model)
        hs         = self.transform(xs)
        ms, _      = _meanstd_vs_given_hs_np(hs, w, z, b)

        return ms

    def transform(self, xs):
        """
        Transform visible variables to hidden variables
        """
        w, z, _, c = _get_params(self.params_model)
        n_rate     = self.params_tr.n_rate
        vs         = xs

        hs = _mean_hs_given_vs_np(vs, w, z, c) * n_rate

        return hs

    def get_str(self, t_start=None, t_end=None):
        sep = '-'

        s  =       'ncmp_' + str(self.params_tr.n_components)
        s += sep + 'algo_' + str(self.params_tr.algorithm)
        s += sep + 'cost_' + str(self.params_tr.str_cost)
        s += sep + 'nbtc_' + str(self.params_tr.batch_size)
        s += sep + 'npar_' + str(self.params_tr.n_particles)
        s += sep + 'lmdw_' + str(self.params_tr.lmd_w)
        s += sep + 'lmdh_' + str(self.params_tr.lmd_h)
        s += sep + 'alph_' + str(self.params_tr.alpha)
        s += sep + 'eps_'  + str(self.params_tr.eps)
        s += sep + 'decy_' + str(self.params_tr.decay)

        return s




"""
Utility functions: RBM training loop
"""

def _cd_learning(
    hist, funcs_th, params_th, params_tr, t_start, t_end, ixss):

    if params_tr.verbose:
        print('CD-1 learning loop: %d steps' % (t_end - t_start))
        begin = time.clock()

    for t in xrange(t_start, t_end):
        # Minibatch learning
        for ixs in ixss:
            funcs_th['update_diffs'](t, ixs)
            funcs_th['update_params']()

        # Log
        if params_tr.verbose and np.mod((t + 1), params_tr.n_mod) == 0:
            end   = time.clock()
            log   = _show_info(funcs_th['get_info'], t, end - begin)
            begin = time.clock()
            hist  += [log]

    return hist

def _pcd_learning(
    hist, funcs_th, params_th, params_tr, t_start, t_end, ixss):
    # Initialization of persistent samples
    n_particles = params_tr.n_particles
    n_samples   = np.max([np.max(ixs) for ixs in ixss])
    _ixs        = np.random.permutation(n_samples)[:n_particles]
    funcs_th['init_particles'](_ixs)

    if params_tr.verbose:
        print('PCD-1 learning loop: %d steps' % (t_end - t_start))
        begin = time.clock()

    for t in xrange(t_start, t_end):
        # Minibatch loop
        for ixs in ixss:
            funcs_th['update_diffs'](t, ixs)
            funcs_th['update_params']()

        # Log
        if params_tr.verbose and np.mod((t + 1), params_tr.n_mod) == 0:
            end   = time.clock()
            log   = _show_info(funcs_th['get_info'], t, end - begin)
            begin = time.clock()
            hist  += [log]

    return hist

def _init_particles(funcs_th, n_samples, params_tr):
    ixs = np.random.permutation(n_samples)[:params_tr.n_particles]
    funcs_th['init_particles'](ixs)

def _show_info(get_info, i, t):
    info = get_info()

    err_tr, err_vl                 = info[0], info[1]
    mean_w, mean_z, mean_b, mean_c = info[2], info[3], info[4], info[5]
    max_w, max_z, max_b, max_c     = info[6], info[7], info[8], info[9]

    print('Iteration %d, Error (train, test) = (%f %f), time = %.2fs'
          % (i, err_tr, err_vl, t))
    print('Means (w, z, b, c) = (%+1.2e, %+1.2e, %+1.2e, %+1.2e)'
          % (mean_w, mean_z, mean_b, mean_c))
    print('Maxs  (w, z, b, c) = (%+1.2e, %+1.2e, %+1.2e, %+1.2e)'
          % (max_w, max_z, max_b, max_c))

    return t, err_tr, err_vl



"""
Utility function: functions for parameters
"""

def _init_params_all(d_vis, params_tr, params_model):
    # Initialize random seed
    np.random.seed(params_tr.seed)

    if params_model is None:
        # Meta parameters
        d_hid = params_tr.n_components

        # Random initialization
        w  = _init_params(params_tr.init_w, (d_vis, d_hid))
        z  = _init_params(params_tr.init_z, (d_vis, ))
        b  = _init_params(params_tr.init_b, (d_vis, ))
        c  = _init_params(params_tr.init_c, (d_hid, ))
        dw = np.zeros((d_vis, d_hid))
        dz = np.zeros(d_vis)
        db = np.zeros(d_vis)
        dc = np.zeros(d_hid)

        # Log of learning
        hist = []
    else:
        w  = params_model['w']
        z  = params_model['z']
        b  = params_model['b']
        c  = params_model['c']
        dw = params_model['dw']
        dz = params_model['dz']
        db = params_model['db']
        dc = params_model['dc']

        hist = params_model['hist']

    return {'w' : w, 'z'  : z, 'b'  : b, 'c'  : c, 
            'dw': dw, 'dz': dz, 'db': db, 'dc': dc}, hist

def _init_params(init_p, size):
    s = init_p['scale']

    if init_p['distribution'] == 'normal':
        return np.random.normal(scale=s, size=size)
    elif init_p['distribution'] == 'const':
        return s * np.ones(size)
    else:
        raise ValueError("Invalid string of init_p['distribution']: " +
                         init_p['distribution'])

def _get_params(params):
    return (params['w'], params['z'], params['b'], params['c'])

def _get_params_np(params):
    return (params['w'], params['z'], params['b'], params['c'])

def _get_diffs(diffs):
    return (diffs['dw'], diffs['dz'], diffs['db'], diffs['dc'])

def _copy_params(params_th):
    w  = params_th['w'].get_value()
    z  = params_th['z'].get_value()
    b  = params_th['b'].get_value()
    c  = params_th['c'].get_value()
    dw = params_th['dw'].get_value()
    dz = params_th['dz'].get_value()
    db = params_th['db'].get_value()
    dc = params_th['dc'].get_value()

    params_np = {'w' : w,  'z' : z,  'b' : b,  'c' : c, 
                 'dw': dw, 'dz': dz, 'db': db, 'dc': dc}

    return params_np



"""
Utility functions: make Theano graph main
"""

def _init_theano(xs_tr_np, xs_vl_np, params_np, params_tr):
    # Alias
    algorithm = params_tr.algorithm
    n_rate    = params_tr.n_rate 

    # Theano shared variables, symbols and random number generator
    xs_tr, xs_vl = _get_shared_vars(xs_tr_np, xs_vl_np)
    params       = _get_shared_params(params_np)
    diffs        = _get_shared_diffs(params_np)
    particles    = _get_shared_particles(params_tr)
    v0s, ixs, t  = _get_symbols()
    rng          = RandomStreams(seed=params_tr.seed)

    # Sampling for positive and negative phase
    poss = _get_poss(rng, v0s, params, n_rate)
    negs = _get_negs(rng, poss, params, n_rate)

    # Cost functions (energy + penalty term, see _get_cost)
    cost, consts = _get_cost(poss, negs, params, params_tr)

    # Updates of model parameters
    us_diff  = _get_updates_diff(t, cost, diffs, params, params_tr, consts)
    us_param = _get_updates_param(diffs, params, params_tr)

    # Concatenation of updates
    if algorithm == 'pcd':
        us_particles = _get_updates_particles(particles, negs)
        us_diff      = us_diff + us_particles

    us = us_diff

    # Reconstruction error
    recon_err_tr = _get_recon_err(xs_tr, params, n_rate)
    recon_err_vl = _get_recon_err(xs_vl, params, n_rate)

    # Information for monitoring learning progress
    info = _get_info(params)

    # Initialization of persistent chain
    us_init_particles = _get_updates_init_particles(
        rng, particles, v0s, params, params_tr)

    # Theano functions
    return _get_theano_functions(
        params, diffs, particles, algorithm, us, us_param, us_init_particles, 
        t, ixs, v0s, xs_tr, recon_err_tr, recon_err_vl, info)

def _get_theano_functions(
    params, diffs, particles, algorithm, us, us_param, us_init_particles, 
    t, ixs, v0s, xs_tr, recon_err_tr, recon_err_vl, info):
    if algorithm == 'cd':
        update_diffs  = function(
            inputs=[t, ixs], outputs=[], updates=us, 
            givens={v0s: xs_tr[ixs]}, name='update_diffs')
    elif algorithm == 'pcd':
        update_diffs = function(
            inputs=[t, ixs], outputs=[], updates=us, 
            givens={v0s: xs_tr[ixs]}, name='update_diffs')
    else:
        raise ValueError('Invalid value of algorithm: ' + str(algorithm))

    update_params    = function(
        inputs=[], outputs=[], updates=us_param, name='update_params')
    init_particles   = function(
        inputs=[ixs], outputs=[], updates=us_init_particles, 
        givens={v0s: xs_tr[ixs]}, name='init_particles')
    get_info         = function(
        inputs=[], outputs=[recon_err_tr, recon_err_vl] + info, 
        name='get_info')

    params.update(diffs)

    return {'update_diffs'  : update_diffs, 
            'update_params' : update_params, 
            'init_particles': init_particles, 
            'get_info'      : get_info}, params, particles



"""
Utility functions: define Theano constants, symbols and shared variables
"""

def _get_consts(xs_, params_tr):
    alpha  = scipy.float32(params_tr.alpha)
    eps    = scipy.float32(params_tr.eps)
    lmd_w  = scipy.float32(params_tr.lmd_w)
    d_vis  = xs_.shape[1]
    d_hid  = params_tr.n_components

    return alpha, eps, lmd_w, d_vis, d_hid

def _get_shared_vars(xs_tr_np, xs_vl_np):
    cp    = copy.deepcopy
    xs_tr = shared(cp(xs_tr_np).astype(cnf.floatX), name='xs_tr')

    if xs_vl_np is None:
        xs_vl = None
    else:
        xs_vl = shared(cp(xs_vl_np).astype(cnf.floatX), name='xs_vl')

    return xs_tr, xs_vl

def _get_shared_params(params_np):
    cp  = copy.deepcopy

    w = shared(cp(params_np['w']).astype(cnf.floatX), name='w')
    z = shared(cp(params_np['z']).astype(cnf.floatX), name='z')
    b = shared(cp(params_np['b']).astype(cnf.floatX), name='b')
    c = shared(cp(params_np['c']).astype(cnf.floatX), name='c')

    return {'w': w, 'z': z, 'b': b, 'c': c}

def _get_shared_diffs(params_np):
    cp  = copy.deepcopy

    dw = shared(cp(params_np['dw']).astype(cnf.floatX), name='w')
    dz = shared(cp(params_np['dz']).astype(cnf.floatX), name='z')
    db = shared(cp(params_np['db']).astype(cnf.floatX), name='b')
    dc = shared(cp(params_np['dc']).astype(cnf.floatX), name='c')

    return {'dw': dw, 'dz': dz, 'db': db, 'dc': dc}

def _get_shared_particles(params_tr):
    n_particles  = params_tr.n_particles
    n_components = params_tr.n_components

    _hs = shared(
        np.zeros((n_particles, n_components)).astype(cnf.floatX), name='_hs')

    return {'hs': _hs}

def _get_symbols():
    v0s = T.matrix('v0s')
    ixs = T.lvector('ixs')
    t   = T.scalar('t')

    return v0s, ixs, t



"""
Utility functions: tuples for Theano variable updates
"""

def _get_updates_diff(t, cost, diffs, params, params_tr, consts):
    # Model parameters 
    w, z, b, c = _get_params(params)

    # Training parameters
    alpha, eps0, decay = params_tr.alpha, params_tr.eps, params_tr.decay
    eps = scipy.float32(eps0) * (scipy.float32(decay)**t)

    # Parameter differences
    dw, dz, db, dc = _get_diffs(diffs)

    # Gradients
    gw = T.grad(cost=cost, wrt=w, consider_constant=consts)
    gz = T.grad(cost=cost, wrt=z, consider_constant=consts)
    gb = T.grad(cost=cost, wrt=b, consider_constant=consts)
    gc = T.grad(cost=cost, wrt=c, consider_constant=consts)

    # # Parameter updates with momentum
    update_dw = (dw, alpha * dw - eps * gw)
    update_dz = (dz, alpha * dz - eps * gz)
    update_db = (db, alpha * db - eps * gb)
    update_dc = (dc, alpha * dc - eps * gc)

    return [update_dw, update_dz, update_db, update_dc]

def _get_updates_param(diffs, params, params_tr):
    # Model parameters 
    w, z, b, c = _get_params(params)

    # Parameter differences
    dw, dz, db, dc = _get_diffs(diffs)

    # Tuples for update
    update_w = (w, (w + dw))
    update_z = (z, (z + dz))
    update_b = (b, (b + db))
    update_c = (c, (c + dc))

    updates = []

    if params_tr.is_update['w'] is True:
        updates += [update_w]
    if params_tr.is_update['z'] is True:
        updates += [update_z]
    if params_tr.is_update['b'] is True:
        updates += [update_b]
    if params_tr.is_update['c'] is True:
        updates += [update_c]

    return updates

def _get_updates_particles(particles, negs):
    _hs = particles['hs']

    return [(_hs, negs['hs'])]

def _get_updates_init_particles(rng, particles, v0s, params, params_tr):
    # Rate coding parameters
    n_rate = params_tr.n_rate

    # Model parameters 
    w, z, b, c = _get_params(params)

    # Particles
    _hs = particles['hs']

    # Samples of spike variables
    hs = _sample_hs_given_vs(rng, v0s, w, z, c, n_rate)

    return [(_hs, hs)]

def _concat_updates(updates_list, updates_scan):
    if updates_scan is None:
        return updates_list
    else:
        for i in updates_list:
            updates_scan[i[0]] = i[1]

        return updates_scan



"""
Utility functions: cost and enrgy functions
"""

def _get_cost(poss, negs, params, params_tr):
    vs_pos, hs_pos = poss['vs'], poss['hs']
    vs_neg, hs_neg = negs['vs'], negs['hs']
    w, z, b, c     = _get_params(params)

    lmd_w  = params_tr.lmd_w
    lmd_h  = params_tr.lmd_h
    n_rate = params_tr.n_rate

    e_pos        = _energy(vs_pos, hs_pos, w, z, b, c)
    e_neg        = _energy(vs_neg, hs_neg, w, z, b, c)
    pw           = _penalty_weight(w, lmd_w)
    ph, const_ph = _penalty_hidden_act(vs_pos, w, z, c, lmd_h, n_rate)
    consts       = [vs_pos, hs_pos, vs_neg, hs_neg] + const_ph

    return e_pos - e_neg + pw + ph, consts

def _penalty_weight(w, lmd_w):
    if lmd_w == 0.0:
        return scipy.float32(0.0)
    else:
        return lmd_w * T.sum(T.abs_(w))

def _penalty_hidden_act(vs, w, z, c, lmd_h, n_rate):
    if lmd_h == 0.0:
        return scipy.float32(0.0), []
    else:
        s  = T.exp(z)
        s2 = 1.0 / s
        vw = T.dot(s2 * vs, w)

        return lmd_h * n_rate * T.sum(T.abs_(T.nnet.sigmoid(vw + c))), [vw]

def _energy(vs, hs, w, z, b, c):
    bsize = vs.shape[0]
    s     = T.exp(z)
    s2    = 1.0 / s
    e1    = T.mean(T.sum(0.5 * s2 * (vs - b)**2, axis=1))
    e2    = T.mean(T.sum(c * hs, axis=1))
    e3    = T.sum(T.dot((s2 * vs).T, hs) * w) / bsize

    return e1 - e2 - e3



"""
Utility functions: monitoring learning progress
"""

def _get_recon_err(vs, params, n_rate):
    if vs is None:
        err = shared(0.0)
    else:
        # Reconstruction
        w, z, b, c = _get_params(params)
        hs         = _mean_hs_given_vs(vs, w, z, c) * n_rate
        vs_rec, _  = _meanstd_vs_given_hs(hs, w, z, b)

        # Error
        scale = T.mean(T.sqrt(T.sum(vs**2, axis=1)))
        errs  = T.sqrt(T.sum((vs - vs_rec)**2, axis=1)) / scale
        err   = T.mean(errs)

    return err

def _get_info(params):
    w, z, b, c = _get_params(params)

    mean_w = T.mean(w)
    mean_z = T.mean(z)
    mean_b = T.mean(b)
    mean_c = T.mean(c)
    max_w  = T.max(w)
    max_z  = T.max(z)
    max_b  = T.max(b)
    max_c  = T.max(c)

    return [mean_w, mean_z, mean_b, mean_c, 
            max_w, max_z, max_b, max_c]



"""
Utility functions: sampling from conditional distribution
"""

def _get_poss(rng, v0s, params, n_rate):
    w, z, b, c = _get_params(params)
    vs         = v0s
    hs         = _sample_hs_given_vs(rng, vs, w, z, c, n_rate)
    
    return {'vs': vs, 'hs': hs}

def _get_negs(rng, particles, params, n_rate):
    w, z, b, c = _get_params(params)
    _hs        = particles['hs']

    def _gibbs_hvh(hs0):
        vs1 = _sample_vs_given_hs(rng, hs0, w, z, b)
        hs1 = _sample_hs_given_vs(rng, vs1, w, z, c, n_rate)

        return [vs1, hs1]

    vs, hs = _gibbs_hvh(_hs)

    return {'vs': vs, 'hs': hs}

def _sample_hs_given_vs(rng, vs, w, z, c, n_rate):
    ps = _mean_hs_given_vs(vs, w, z, c)
    hs = rng.binomial(size=ps.shape, n=n_rate, p=ps, dtype=cnf.floatX)
    return hs

def _sample_vs_given_hs(rng, hs, w, z, b):
    ms, ss = _meanstd_vs_given_hs(hs, w, z, b)
    vs     = rng.normal(size=ms.shape, avg=ms, std=ss, dtype=cnf.floatX)
    return vs

def _sample_hs_given_vs_np(vs, w, z, c, n_rate):
    ps = _mean_hs_given_vs_np(vs, w, z, c)
    hs = np.random.binomial(size=ps.shape, n=n_rate, p=ps)
    return hs

def _sample_vs_given_hs_np(rng, hs, w, z, b):
    ms, ss = _meanstd_vs_given_hs_np(hs, w, z, b)
    vs     = np.random.normal(size=ms.shape, loc=ms, scale=ss)
    return vs



"""
Utility functions: statistics
"""

def _mean_hs_given_vs(vs, w, z, c):
    s  = T.exp(z)
    s2 = 1.0 / s
    return T.nnet.sigmoid(T.dot(s2 * vs, w) + c)

def _meanstd_vs_given_hs(hs, w, z, b):
    s = T.exp(z)
    m = T.dot(hs, w.T) + b
    return m, T.sqrt(s)

def _mean_hs_given_vs_np(vs, w, z, c):
    s  = np.exp(z)
    s2 = 1.0 / s
    return _sigmoid_np(np.dot(s2 * vs, w) + c)

def _meanstd_vs_given_hs_np(hs, w, z, b):
    s = np.exp(z)
    m = np.dot(hs, w.T) + b
    return m, np.sqrt(s)

def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))



"""
Utility functions: misc
"""

def _get_minibatches(batchSize, nSamples):
    bsize = np.clip(batchSize, 0, nSamples)
    nbs   = float(nSamples) / bsize
    imax  = nSamples - bsize
    ix0s  = np.arange(0, imax + 1, (imax + 1) / nbs).astype(int)
    ixss  = [np.arange(ix0, ix0 + bsize) for ix0 in ix0s]

    return ixss, bsize


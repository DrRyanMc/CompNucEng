import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def homog_slab_k(N,Sig_t,Sig_s,Sig_f,nu, thickness, inactive_cycles = 5, active_cycles = 20):
    Sig_a = Sig_t - Sig_s
    #initial fission sites are random
    fission_sites = np.random.uniform(0,thickness,N)
    positions = fission_sites.copy()
    weights = nu*np.ones(N)
    mus = np.random.uniform(-1,1,N)
    old_gen = np.sum(weights)
    k = np.zeros(inactive_cycles+active_cycles)
    for cycle in range(inactive_cycles+active_cycles):
        fission_sites = np.empty(1)
        fission_site_weights = np.empty(1)
        assert(weights.size == positions.size)
        for neut in range(weights.size):
            #grab neutron from stack
            position = positions[neut]
            weight = weights[neut]
            mu = mus[neut]
            alive = 1
            while (alive):
                #compute distance to collision
                l = -np.log(1-np.random.random(1))/Sig_t
                #move neutron
                position += l*mu
                #are we still in the slab
                if (position > thickness) or (position < 0):
                    alive = 0
                else:
                    #decide if collision is abs or scat
                    coll_prob = np.random.rand(1)
                    if (coll_prob < Sig_s/Sig_t):
                        #scatter
                        mu = np.random.uniform(-1,1,1)
                    else:
                        fiss_prob = np.random.rand(1)
                        alive = 0
                        if (fiss_prob <= Sig_f/Sig_a):
                            #fission
                            fission_sites = np.vstack((fission_sites,position))
                            fission_site_weights = np.vstack((fission_site_weights,weight))
        fission_sites =  np.delete(fission_sites,0,axis=0) #delete the initial site
        fission_site_weights =  np.delete(fission_site_weights,0,axis=0) #delete the initial site
        #sample neutrons for next generation from fission sites
        num_per_site = int(np.ceil(N/fission_sites.size))
        positions = np.zeros(1)
        weights = np.zeros(1)
        mus = np.random.uniform(-1,1,num_per_site*fission_sites.size)
        for site in np.hstack((fission_sites, fission_site_weights)):
            positions = np.vstack((positions,site[0]*np.ones((num_per_site,1))))
            weights = np.vstack((weights,site[1] * nu/num_per_site*np.ones((num_per_site,1))))
        positions =  np.delete(positions,0,axis=0) #delete the initial site
        weights =  np.delete(weights,0,axis=0) #delete the initial site
        new_gen = np.sum(weights)
        k[cycle] = new_gen/old_gen
        old_gen = new_gen
    return k

def fission_matrix(N,Sig_t,Sig_s,Sig_f,nu, thickness,Nx):
    Sig_a = Sig_t - Sig_s
    H = np.zeros((Nx,Nx))
    dx = thickness/Nx
    lowX = np.linspace(0,thickness-dx,Nx)
    highX = np.linspace(dx,thickness,Nx)
    midX = np.linspace(dx*0.5,thickness-dx*0.5,Nx)
    for col in range(Nx):
        #create source neutrons
        positions = np.random.uniform(lowX[col],highX[col],N)
        mus = np.random.uniform(-1,1,N)
        weights = np.ones(N)*(1.0/N)
        #track neutrons
        for neut in range(positions.size):
            #grab neutron from stack
            position = positions[neut]
            mu = mus[neut]
            weight = weights[neut]
            alive = 1
            while (alive):
                #compute distance to collision
                l = -np.log(1-np.random.random(1))/Sig_t
                #move neutron
                position += l*mu
                #are we still in the slab
                if (position > thickness) or (position < 0):
                    alive = 0
                else:
                    #decide if collision is abs or scat
                    coll_prob = np.random.rand(1)
                    if (coll_prob < Sig_s/Sig_t):
                        #scatter
                        mu = np.random.uniform(-1,1,1)
                    else:
                        fiss_prob = np.random.rand(1)
                        alive = 0
                        if (fiss_prob <= Sig_f/Sig_a):
                            #find which bin we are in
                            row = np.argmin(np.abs(position - midX))
                            H[row,col] += weight*nu
    return H, midX
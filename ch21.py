import numpy as np
import matplotlib.pyplot as plt

def hide_spines(intx=False,inty=False):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    if (plt.gca().get_legend()):
        plt.setp(plt.gca().get_legend().get_texts(), fontproperties=font) 
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
           # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
            for label in ax.get_xticklabels() :
                label.set_fontproperties(font)
            for label in ax.get_yticklabels() :
                label.set_fontproperties(font)
            #ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
            ax.set_xlabel(ax.get_xlabel(), fontproperties = font)
            ax.set_ylabel(ax.get_ylabel(), fontproperties = font)
            ax.set_title(ax.get_title(), fontproperties = font)
            if (inty):
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
            if (intx):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
def show(nm,a=0,b=0):
    hide_spines(a,b)
    #ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    #plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
    #ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
    plt.savefig(nm);
    plt.show()
def slab_transmission(Sig_s,Sig_a,thickness,N,isotropic=False, implicit_capture = True):
    """Compute the fraction of neutrons that leak through a slab
    Inputs:
    Sig_s:     The scattering macroscopic x-section
    Sig_a:     The absorption macroscopic x-section
    thickness: Width of the slab
    N:         Number of neutrons to simulate
    isotropic: Are the neutrons isotropic or a beam
    
    Returns:
    transmission:  The fraction of neutrons that made it through
    """
    Sig_t = Sig_a + Sig_s
    transmission = 0.0
    N = int(N)
    for i in range(N):
        if (isotropic):
            mu = np.random.random(1)
        else:
            mu = 1.0
        x = 0
        alive = 1
        weight = 1.0/N
        while (alive):
            if (implicit_capture):
                #get distance to collision
                if (Sig_s > 0):
                    l = -np.log(1-np.random.random(1))/Sig_s 
                else:
                    l = 10.0*thickness/mu #something that will make it through
            else:
                #get distance to collision
                l = -np.log(1-np.random.random(1))/Sig_t
            #make sure that l is not too large. If it is, move it to the edge.
            if (mu > 0):
                l = np.min([l,(thickness-x)/mu]) 
            else:
                l = np.min([l,-x/mu])
            #move particle
            x += l*mu
            if (implicit_capture):
                if not(l>=0):
                    print(l,x,mu)
                assert(l>=0)
                weight *= np.exp(-l*Sig_a)
            #still in the slab? 
            #It should be either at the edge of the slab on the right, or have a negative x value
            if (np.abs(x-thickness) < 1.0e-14):
                transmission += weight
                alive = 0
            elif (x<= 1.0e-14):
                alive = 0
            else:
                if (implicit_capture):
                    mu = np.random.uniform(-1,1,1)
                else:
                    #scatter or absorb
                    if (np.random.random(1) < Sig_s/Sig_t): 
                        #scatter, pick new mu
                        mu = np.random.uniform(-1,1,1)
                    else: #absorbed
                        alive = 0
    return transmission

def slab_source(Nx,Sig_s,Sig_a,thickness,N,Q,isotropic=False, implicit_capture = True):
    """Compute the fraction of neutrons that leak through a slab
    Inputs:
    Nx:        The number of grid points
    Sig_s:     The scattering macroscopic x-section
    Sig_a:     The absorption macroscopic x-section
    thickness: Width of the slab
    N:         Number of neutrons to simulate
    isotropic: Are the neutrons isotropic or a beam
    
    Returns:
    transmission:  The fraction of neutrons that made it through
    scalar_flux:   The scalar flux in each of the Nx cells
    X:             The value of the cell centers in the mesh
    """
    dx = thickness/Nx
    X = np.linspace(dx*0.5, thickness - 0.5*dx,Nx)
    scalar_flux = np.zeros(Nx)
    Sig_t = Sig_a + Sig_s
    leak_left = 0.0
    leak_right = 0
    N = int(N)
    for i in range(N):
        if (isotropic):
            mu = np.random.uniform(-1,1,1)
        else:
            mu = 1.0
        x = np.random.random(1)*thickness
        alive = 1
        weight = Q*thickness/N
        while (alive):
            if (implicit_capture):
                #get distance to collision
                if (Sig_s > 0):
                    l = -np.log(1-np.random.random(1))/Sig_s 
                else:
                    l = 10.0*thickness/mu #something that will make it through
            else:
                #get distance to collision
                l = -np.log(1-np.random.random(1))/Sig_t
            #make sure that l is not too large
            if (mu > 0):
                l = np.min([l,(3-x)/mu]) 
            else:
                l = np.min([l,-x/mu])
            #move particle
            x += l*mu
            if (implicit_capture):
                if not(l>=0):
                    print(l,x,mu)
                assert(l>=0)
                weight_old = weight
                weight *= np.exp(-l*Sig_a)
            #still in the slab?
            if (np.abs(x-thickness) < 1.0e-14):
                leak_right += weight
                alive = 0
            elif (x<= 1.0e-14):
                alive = 0
                leak_left += weight
            else:
                #compute cell particle collision is in
                cell= np.argmin(np.abs(X-x))
                if (implicit_capture):
                    mu = np.random.uniform(-1,1,1)
                    scalar_flux[cell] += weight/Sig_s/dx
                else:
                    #scatter or absorb
                    scalar_flux[cell] += weight/Sig_t/dx
                    if (np.random.random(1) < Sig_s/Sig_t): 
                        #scatter, pick new mu
                        mu = np.random.uniform(-1,1,1)
                    else: #absorbed
                        alive = 0
    return leak_left,leak_right, scalar_flux, X
        
def slab_source2(Nx,Sig_s,Sig_a,thickness,N,Q,isotropic=False, implicit_capture = True):
    """Compute the fraction of neutrons that leak through a slab
    Inputs:
    Nx:        The number of grid points
    Sig_s:     The scattering macroscopic x-section
    Sig_a:     The absorption macroscopic x-section
    thickness: Width of the slab
    N:         Number of neutrons to simulate
    isotropic: Are the neutrons isotropic or a beam
    
    Returns:
    transmission:  The fraction of neutrons that made it through
    scalar_flux:   The scalar flux in each of the Nx cells
    scalar_flux_tl:   The scalar flux in each of the Nx cells from track length estimator
    X:             The value of the cell centers in the mesh
    """
    dx = thickness/Nx
    X = np.linspace(dx*0.5, thickness - 0.5*dx,Nx)
    scalar_flux = np.zeros(Nx)
    scalar_flux_tl = np.zeros(Nx)
    Sig_t = Sig_a + Sig_s
    leak_left = 0.0
    leak_right = 0
    N = int(N)
    for i in range(N):
        if (isotropic):
            mu = np.random.uniform(-1,1,1)
        else:
            mu = 1.0
        x = np.random.random(1)*thickness
        alive = 1
        weight = Q*thickness/N
        #which cell am I in
        cell = np.argmin(np.abs(X-x))
        while (alive):
            if (implicit_capture):
                #get distance to collision
                if (Sig_s > 0):
                    l = -np.log(1-np.random.random(1))/Sig_s 
                else:
                    l = 10.0*thickness/np.abs(mu) #something that will make it through
            else:
                #get distance to collision
                l = -np.log(1-np.random.random(1))/Sig_t
            #compare distance to collision to distance to cell edge
            distance_to_edge = ((mu > 0.0)*( (cell+1)*dx - x) + 
                                (mu<0.0)*( x - cell*dx) + 1.0e-8)/np.abs(mu)
            if (distance_to_edge < l):
                l = distance_to_edge
                collide = 0
            else:
                collide = 1
            #move particle
            x += l*mu
            #score track length tally
            if (implicit_capture):
                scalar_flux_tl[cell] += weight*(1.0 - np.exp(-l*Sig_a))/(Sig_a + 1.0e-14)
            else:
                scalar_flux_tl[cell] += weight*l
            if (implicit_capture):
                if not(l>=0):
                    print(l,x,mu,cell,distance_to_edge)
                assert(l>=0)
                weight_old = weight
                weight *= np.exp(-l*Sig_a)
            #still in the slab?
            if (np.abs(x-thickness) < 1.0e-14) or (x > thickness):
                leak_right += weight
                alive = 0
            elif (x<= 1.0e-14):
                alive = 0
                leak_left += weight
            else:
                #compute cell particle collision is in
                cell= np.argmin(np.abs(X-x))
                if (implicit_capture):
                    if (collide):
                        mu = np.random.uniform(-1,1,1)
                    scalar_flux[cell] += weight/Sig_s/dx
                else:
                    #scatter or absorb
                    scalar_flux[cell] += weight/Sig_t/dx
                    if (collide) and (np.random.random(1) < Sig_s/Sig_t): 
                        #scatter, pick new mu
                        mu = np.random.uniform(-1,1,1)
                    elif (collide): #absorbed
                        alive = 0
            #print(x,mu,alive,l*mu,weight*l)
    return leak_left,leak_right, scalar_flux, scalar_flux_tl/dx, X

def create_particles(N,Q,X,Y,dx,dy):
    """Create N source particles in 2-D regular grid with source strengths in the 2-D array Q
    Inputs:
    N:         Number of neutrons to create
    Q:         2-D array of source strengths
    X,Y:       2-D array of zone centers
    dx,dy:     Width and height of zones
    
    Returns:
    census:    N by 7 array containing, weight, position (x,y), mu, gamma, and zone numbers
    """
    total = np.sum(Q)
    I,J = Q.shape
    census = np.empty((1,7))
    for i in range(I):
        for j in range(J):
            if Q[i,j] > 1.0e-14:
                num_emit = (np.ceil(Q[i,j]/total*N))
                #set weight
                wgt = Q[i,j]*dx*dy/(num_emit+1.0e-14)
                for emit in range(int(num_emit)):
                    #set position
                    pos = np.random.uniform(-0.5,0.5,2)
                    x_pos = dx * pos[0] + X[i,j]
                    y_pos = dy * pos[1] + Y[i,j]
                    mu = np.random.uniform(-1,1,1)
                    gamma = np.random.uniform(0,2*np.pi,1)
                    census = np.vstack((census,[wgt,x_pos,y_pos,mu[0],gamma[0],i,j]))
    return np.delete(census,0,axis=0)

def move_particles(census,X,Y,dx,dy,Sig_t,Sig_a,implicit_capture = True):
    """Create N source particles in 2-D regular grid with source strengths in the 2-D array Q
    Inputs:
    census:    List of particles created by the source function
    X,Y:       2-D arrays of cell centers
    dx,dy:     Widths of zones
    Sig_t:     2-D array of total macroscopic cross-sections
    Sig_a:     2-D array of absorption macroscopic cross-sections
    implicit_capture: whether or not to use implicit capture tracking
    
    Returns:
    scalar_flux_coll:    collision-estimated scalar flux array the same size as X and Y
    scalar_flux_tl:      track-length-estimated scalar flux array the same size as X and Y
    """
    Sig_s = Sig_t - Sig_a
    scalar_flux_coll = 0*X + 1e-14
    scalar_flux_tl =  0*X + 1e-14
    Lx, Ly = X.shape
    for neut in census:
        alive = 1
        while (alive):
            cell = np.array(neut[5:7], dtype=int)
            #compute distance to collision
            if (implicit_capture):
                #get distance to collision
                l = -np.log(1-np.random.random(1))/(Sig_s[cell[0], cell[1]] + 1.0e-14)
            else:
                #get distance to collision
                l = -np.log(1-np.random.random(1))/Sig_t[cell[0], cell[1]]
            #distance to x boundary
            center = [ X[cell[0], cell[1]], Y[cell[0], cell[1]]]
            pos = neut[1:3]
            mu = neut[3]
            gamma = neut[4]
            omega_x = np.sqrt(1.0-mu*mu)*np.cos(gamma)
            omega_y = np.sqrt(1.0-mu*mu)*np.sin(gamma)
            if (omega_x > 0):
                dist_x = (center[0] + dx*0.5 - pos[0])/omega_x + 1.0e-14
            else:
                dist_x = -(pos[0] - (center[0] - dx*0.5))/omega_x + 1.0e-14
            if (omega_y > 0):
                dist_y = (center[1] + dy*0.5 - pos[1])/omega_y + 1.0e-14
            else:
                dist_y = -(pos[1] - (center[1] - dy*0.5))/omega_y + 1.0e-14
            assert(dist_y>0)
            assert(dist_x>0)
            
            #find smallest distance
            if (l < dist_x) and (l < dist_y):
                neut[1] += l*omega_x
                neut[2] += l*omega_y
                #score in collision tally
                if (implicit_capture):
                    scalar_flux_coll[cell[0], cell[1]] += neut[0]/Sig_s[cell[0], cell[1]]
                else:
                    scalar_flux_coll[cell[0], cell[1]] += neut[0]/Sig_t[cell[0], cell[1]]
                if (implicit_capture and (Sig_a[cell[0], cell[1]] > 0)):
                    scalar_flux_tl[cell[0],cell[1]] += neut[0]*((1.0 - 
                                                                 np.exp(-l*Sig_a[cell[0], cell[1]]))
                                                                /(Sig_a[cell[0], cell[1]] + 1.0e-14))
                else:
                    scalar_flux_tl[cell[0],cell[1]] += neut[0]*l
                
                if (implicit_capture):
                    neut[3] = np.random.uniform(-1,1,1)
                    neut[4] = np.random.uniform(0,2*np.pi,1)
                    neut[0] *= np.exp(-l*Sig_a[cell[0], cell[1]] )
                else:
                    #scatter or absorb
                    if (np.random.random(1) < Sig_s[cell[0], cell[1]]/Sig_t[cell[0], cell[1]]): 
                        #scatter, pick new mu
                        neut[3] = np.random.uniform(-1,1,1)
                        neut[4] = np.random.uniform(0,2*np.pi,1)
                    else: #absorbed
                        #print("killed")
                        alive = 0
            elif (l >= dist_x) or (l >= dist_y):
                if (dist_y < dist_x):
                    pos[0] += (dist_y)*omega_x
                    neut[6] += np.sign(omega_y)
                    pos[1] += (dist_y + 1e-10)*omega_y
                    neut[1] = pos[0]
                    neut[2] = pos[1]
                    if (implicit_capture) and (Sig_a[cell[0], cell[1]] > 0):
                        scalar_flux_tl[cell[0],cell[1]] += neut[0]*((1.0 - 
                                                                     np.exp(-dist_y*Sig_a[cell[0], cell[1]]))
                                                                    /(Sig_a[cell[0], cell[1]] + 1.0e-14))
                        neut[0] *= np.exp(-dist_y*Sig_a[cell[0], cell[1]] )
                    else:
                        scalar_flux_tl[cell[0],cell[1]] += neut[0]*dist_y
                else:
                    pos[1] += (dist_x)*omega_y
                    neut[5] += np.sign(omega_x)
                    pos[0] += (dist_x + 1e-10)*omega_x
                    neut[1] = pos[0]
                    neut[2] = pos[1]
                    if (implicit_capture) and (Sig_a[cell[0], cell[1]] > 0):
                        scalar_flux_tl[cell[0],cell[1]] += neut[0]*((1.0 - 
                                                                     np.exp(-dist_x*Sig_a[cell[0], cell[1]]))
                                                                    /(Sig_a[cell[0], cell[1]] + 1.0e-14))
                        neut[0] *= np.exp(-dist_x*Sig_a[cell[0], cell[1]] )
                    else:
                        scalar_flux_tl[cell[0],cell[1]] += neut[0]*dist_x
            else:
                assert(0==1)
            
            #are we still in the problem?
            if ((pos[0] >= np.max(X)+ dx*0.5) or (pos[1] >= np.max(Y)+ dy*0.5) or 
                 ((pos[0]) < 1.0e-8) or ((pos[1]) < 1.0e-8)) :
                alive = 0
            if (neut[5] >= Lx) or (neut[5] < 0):
                alive = 0
            if (neut[6] >= Ly) or (neut[6] < 0):
                alive = 0
    return scalar_flux_coll/dx/dy, scalar_flux_tl/dx/dy

def lattice(Lengths,Dims):
    I = Dims[0]
    J = Dims[1]
    L = I*J
    Nx = Lengths[0]
    Ny = Lengths[1]
    hx,hy = np.array(Lengths)/np.array(Dims)
    
    Sigma_t = np.ones((I,J))*1
    Sigma_a = 0*Sigma_t
    Q = 0*Sigma_t
    for j in range(J):
        for i in range(I):
            x = (i+0.5)*hx
            y = (j+0.5)*hy

            if (x>=3.0) and (x<=4.0): 
                if (y>=3.0) and (y<=4.0):
                    Q[i,j] = 1.0
                if (y>=1.0) and (y<=2.0):
                    Sigma_t[i,j] = 10.0
                    Sigma_a[i,j] = 10.0
            if ( ((x>=1.0) and (x<=2.0)) or ((x>=5.0) and (x<=6.0))): 
                if ( ((y>=1.0) and (y<=2.0)) or
                    ((y>=3.0) and (y<=4.0)) or
                    ((y>=5.0) and (y<=6.0))):
                    Sigma_t[i,j] = 10.0
                    Sigma_a[i,j] = 10.0
            if ( ((x>=2.0) and (x<=3.0)) or ((x>=4.0) and (x<=5.0))): 
                if ( ((y>=2.0) and (y<=3.0)) or
                    ((y>=4.0) and (y<=5.0))):
                    Sigma_t[i,j] = 10.0
                    Sigma_a[i,j] = 10.0
    return Sigma_t, Sigma_a, Q

def strat_sample(N,S):    
    """Create N samples in S strata.
    N must be divisible by S
    Inputs:
    N:             number of samples
    S:             number of strata
    Returns:
    place_in_bin:  a numpy vector containing the samples
    """
    N = N + (N % S)
    assert(N%S == 0 )
    dS = 1.0/S
    bins = np.zeros(N,dtype=int)
    count = 0
    for i in range(N//S):
        bins[count:count+S] = np.random.permutation(S)
        count += S
    place_in_bin = np.random.uniform(-0.5*dS,0.5*dS,N) + (bins+0.5)*dS
    return place_in_bin

def strat_sample_2D(N,S): 
    """Create N samples in S*S strata.
    Inputs:
    N:             number of samples
    S:             number of strata in each dimension
    Returns:
    samples:       an N by 2 numpy vector containing the samples
    """
    #number of bins is S*S
    bins = S*S
    #make sure we have enough points
    if (N<bins):
        N = bins
    N += N % bins
    Num_per_bin = N//bins
    assert(N % bins == 0)
    samples = np.zeros((N,2))
    count = 0;
    for bin_x in range(S):
        for bin_y in range(S):
            for i in range(Num_per_bin):
                center = (bin_x/S + 0.5/S,bin_y/S + 0.5/S)
                samples[count,0:2] = center + np.random.uniform(low=-0.5,high=0.5,size=2)/S
                count += 1
            
    return samples

def russian_roulette(weight, wa):
    """Perform Russian roulette on Neutron
    Inputs:
    weight:        current weight
    wa:            average weight of surviving neutrons
    Returns:
    alive:         0,1 for whether neutron survived
    weight_final:  weight after roulette
    """
    pk = 1- weight/wa
    alive = np.random.random(1)<pk
    weight_final = wa*(alive) + 0
    return alive, weight_final

def split(neutron, wd):
    """Perform Russian roulette on Neutron
    Inputs:
    neutron:       census entry for current neutron
    wd:            desired weight for split neutrons
    Returns:
    new_neutrons:  census entries for new neutrons
    """
    w = neutron[0]
    num_split = int(np.round(w/wd))
    wsplit = w/num_split
    new_neutrons = np.zeros([num_split-1,7])
    neutron[0] = wsplit
    for i in range(num_split-1):
        new_neutrons[i,:] = neutron.copy()
    return new_neutrons
        
def expfiss(x):
    return 0.453*math.exp(-1.036*x)*math.sinh(math.sqrt(2.29*x))

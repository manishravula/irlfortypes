import math
from numpy.random import uniform
from random import randint
def resample_particles(particles, max_particles, resamp_particles):
    if(len(particles) <= max_particles):
        return particles
    normalize_MAP(particles)

    new_particles = []
    for particle in particles:
        if(not math.isnan(particle.NMAP) and not particle.NMAP == -float('inf')):
            new_particles.append(particle)
    
    particles = new_particles
    if(len(particles) <= max_particles):
        return particles

    particles.sort(key=lambda p: p.MAP)
    alpha = compute_alpha(particles, resamp_particles)

    left = []
    new_particles = []
    for particle in particles:
        if(particle.NMAP >= alpha):
            new_particles.append(particle)
        else:
            left.append(particle)

    particles = new_particles
    u = uniform(0.0, alpha)
    for particle in left:
        u = u - particle.NMAP
        if(u <= 0):
            particles.append(particle)
            u = u + alpha

    while(len(particles) < resamp_particles):
        ind = randint(0, len(left)-1)
        particles.append(left.pop(ind))

    return particles
    

def normalize_MAP(particles):
    max_MAP = particles[0].MAP
    for i in range(1, len(particles)):
        if(particles[i].MAP > max_MAP):
            max_MAP = particles[i].MAP
    for particle in particles:
        particle.update_NMAP(math.exp(particle.MAP - max_MAP))

    total = 0.0
    for particle in particles:
        if(not math.isnan(particle.NMAP) and not particle.NMAP == -float('inf')):
           total = total + particle.NMAP

    for particle in particles:
        particle.update_NMAP(particle.NMAP/total)

def compute_alpha(particles, M):
    N = len(particles)
    for i in range(N - M - 1, N):
        A = N - i - 1
        B = 0.0
        for j in range(0, i+1):
            B = B + particles[j].NMAP

        kappa = particles[i].NMAP
        if(kappa == 0.0):
            continue

        stat = (1.0/kappa) * B + A
        if(stat <= M):
            alpha = B/(M-A)
            return alpha
    print("error when computing alpha")
    return -1
        

import time
from concurrent.futures import ThreadPoolExecutor
from timeit import timeit
from turtledemo.chaos import plot

import numpy as np
from numba import njit,prange

@njit(nogil=True)
def friction_fn(v,vt):
    if v>vt:
        return - v*3
    else:
        return - v*3 *np.sign(v)
@njit(nogil=True)
def simulate_spring_mass_funky_damper(x0,Time=10,dt=0.0001,vt=0.1):

    times=np.arange(0,Time,dt)
    positions=np.zeros_like(times)
    v=0
    a=0
    x=x0
    positions[0]=x0/x0

    for ii in range(1,len(times)):

        t =times[ii]
        a =friction_fn(v,vt)-100*x
        v=v+a*dt
        x=x+v*dt
        positions[ii]=x/x0
    return times,positions

@njit(parallel=True)
def Run_Sims(End=1000):
    for ii in prange(int(End/0.1)):
        if ii ==0:
            continue
        simulate_spring_mass_funky_damper(ii*0.1)
Run_Sims(10)


ST=time.time()
simulate_spring_mass_funky_damper(1)
print(time.time()-ST)

ST=time.time()
simulate_spring_mass_funky_damper(1)
print(time.time()-ST)

ST=time.time()
with ThreadPoolExecutor(6) as executor:
    executor.map(simulate_spring_mass_funky_damper,np.arange(0.1,1000,0.1))
print(time.time()-ST)

ST=time.time()
Run_Sims(1000)
print(time.time()-ST)

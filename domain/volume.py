"""
Volume computation (sphere-like / star-shape domain)

"""
from typing import Optional
from typing import Tuple

from numpy import float64
from numpy.typing import NDArray
import numpy

from numba import njit
from numba import prange

def directions(
    dimension:int, 
    count:int,
    random:bool=True,
    seed:Optional[int]=None,
    ij:Optional[Tuple[int, int]]=None,
    omega_min:float = 0.0,
    omega_max:float = 2.0*numpy.pi,
    endpoint:bool = False
) -> NDArray[float64]:
    """
    Generates random or plane direction for DA computation

    Parameters
    ----------
    dimension: int
        phase space dimension
    count: int
        number of directions to generate
    random: bool, default=True
        flag to generate random directions
    seed: Optional[int]
        random seed (random generation)
    ij: Optional[Tuple[int, int]]
        target planes (in plane directions generation)
    omega_min: float, default==0.0
        min angle value (in plane directions generation)
    omega_max: float, default==2.0*numpy.pi
        max angle value (in plane directions generation)
    endpoint: bool, default=False
        flag to include interval end points (in plane directions generation)

    Returns
    -------
    NDArray[float64]
    
    """
    if random:
        generator = numpy.random.default_rng(seed)
        points = generator.normal(size=(count, dimension))
        return points/numpy.linalg.norm(points, axis=1, keepdims=True)
    i, j = ij
    angles = numpy.linspace(omega_min, omega_max, count, endpoint=endpoint, dtype=float64)
    directions = numpy.zeros((count, dimension), dtype=float64)
    directions[:, i] = numpy.cos(angles)
    directions[:, j] = numpy.sin(angles)
    return directions


@njit(parallel=True)
def rays(
    dimension:int,
    n:int,
    m:int
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    Generate radial directions for volume computation and compute Jacobian factors (sector phase space ordering)

    (q_1, ..., q_k, p_1, ..., p_k)

    x_i = (q_i, p_i) = rho r_i (cos(phi_i), sin(phi_i))

    sum_i r_i^2 = 1
    
    r_1 = cos(psi_1)
    r_2 = sin(psi_1) cos(psi_2)
    r_3 = sin(psi_1) sin(psi_2) cos(psi_3)
    ...
    r_k-1 = sin(psi_1) sin(psi_2) ...  cos(psi_k-1)
    r_k   = sin(psi_1) sin(psi_2) ...  sin(psi_k-1)

    phi_i in (0, 2*pi)
    psi_i in (0, pi/2)

    2D
    q_1 = rho cos(phi_1)
    p_1 = rho sin(phi_1)
    
    4D
    q_1 = rho cos(psi_1) cos(phi_1)
    q_2 = rho sin(psi_1) cos(phi_2)
    p_1 = rho cos(psi_1) sin(phi_1)
    p_2 = rho sin(psi_1) sin(phi_2)

    6D
    q_1 = rho cos(psi_1) cos(phi_1)
    q_2 = rho sin(psi_1) cos(psi_2) cos(phi_2)
    q_3 = rho sin(psi_1) sin(psi_2) cos(phi_3)
    p_1 = rho cos(psi_1) sin(phi_1)
    p_2 = rho sin(psi_1) cos(psi_2) sin(phi_2)
    p_3 = rho sin(psi_1) sin(psi_2) sin(phi_3)

    Parameters
    ----------
    dimension: int
        phase space dimension
    n: int
        number of mixing counts
    m: int
        number of in-plane counts

    Returns
    -------
    Tuple[NDArray[float64], NDArray[float64]]
        directions: (n**(k - 1)*m**k, 2*k)
        factors:    (n**(k - 1), )

    """
    k = dimension // 2
    n_psi = n**(k - 1)
    n_phi = m**k
    psi = (numpy.arange(n) + 0.5)*(0.5*numpy.pi/n)
    phi = (numpy.arange(m) + 0.5)*(2.0*numpy.pi/m)
    sa = numpy.sin(psi)
    ca = numpy.cos(psi)
    sp = numpy.sin(phi)
    cp = numpy.cos(phi)
    directions = numpy.empty((n_psi * n_phi, dimension), dtype=float64)
    factors = numpy.empty(n_psi, dtype=float64)
    for i_psi in prange(n_psi):
        rho = numpy.empty(k, dtype=float64)
        t = 1*i_psi
        product = 1.0
        weight = 1.0
        for j in range(k - 1):
            s = sa[t % n]
            c = ca[t % n]
            rho[j] = product * c
            product *= s
            weight *= (s**(2*(k - j - 1) - 1))*c
            t //= n
        rho[k - 1] = product
        factors[i_psi] = weight
        for i_phi in range(n_phi):
            t = 1*i_phi
            index = i_phi + i_psi*n_phi
            for i in range(k):
                directions[index, i] = rho[i]*cp[t % m]
                directions[index, k + i] = rho[i]*sp[t % m]
                t //= m
    return directions, factors


@njit
def volume(
    dimension:int,
    n:int,
    m:int,
    rb: NDArray[float64],
    factors: NDArray[float64]
) -> Tuple[float, float]:
    """
    Compute equivalent hyper-shpere radius and volume

    Parameters
    ----------
    dimension: int
        phase space dimension
    n: int
        number of mixing counts
    m: int
        number of in-plane counts
    rb: NDArray[float64]
        final radii for all directions, (n**(k - 1)*m**k, 2*k)
    factors: NDArray[float64]
        multiplicaton factors, (n**(k - 1), )

    Returns
    -------
    Tuple[float, float]
        radius and volume
        
    """
    k = dimension // 2
    n_psi = n**(k - 1)
    n_phi = m**k
    d_psi = 0.5*numpy.pi/n
    d_phi = 2.0*numpy.pi/m
    total = 0.0
    shift = 0
    for i_psi in range(n_psi):
        local = 0.0
        for i_phi in range(n_phi):
            local += rb[shift + i_phi]**dimension
        total += factors[i_psi]*local
        shift += n_phi
    integral = total*(d_psi**(k - 1))*(d_phi**k)
    scale = numpy.pi**k
    factor = 1.0
    for i in range(1, k):
        factor *= i
    area = 2.0*scale/factor
    unit = scale/(factor*k)
    radius = (integral/area)**(1.0/dimension)
    return radius, unit*(integral/area)

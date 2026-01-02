from typing import NamedTuple
import jax.numpy as jnp

class Camera(NamedTuple):
    W: int      
    H: int      
    fx: float   
    fy: float   
    cx: float   
    cy: float   
    W2C: jnp.ndarray  
    full_proj: jnp.ndarray  

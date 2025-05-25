from pydantic import BaseModel

class Quadcopter(BaseModel):
    mass: float = 1.0 #kg
    Ixx: float = 0.01 #kg·m²
    Iyy: float = 0.01 #kg·m²
    Izz: float = 0.02 #kg·m²
    g: float = 9.81
    T_max: int = 20
    T_min: int = 5
    torque_max: int = 0.3
    torque_min: int = -0.3
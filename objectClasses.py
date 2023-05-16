import numpy as np
from Custom.funcs import normalize


def normalize(vector: np.array) -> tuple:
    return vector / np.linalg.norm(vector)


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = normalize(np.array(direction) - self.origin)


class HitInfo:
    def __init__(self, diHit, dst, hitPoint, normal, material):
        self.material = material
        self.didHit = diHit
        self.dst = dst
        self.hitPoint = hitPoint
        self.normal = normal


class Material:
    def __init__(self, colour, emissionColour, emissionStrength):
        self.colour = colour
        self.emissionColour = emissionColour
        self.emissionStrength = emissionStrength


class Sphere:
    def __init__(self, position, radius, material: Material):
        self.position = np.array(position)
        self.radius = radius
        self.material = material

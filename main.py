import pygame as pg
from random import random
import numpy as np
import objectClasses as oC


def normalize(vector: np.array) -> tuple:
    return vector / np.linalg.norm(vector)


def main():
    pg.init()
    surface = pg.display.set_mode([500, 500])
    frame = 1
    while True:
        for x in range(surface.get_width()):
            for y in range(surface.get_height()):
                shader(surface, frame, x, y, raysPerPixel=1)
                [exit() for event in pg.event.get() if event.type == pg.QUIT]
                pg.display.flip()
        frame += 1


def drawPixel(surface: pg.Surface, pos, colour):
    pos = round(pos[0]), round(pos[1])
    r, g, b = colour[0] * 255, colour[1] * 255, colour[2] * 255
    surface.set_at(pos, (r, g, b))


def shader(surface, frame, x: int, y: int, raysPerPixel=1):
    oldPixelColour = surface.get_at((x, y))
    oldPixelColour = oldPixelColour[0] / 255, oldPixelColour[1] / 255, oldPixelColour[2] / 255
    origin = np.array(surface.get_size()) / 2
    adjustedX = x - origin[0]
    adjustedY = (origin[1] - y)
    camera_pos = (0, 0, -500)
    incomingLight = np.array((0, 0, 0))
    for ray in range(raysPerPixel):
        Ray = oC.Ray(camera_pos, (adjustedX, adjustedY, 0))
        incomingLight = incomingLight + traceRay(Ray, bounceLimit=10)
    pixelColour = incomingLight / raysPerPixel
    if frame > 1:
        weight = 1 / (frame + 1)
        adjustedPixelColour = np.array(oldPixelColour) * (1 - weight) + np.array(pixelColour) * weight
    else:
        adjustedPixelColour = pixelColour
    drawPixel(surface, (x, y), adjustedPixelColour)


def avgPixel(oldPixelColour, newPixelColour):
    return (np.array(oldPixelColour) + np.array(newPixelColour)) / 2


def traceRay(ray, bounceLimit):
    incomingLight = np.array((0, 0, 0))
    rayColour = np.array((1, 1, 1))

    for bounce in range(bounceLimit):
        HitInfo = objectDetection(ray)

        if HitInfo.didHit:
            ray.origin = HitInfo.hitPoint
            ray.direction = diffuseBounce(HitInfo.normal)
            emittedLight = np.array(HitInfo.material.emissionColour * HitInfo.material.emissionStrength)

            if not emittedLight.any():
                emittedLight = np.array([0, 0, 0])

            incomingLight = incomingLight + emittedLight * rayColour
            rayColour = rayColour * HitInfo.material.colour

        else:
            break
    return incomingLight


def objectDetection(Ray: oC.Ray):
    Spheres = [oC.Sphere((-350, 700, 350), 500, oC.Material((0, 0, 0), (1, 1, 1), 1)),
               oC.Sphere((0, -550, 0), 500, oC.Material((0, 1, 1), (0, 0, 0), 0)),
               oC.Sphere((0, 0, 0), 50, oC.Material((1, 0, 1), (0, 0, 0), 0)), ]
    closestHit = oC.HitInfo(False, 1000000000000000, None, None, None)
    for sphere in Spheres:
        hitInfo = rayHitSphere(Ray, sphere)
        if hitInfo.didHit and hitInfo.dst < closestHit.dst:
            closestHit = hitInfo
    return closestHit


def rayHitSphere(ray: oC.Ray, sphere: oC.Sphere):
    hit: oC.HitInfo = oC.HitInfo(False, None, None, None, None)
    offsetRay = ray.origin - sphere.position
    a = np.dot(ray.direction, ray.direction)
    b = 2 * np.dot(offsetRay, ray.direction)
    c = np.dot(offsetRay, offsetRay) - sphere.radius * sphere.radius
    discriminant = b * b - 4 * a * c
    if discriminant >= 0:
        dst = (-b - np.sqrt(discriminant)) / (2 * a)
        if dst >= 0:
            hit.didHit = True
            hit.dst = dst
            hit.hitPoint = ray.origin + ray.direction * dst
            hit.normal = normalize(hit.hitPoint - sphere.position)
            hit.material = sphere.material
    return hit


def randomNormalDistribution():
    theta = 2 * np.pi * random()
    rho = np.sqrt(-2 * np.log(random()))
    return rho * np.cos(theta)


def randomDirection():
    x, y, z = randomNormalDistribution(), randomNormalDistribution(), randomNormalDistribution()
    return normalize((x, y, z))


def diffuseBounce(normal):
    direction = randomDirection()
    normalizedBounce = normalize(normal + direction)
    return normalizedBounce


if __name__ == '__main__':
    main()

from math import sin, pi

def black_box_projectile(theta, v0=10, g=9.81):
    assert theta >= 0
    assert theta <= 90
    return (v0 ** 2) * sin(2 * pi * theta / 180) / g


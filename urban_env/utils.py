######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
#######################################################################

from __future__ import division, print_function

import importlib
from itertools import combinations
import numpy as np
from math import sqrt
from numpy.linalg import norm

EPSILON = 0.01


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x):
    if abs(x) > EPSILON:
        return x
    elif x > 0:
        return EPSILON
    else:
        return -EPSILON


def wrap_to_pi(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def triangle_area(p1, p2, p3):
    return abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))/2.0


def isRectangle(p):
    cx = (p[1][0] + p[2][0] + p[3][0] + p[0][0])/4;
    cy = (p[1][1] + p[2][1] + p[3][1] + p[0][1])/4;
    c = np.array([cx, cy])

    dd1 = norm(p[1]-c, 2)
    dd2 = norm(p[2]-c, 2)
    dd3 = norm(p[3]-c, 2)
    dd4 = norm(p[0]-c, 2)

    eps = 0.01
    return max(abs(dd1-dd2), abs(dd1-dd3), abs(dd1-dd4)) < eps



def point_within_rectangle(point, rect2):
    """
        Check if a point is inside an rectangle
    :param point: a point
    :param rect2: (center, length, width, angle)
    """
    (center, length, width, angle) = rect2
    rect2_corner_points = corner_points(rect2)
    if not isRectangle(rect2_corner_points):
        isRectangle(rect2_corner_points)
    corner_point_pairs = [
                            (rect2_corner_points[0], rect2_corner_points[1]),
                            (rect2_corner_points[1], rect2_corner_points[2]),
                            (rect2_corner_points[2], rect2_corner_points[3]),
                            (rect2_corner_points[3], rect2_corner_points[0]),
                         ]

    
    triangle_areas = []
    for corner_point_pair in corner_point_pairs:  # 2 for pairs, 3 for triplets, etc
        triangle_areas.append(triangle_area(point, corner_point_pair[0], corner_point_pair[1]))
    triangle_area_sum = sum(triangle_areas)
    rectangle_area = length*width
    if triangle_area_sum <= rectangle_area:
        return True
    else:
        return False
           

def point_in_rectangle(point, rect_min, rect_max):
    """
        Check if a point is inside a rectangle
    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point, rect2):
    """
        Check if a point is inside a rotated rectangle
    :param point: a point
    :param rect2: (center, length, width, angle)
    """
    point_is_inside_rotated_rectangle = False
    (center, length, width, angle) = rect2
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    point_is_inside_rotated_rectangle = point_in_rectangle(ru, [-length/2, -width/2], [length/2, width/2])
    return point_is_inside_rotated_rectangle


def point_in_ellipse(point, center, angle, length, width):
    """
        Check if a point is inside an ellipse
    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1, rect2):
    """
        Do two rotated rectangles intersect?
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1, rect2):
    """
        Check if rect1 has a corner inside rect2
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    rect1_corner_points = corner_points(rect1)
    any_corner_inside = False
    for p in rect1_corner_points:
        this_corner_inside = point_within_rectangle(p, rect2)
        if this_corner_inside:
            any_corner_inside = True
            break
    return any_corner_inside


def corner_points(rect1):
    (c1, l1, w1, a1) = rect1
    c1 = np.array(c1)
    l1v = np.array([l1/2, 0])
    w1v = np.array([0, w1/2])
    r1_points = np.array([
                           #[0, 0],
                           #- l1v, l1v, w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v + w1v, + l1v - w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    corner_pts = []
    for corner_pt in rotated_r1_points:
        corner_pts.append(corner_pt + c1)
    return corner_pts


def do_every(duration, timer):
    return duration < timer


def remap(v, x, y):
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])


def class_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object

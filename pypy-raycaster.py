#! /usr/bin python3
"""
Joseph Navarro 2021
Adapted from C++ raycasting tutorial at `https://lodev.org/cgtutor/raycasting.html`.

Dependencies:
    * PyPy 3.7+
    * Pillow 8.0+
    * PyGame 2.0+
    * NumPy

Additionally, although this is more or less a novel adaptation of the abovementioned tutorial, the following copyright
information has nevertheless been copied over from Lode Vandevenne's original C++ source code:

    Copyright (c) 2004-2019, Lode Vandevenne

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this list of conditions and the
          following disclaimer.
        * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
          the following disclaimer in the documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import pygame
import numpy as np
import math
from PIL import Image


SCREEN_WIDTH: int = 1024
SCREEN_HEIGHT: int = 768
SCALE: int = 1
WINDOW_WIDTH: int = SCREEN_WIDTH * SCALE
WINDOW_HEIGHT: int = SCREEN_HEIGHT * SCALE
TEX_WIDTH: int = 64
TEX_HEIGHT: int = 64
MAP_WIDTH: int = 24
MAP_HEIGHT: int = 24
FPS: int = 30
CLIPPING: int = 100  # Far clipping plane's distance from camera; high value effectively disables clipping plane

WORLD_MAP = [
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7],
    [4, 0, 4, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 0, 7, 7, 7, 7, 7],
    [4, 0, 5, 0, 0, 0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 7, 0, 0, 0, 7, 7, 7, 1],
    [4, 0, 6, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 8],
    [4, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 1],
    [4, 0, 8, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 8],
    [4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 7, 7, 7, 1],
    [4, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 1],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 6, 0, 6, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0, 0, 5, 0, 0, 2, 0, 0, 0, 2],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2],
    [4, 0, 6, 0, 6, 0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 2],
    [4, 0, 0, 5, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2],
    [4, 0, 6, 0, 6, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 5, 0, 0, 2, 0, 0, 0, 2],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 0, 6, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    ]

CEIL_MAP = [
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 7, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 7, 4, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 7, 5, 7, 7, 7, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 7, 7, 7, 7, 7, 7, 1],
    [4, 7, 6, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 8],
    [4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1],
    [4, 7, 8, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 8],
    [4, 7, 7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 1],
    [4, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 1],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4],
    [6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 6, 7, 6, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    [4, 7, 7, 7, 7, 7, 7, 7, 7, 4, 6, 7, 6, 2, 7, 7, 7, 7, 7, 2, 7, 7, 7, 2],
    [4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 2, 7, 7, 5, 7, 7, 2, 7, 7, 7, 2],
    [4, 7, 7, 7, 7, 7, 7, 7, 7, 4, 6, 7, 6, 2, 7, 7, 7, 7, 7, 2, 2, 7, 2, 2],
    [4, 7, 6, 7, 6, 7, 7, 7, 7, 4, 6, 7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 2],
    [4, 7, 7, 5, 7, 7, 7, 7, 7, 4, 6, 7, 6, 2, 7, 7, 7, 7, 7, 2, 2, 7, 2, 2],
    [4, 7, 6, 7, 6, 7, 7, 7, 7, 4, 6, 7, 6, 2, 7, 7, 5, 7, 7, 2, 7, 7, 7, 2],
    [4, 7, 7, 7, 7, 7, 7, 7, 7, 4, 6, 7, 6, 2, 7, 7, 7, 7, 7, 2, 7, 7, 7, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    ]

FLOOR_MAP = [
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 7],
    [4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7],
    [4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7],
    [4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 7],
    [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 4, 7, 7, 7, 7, 7],
    [4, 4, 5, 4, 4, 4, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 7, 4, 4, 4, 7, 7, 7, 1],
    [4, 4, 6, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 7, 4, 4, 4, 4, 4, 4, 8],
    [4, 4, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 1],
    [4, 4, 8, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 7, 4, 4, 4, 4, 4, 4, 8],
    [4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 7, 4, 4, 4, 7, 7, 7, 1],
    [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 1],
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 2, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 2, 4, 4, 5, 4, 4, 2, 4, 4, 4, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 2, 4, 4, 4, 4, 4, 2, 2, 4, 2, 2],
    [4, 4, 6, 4, 6, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 2],
    [4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 6, 4, 6, 2, 4, 4, 4, 4, 4, 2, 2, 4, 2, 2],
    [4, 4, 6, 4, 6, 4, 4, 4, 4, 4, 6, 4, 6, 2, 4, 4, 5, 4, 4, 2, 4, 4, 4, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 2, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    ]


def load_image(filename):
    """ Loads an image as a NumPy array.
    """
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    data = np.transpose(data, (1, 0, 2))

    texture = bytearray(TEX_HEIGHT * TEX_WIDTH * 3)
    for tex_y in range(TEX_HEIGHT):
        for tex_x in range(TEX_WIDTH):
            # This is actually the most important line: force the use of a Python int, vs a numpy uint8
            red, green, blue = [int(x) for x in data[tex_x, tex_y]]
            texture[(tex_x * TEX_HEIGHT + tex_y) * 3 + 0] = red
            texture[(tex_x * TEX_HEIGHT + tex_y) * 3 + 1] = green
            texture[(tex_x * TEX_HEIGHT + tex_y) * 3 + 2] = blue

    return texture


def wallcast(x, w, h, dir_x, plane_x, dir_y, plane_y, pos_x, pos_y):
    """ Calculations for wall raycasting.
    """
    # Calculate ray position and direction
    camera_x: float = 2 * x / w - 1  # x-coordinate in camera space
    ray_dir_x, ray_dir_y = dir_x + plane_x * camera_x, dir_y + plane_y * camera_x

    # Which box of the map we're in
    map_x, map_y = int(pos_x), int(pos_y)

    # Length of ray from one x- or y-side to next x- or y-side
    delta_dist_x: float = 0 if ray_dir_y == 0 else (1 if ray_dir_x == 0 else abs(1 / ray_dir_x))
    delta_dist_y: float = 0 if ray_dir_x == 0 else (1 if ray_dir_y == 0 else abs(1 / ray_dir_y))

    # What direction to step in, x- or y-direction
    hit, side = False, False

    # Calculate step and initial side_dist
    if ray_dir_x < 0:
        step_x, side_dist_x = -1, (pos_x - map_x) * delta_dist_x
    else:
        step_x, side_dist_x = 1, (map_x + 1.0 - pos_x) * delta_dist_x

    if ray_dir_y < 0:
        step_y, side_dist_y = -1, (pos_y - map_y) * delta_dist_y
    else:
        step_y, side_dist_y = 1, (map_y + 1.0 - pos_y) * delta_dist_y

    # Perform DDA
    while not hit:
        if side_dist_x < side_dist_y:
            side_dist_x += delta_dist_x
            map_x += step_x
            side = False
        else:
            side_dist_y += delta_dist_y
            map_y += step_y
            side = True

        if WORLD_MAP[map_x][map_y] > 0:
            hit = True

    # Calculate distance of perpendicular ray
    if not side:
        perp_wall_dist = (map_x - pos_x + (1 - step_x) * 0.5) / ray_dir_x
    else:
        perp_wall_dist = (map_y - pos_y + (1 - step_y) * 0.5) / ray_dir_y

    if perp_wall_dist > CLIPPING:
        return False, 0, 0, 0, 0, 0, 0

    # Calculate height of line to draw on screen
    line_height: int = int(h / perp_wall_dist)

    # Calculate lowest and highest pixel to fill current stripe
    draw_start: int = ((-line_height) >> 1) + (h >> 1)
    if draw_start < 0:
        draw_start = 0

    draw_end: int = (line_height >> 1) + (h >> 1)
    if draw_end >= h:
        draw_end = h

    # Value of wall_x
    if not side:
        wall_x = pos_y + perp_wall_dist * ray_dir_y
    else:
        wall_x = pos_x + perp_wall_dist * ray_dir_x
    wall_x -= int(wall_x)

    # X-coord on texture
    tex_x: int = int(wall_x * TEX_WIDTH)
    if not side and ray_dir_x > 0:
        tex_x = TEX_WIDTH - tex_x - 1
    if side and ray_dir_y < 0:
        tex_x = TEX_WIDTH - tex_x - 1

    # How much to increase texture coordinate per screen pixel
    step: float = TEX_HEIGHT / line_height

    # Texturing calculations
    return True,\
        (draw_start - h * 0.5 + line_height * 0.5) * step, \
        draw_start,\
        draw_end,\
        step, \
        WORLD_MAP[map_x][map_y] - 1, \
        tex_x


def floorcast_y(y, w, h, dir_x, plane_x, dir_y, plane_y, pos_x, pos_y) -> tuple:
    """ Outer loop calculations for floor and ceiling raycasting.
    """
    # Ray direction for leftmost ray (x = 0) and rightmost ray (x = w)
    ray_dir_x0, ray_dir_y0 = dir_x - plane_x, dir_y - plane_y
    ray_dir_x1, ray_dir_y1 = dir_x + plane_x, dir_y + plane_y

    # Current y-position compared to the center of the screen (horizon)
    hh: int = h >> 1
    p: int = y - hh

    # Horizontal distance from camera to floor for current row
    # (0.5 = z-position of midpoint between floor and ceiling)
    row_distance: float = 1.0 if not p else hh / p

    if not (row_distance < CLIPPING) or row_distance == 1:
        return False, 0, 0, 0, 0

    # Calculate real-world step vector for each x-step (parallel to camera plane)
    # and the real-world coordinates of leftmost column
    inv_w: float = 1 / w

    return True,\
        row_distance * (ray_dir_x1 - ray_dir_x0) * inv_w, \
        row_distance * (ray_dir_y1 - ray_dir_y0) * inv_w, \
        pos_x + row_distance * ray_dir_x0, \
        pos_y + row_distance * ray_dir_y0


def floorcast_x2(buffer, x, y, h, colormap, floor_x, floor_y, step_x, step_y) -> None:
    """ Inner loop calculations for floor and ceiling raycasting. In-place.
    """
    floor_x += step_x * x
    floor_y += step_y * x

    # Get tile coordinates for floor and ceiling texture
    map_x: int = int(floor_x)
    map_y: int = int(floor_y)

    # Get texture coordinate from fractional parts
    tx: int = int(TEX_WIDTH * (floor_x - map_x)) & (TEX_WIDTH - 1)
    ty: int = int(TEX_HEIGHT * (floor_y - map_y)) & (TEX_HEIGHT - 1)

    if 0 <= map_x < MAP_WIDTH and 0 <= map_y < MAP_HEIGHT:
        copy_color(buffer, x, y, h, colormap[FLOOR_MAP[map_x][map_y] - 1], tx, ty)  # Floor
        copy_color(buffer, x, h - y - 1, h, colormap[CEIL_MAP[map_x][map_y] - 1], tx, ty)  # Ceiling


def copy_color(buffer, x, y, h, source, tex_x, tex_y) -> None:
    base_screen = (x * h + y) * 3
    base_tex = (tex_x * TEX_HEIGHT + tex_y) * 3
    buffer[base_screen + 0] = source[base_tex + 0]
    buffer[base_screen + 1] = source[base_tex + 1]
    buffer[base_screen + 2] = source[base_tex + 2]


def update_display(surface: pygame.Surface, display: pygame.Surface, buffer, caption: str) -> None:
    """ Updates window contents.
    """
    pygame.surfarray.blit_array(surface, np.frombuffer(buffer, dtype="uint8").reshape(SCREEN_WIDTH, SCREEN_HEIGHT, 3))

    pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT), display)
    pygame.display.set_caption(caption)
    pygame.display.flip()


def update_events(dt: float, pos_x: float, pos_y: float, dir_x: float, dir_y: float, plane_x: float, plane_y: float):
    """ Updates player position in response to user input.
    """
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                pygame.quit()
                raise SystemExit
        elif e.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    move_speed: float = dt * 5.0
    rot_speed: float = dt * 3.0

    pressed = pygame.key.get_pressed()

    new_xpos_plus: int = int(pos_x + dir_x * move_speed)
    new_ypos_plus: int = int(pos_y + dir_y * move_speed)
    if pressed[pygame.K_UP]:
        if not WORLD_MAP[new_xpos_plus][int(pos_y)]:
            pos_x += dir_x * move_speed
        if not WORLD_MAP[int(pos_x)][new_ypos_plus]:
            pos_y += dir_y * move_speed

    new_xpos_minus: int = int(pos_x - dir_x * move_speed)
    new_ypos_minus: int = int(pos_y - dir_y * move_speed)
    if pressed[pygame.K_DOWN]:
        if not WORLD_MAP[new_xpos_minus][int(pos_y)]:
            pos_x -= dir_x * move_speed
        if not WORLD_MAP[int(pos_x)][new_ypos_minus]:
            pos_y -= dir_y * move_speed

    if pressed[pygame.K_RIGHT]:
        old_dir_x: float = dir_x
        dir_x = dir_x * math.cos(-rot_speed) - dir_y * math.sin(-rot_speed)
        dir_y = old_dir_x * math.sin(-rot_speed) + dir_y * math.cos(-rot_speed)
        old_plane_x: float = plane_x
        plane_x = plane_x * math.cos(-rot_speed) - plane_y * math.sin(-rot_speed)
        plane_y = old_plane_x * math.sin(-rot_speed) + plane_y * math.cos(-rot_speed)

    if pressed[pygame.K_LEFT]:
        old_dir_x: float = dir_x
        dir_x = dir_x * math.cos(rot_speed) - dir_y * math.sin(rot_speed)
        dir_y = old_dir_x * math.sin(rot_speed) + dir_y * math.cos(rot_speed)
        old_plane_x: float = plane_x
        plane_x = plane_x * math.cos(rot_speed) - plane_y * math.sin(rot_speed)
        plane_y = old_plane_x * math.sin(rot_speed) + plane_y * math.cos(rot_speed)

    return pos_x, pos_y, dir_x, dir_y, plane_x, plane_y


def main() -> None:
    """ Setup and main loop.
    """
    pos_x: float = 22.0
    pos_y: float = 11.5
    dir_x: float = -1.0
    dir_y: float = 0.0
    plane_x: float = 0.0
    plane_y: float = 0.66

    # Create display-related objects (PyGame)
    pygame.display.set_caption("Textured Raycaster")
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SWSURFACE, 32)
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    colormap = [
        load_image("pics/eagle.png"),
        load_image("pics/redbrick.png"),
        load_image("pics/purplestone.png"),
        load_image("pics/greystone.png"),
        load_image("pics/bluestone.png"),
        load_image("pics/mossy.png"),
        load_image("pics/wood.png"),
        load_image("pics/colorstone.png"),
    ]

    # buffer: list = np.empty((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype="uint8").tolist()
    w: int = SCREEN_WIDTH
    h: int = SCREEN_HEIGHT
    buffer = bytearray(b"0" * w * h * 3)

    while True:
        # Raycasting for floor/ceiling textures
        for y in range(h >> 1, h):
            do_continue, step_x, step_y, floor_x, floor_y = floorcast_y(y, w, h, dir_x, plane_x, dir_y, plane_y,
                  pos_x, pos_y)

            if do_continue:
                for x in range(w):
                    floorcast_x2(buffer, x, y, h, colormap, floor_x, floor_y, step_x, step_y)

        # Raycasting for wall textures
        for x in range(w):
            do_continue, tex_pos, y1, y2, step, tex_num, tex_x = wallcast(x, w, h, dir_x, plane_x, dir_y, plane_y,
                  pos_x, pos_y)

            if do_continue:
                texture = colormap[tex_num]
                for y in range(y1, y2):
                    copy_color(buffer, x, y, h, texture, tex_x, int(tex_pos + step * (y - y1)) & (TEX_HEIGHT - 1))

        # Update display
        caption: str = "Textured Raycaster | FPS = {0:.2f}".format(clock.get_fps())
        update_display(surface, display, buffer, caption)
        buffer[:] = b'\x00' * len(buffer)

        # Grab user input
        dt: float = clock.tick(FPS) * 0.001
        pos_x, pos_y, dir_x, dir_y, plane_x, plane_y = update_events(dt, pos_x, pos_y, dir_x, dir_y, plane_x, plane_y)


if __name__ == "__main__":
    import os
    import sys

    logfile = os.path.join(os.path.dirname(__file__), "replay.py")
    if "--record" in sys.argv:
        recording = open(logfile, "w")
        original = update_events
        events = {}
        def update_events(*args):
            try:
                result = original(*args)
                events[result] = None
            except SystemExit:
                recording.write("REPLAY_LOG = %r" % list(events.keys()))
                raise
            return result

    if "--replay" in sys.argv:
        from replay import REPLAY_LOG
        replay_idx = -1
        def update_events(*args):
            global replay_idx
            replay_idx += 1
            try:
                return REPLAY_LOG[replay_idx]
            except IndexError:
                raise SystemExit

    if "--bench" in sys.argv:
        from replay import REPLAY_LOG
        BENCH_ITERATIONS = 5
        print("Running %d iterations at %d frames each" % (BENCH_ITERATIONS, len(REPLAY_LOG)))
        replay_idx = -1
        iteration = 0
        start = pygame.time.get_ticks()
        def update_events(*args):
            global start, replay_idx, iteration
            replay_idx += 1
            try:
                return REPLAY_LOG[replay_idx]
            except IndexError:
                print("Iteration %d time: %dms" % (iteration, pygame.time.get_ticks() - start))
                start = pygame.time.get_ticks()
                iteration += 1
                replay_idx = 0
                if iteration > BENCH_ITERATIONS:
                    raise SystemExit
                else:
                    return REPLAY_LOG[replay_idx]

    main()

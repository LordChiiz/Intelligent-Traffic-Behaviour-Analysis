import numpy as np
import cv2

IMAGE_POINTS = np.array([
    [481, 47],   # TL, Points were gotten from running the point picker on the video file
    [697, 47],   # TR
    [1270, 546],   # BR
    [184, 546],    # BL
], dtype=np.float32)


WORLD_POINTS = np.array([
    [0, 0],       # TL, meters, estimate of how wide and long the road is from our video data. adjustable
    [17.5, 0],     # TR, if an average lane is 3.5m, there are 4 main lanes, and one bus lane in traffic vidoe file ~ 5(3.5m)
    [17.5, 100],    # BR,  using length of the white dashes on the road(~3m) and the gaps between the dashes(~5m)
    [0, 100],      # BL
], dtype=np.float32)

_H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)


def pixel_to_world(point): #magic :)
    px, py = point
    p = np.array([[[px, py]]], dtype=np.float32)
    world_p = cv2.perspectiveTransform(p, _H)
    return float(world_p[0][0][0]), float(world_p[0][0][1])


def ground_point(x1, y1, x2, y2): #this to get the exact points that the vehicle is touching the ground (bottom-center) for better accuracy
    return (x1 + x2) / 2, y2
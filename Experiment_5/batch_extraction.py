import cv2
import numpy as np
import math
import time
from multiprocessing import Pool, cpu_count
import os
import re
import json
from pathlib import Path
import csv

def worker(args):
    pts, start, end = args
    max_dist = 0
    p1, p2 = None, None

    for i in range(start, end):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dist = dx*dx + dy*dy

            if dist > max_dist:
                max_dist = dist
                p1 = tuple(pts[i])
                p2 = tuple(pts[j])

    return max_dist, p1, p2

def farthest_points_bruteforce(contour):
    pts = contour.reshape(-1, 2)
    n = len(pts)

    num_workers = cpu_count()  # usually 8 for your CPU
    chunk_size = n // num_workers

    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = n if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((pts, start, end))

    with Pool(num_workers) as pool:
        results = pool.map(worker, tasks)

    # Combine results
    max_dist, p1, p2 = 0, None, None
    for dist, a, b in results:
        if dist > max_dist:
            max_dist = dist
            p1, p2 = a, b

    return p1, p2

def get_global_perp_axis(contour, direction_vec):
    pts = contour.reshape(-1, 2).astype(np.float32)

    direction_vec = direction_vec / np.linalg.norm(direction_vec)

    # project all points
    projections = pts @ direction_vec

    idx_max = np.argmax(projections)
    idx_min = np.argmin(projections)

    return tuple(map(int, pts[idx_max])), tuple(map(int, pts[idx_min]))

def worker_short_axis(args):
    pts, vec_major, start, end, tol = args

    max_dist = 0
    best_p1, best_p2 = None, None

    for i in range(start, end):
        for j in range(i + 1, len(pts)):
            v = pts[j] - pts[i]
            dist = np.linalg.norm(v)
            if dist == 0:
                continue

            v_unit = v / dist

            if abs(np.dot(v_unit, vec_major)) < tol:
                if dist > max_dist:
                    max_dist = dist
                    best_p1 = tuple(map(int, pts[i]))
                    best_p2 = tuple(map(int, pts[j]))

    return max_dist, best_p1, best_p2

def brute_force_short_axis(contour, vec_major, tol=0.01):
    pts = contour.reshape(-1, 2).astype(np.float32)

    vec_major = vec_major / np.linalg.norm(vec_major)

    n = len(pts)
    num_workers = cpu_count()
    chunk_size = n // num_workers

    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = n if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((pts, vec_major, start, end, tol))

    with Pool(num_workers) as pool:
        results = pool.map(worker_short_axis, tasks)

    max_dist = 0
    best_p1, best_p2 = None, None

    for dist, p1, p2 in results:
        if dist > max_dist:
            max_dist = dist
            best_p1, best_p2 = p1, p2

    return best_p1, best_p2

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    # avoid division by zero (nearly parallel)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return (int(px), int(py))

def draw_label(img, p, text):
    x, y = int(p[0]), int(p[1])
    cv2.putText(img, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def vertical_intersections(contour, X):
    """
    contour: list or numpy array of shape (N,2)
    X: tuple (x, y)
    return: list of intersection points [(x, y1), (x, y2)]
    """
    x0 = X[0]
    intersections = []

    n = len(contour)

    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]  # wrap around

        x1, y1 = p1
        x2, y2 = p2

        # Check if segment crosses vertical line x = x0
        if (x1 - x0) * (x2 - x0) <= 0 and x1 != x2:
            # Compute intersection y using linear interpolation
            t = (x0 - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)

            intersections.append((x0, y))

    return intersections

def point_on_line(P1, P2, p):
    return (
        int(P1[0] + p * (P2[0] - P1[0])),
        int(P1[1] + p * (P2[1] - P1[1]))
    )

def horizontal_line_contour_intersections(y0, contour, eps=1e-9):
    """
    Find intersections between horizontal line y = y0 and contour

    contour: OpenCV contour (Nx1x2 or Nx2)
    returns: list of (x, y0)
    """

    pts = contour.reshape(-1, 2)
    n = len(pts)

    intersections = []

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]

        # Skip horizontal edges (avoid division by zero)
        if abs(y2 - y1) < eps:
            continue

        # Check if y0 crosses the segment
        if (y0 >= min(y1, y2)) and (y0 <= max(y1, y2)):
            # Compute intersection x
            x = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)
            intersections.append((x, y0))

    return intersections

def unique_points(points, tol=1e-5):
    unique = []
    for p in points:
        if not any(np.linalg.norm(np.array(p) - np.array(q)) < tol for q in unique):
            unique.append(p)
    return unique

def find_ApBp(outer_cnt, inner_cnt, AB_len, O, max_scan=500):
    target = 0.9 * AB_len
    y0 = int(O[1])

    best_pair = None
    best_error = float('inf')

    for dy in range(max_scan):
        y = y0 - dy   # scan upward (or both directions if needed)

        outer_pts = horizontal_line_contour_intersections(y, outer_cnt)
        outer_pts = unique_points(outer_pts)
        # print(outer_pts)
        inner_pts = horizontal_line_contour_intersections(y, inner_cnt)
        inner_pts = unique_points(inner_pts)
        # print(inner_pts)

        if len(outer_pts) < 2 or len(inner_pts) < 2:
            break

        outer_pts.sort(key=lambda p: p[0])
        inner_pts.sort(key=lambda p: p[0])
        
        pairs = [
            (outer_pts[0], inner_pts[0]),
            (outer_pts[-1], inner_pts[-1]),
        ]

        for p1, p2 in pairs:
            l = abs(p1[0] - p2[0])
            err = abs(l - target)

            if err < best_error:
                best_error = err
                best_pair = (p1, p2)

    return best_pair


def contour_smoothness_score(contour):

    ellipse = cv2.fitEllipse(contour)

    mask = np.zeros((2000, 2000), dtype=np.uint8)

    cv2.ellipse(mask, ellipse, 255, 2)

    ellipse_pts = np.column_stack(np.where(mask > 0))

    contour_pts = contour.reshape(-1, 2)

    total = 0

    for p in contour_pts:

        d = np.min(
            np.linalg.norm(
                ellipse_pts - np.array([p[1], p[0]]),
                axis=1
            )
        )

        total += d

    return total / len(contour_pts)

def feature_extraction(img_dir, img_name, output_dir):
    # Load image

    img = cv2.imread(img_dir) # Load in BGR to draw colored lines

    img = cv2.copyMakeBorder(
        img, top = 300, bottom = 300, left = 300, right = 300,
        borderType=cv2.BORDER_CONSTANT, value=(255,255,255))   # increase canvas size 300 on each edge

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #----------------------------------------------------------------------------
    # region Original Image
    cnt = max(contours, key=cv2.contourArea)
    
    # start = time.time()
    # Long axis
    long_p1, long_p2 = farthest_points_bruteforce(cnt)

    vec_major = np.array(long_p2) - np.array(long_p1)
    vec_major = vec_major / np.linalg.norm(vec_major)

    vec_minor = np.array([-vec_major[1], vec_major[0]])

    # short_p1, short_p2 = get_global_perp_axis(cnt, vec_minor)
    short_p1, short_p2 = brute_force_short_axis(cnt, vec_major, tol=0.01)

    O = line_intersection(long_p1, long_p2, short_p1, short_p2)

    v1 = np.array(long_p2) - np.array(long_p1)
    v2 = np.array(short_p2) - np.array(short_p1)

    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)

    dot = np.dot(v1_unit, v2_unit)
    print(img_name, " - Cos(angle): ", dot)

    # end = time.time()

    # print("Time:", end - start, "seconds")

    # -------------------------------------------------------------------------
    # region Image Rotation  
    dx = long_p2[0] - long_p1[0]
    dy = long_p2[1] - long_p1[1]

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    rotation_angle = -angle_deg + 180
    center = (int(O[0]), int(O[1]))

    (h, w) = img.shape[:2]

    M = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    # Refind the contour on the rotated image
    gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Rotate O and axis accordingly
    def rotate_point(p, M):
        x, y = p
        new = M @ np.array([x, y, 1])
        return (new[0], new[1])

    long_p1_r = rotate_point(long_p1, M)
    long_p2_r = rotate_point(long_p2, M)
    short_p1_r = rotate_point(short_p1, M)
    short_p2_r = rotate_point(short_p2, M)
    O_r = rotate_point(O, M)

    # Ensure long_p1_r is on the left and long_p2_r is on the right <=> long_p1_r[0] < long_p2_r[0]
    if long_p1_r[0] > long_p2_r[0]:
        print("Swap position long")
        # Swap position of long_p1_r and long_p2_r
        temp = long_p1_r
        long_p1_r = long_p2_r
        long_p2_r = temp

    # Ensure short_p1_r is at the top and short_p2_r is at the bottom <=> short_p1_r[1] < short_p2_r[1]
    if short_p1_r[1] > short_p2_r[1]:
        print("Swap position short")
        # Swap position of short_p1_r and short_p2_r
        temp = short_p1_r
        short_p1_r = short_p2_r
        short_p2_r = temp


    # Find the large point and small point of the egg
    # Ensure A is on the small point and on the left, B is the large point and on the right
    long_p1_r_NpArray = np.array(long_p1_r) # Np array version of long_p1_r to calculate easier
    long_p2_r_NpArray = np.array(long_p2_r)

    A_side_point = long_p1_r_NpArray + 0.1 * (long_p2_r_NpArray - long_p1_r_NpArray)  # A is near long_p1_r
    B_side_point = long_p1_r_NpArray + 0.9 * (long_p2_r_NpArray - long_p1_r_NpArray)  # B is near long_p2_r

    # Calculate the vertical line that go through A and B
    points_1 = vertical_intersections( contour.reshape(-1, 2), A_side_point )

    length_A = np.linalg.norm(    np.array(points_1[0])   -   np.array(points_1[1])    )
    print("Length A: ", length_A)

    points_2 = vertical_intersections( contour.reshape(-1, 2), B_side_point )

    length_B = np.linalg.norm(    np.array(points_2[0])   -   np.array(points_2[1])    )
    print("Length B: ", length_B)

    if length_A > length_B: # Means the large point is on the left side
        print("Prepare to rotate 2nd")
        # Rotate the image 180 degrees
        rotation_angle = 180
        center = (int(O_r[0]), int(O_r[1]))

        (h, w) = rotated_img.shape[:2]

        M = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
        rotated_img = cv2.warpAffine(rotated_img, M, (w, h), borderValue=(255, 255, 255))

        # Refind the contour on the rotated image
        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)

        # Subsequently rotate the point 180 degrees
        long_p1_r = rotate_point(long_p1_r, M) # long_p1 and long_p2 are the furthest line but with unknown direction
        long_p2_r = rotate_point(long_p2_r, M)
        short_p1_r = rotate_point(short_p1_r, M)
        short_p2_r = rotate_point(short_p2_r, M)
        O_r = rotate_point(O_r, M)

        # Similarly Ensure the points's position
        # Ensure long_p1_r is on the left and long_p2_r is on the right <=> long_p1_r[0] < long_p2_r[0]
        if long_p1_r[0] > long_p2_r[0]:
            # Swap position of long_p1_r and long_p2_r
            temp = long_p1_r
            long_p1_r = long_p2_r
            long_p2_r = temp

        # Ensure short_p1_r is at the top and short_p2_r is at the bottom <=> short_p1_r[1] < short_p2_r[1]
        if short_p1_r[1] > short_p2_r[1]:
            # Swap position of short_p1_r and short_p2_r
            temp = short_p1_r
            short_p1_r = short_p2_r
            short_p2_r = temp

    #------------------------------------------------------------------------------
    # ---------------------------------------------------------
    # region Lxt and Lxh

    # A, B is 2 points of the long axis, A is the small point  and on the left
    # C, D is 2 points of short axis, C is at the top
    O = np.array(O_r)
    A = np.array(long_p1_r)
    B = np.array(long_p2_r)
    C = np.array(short_p1_r)
    D = np.array(short_p2_r)

    Lx = np.linalg.norm(C - D)
    Ly = np.linalg.norm(A - B)

    Lxt = np.linalg.norm(O - A)  # narrow side
    Lxh = np.linalg.norm(O - B)

    # ---------------------------------------------------------
    # region Wh and Wt
    X1 = O + 0.5 * (B - O)

    # v_long = B - A
    # v_long = v_long / np.linalg.norm(v_long)

    # # perpendicular vector
    # v_perp = np.array([-v_long[1], v_long[0]])

    points = vertical_intersections(contour.reshape(-1, 2), X1)
    Wh50_p1, Wh50_p2 = np.array(points[0]), np.array(points[1])

    Wh50 = np.linalg.norm(Wh50_p1 - Wh50_p2)

    X2 = O + 0.85 * (B - O)
    points = vertical_intersections(contour.reshape(-1, 2), X2)
    Wh85_p1, Wh85_p2 = np.array(points[0]), np.array(points[1])

    Wh85 = np.linalg.norm(Wh85_p1 - Wh85_p2)

    X3 = O + 0.90 * (B - O)
    points = vertical_intersections(contour.reshape(-1, 2), X3)
    Wh90_p1, Wh90_p2 = np.array(points[0]), np.array(points[1])

    Wh90 = np.linalg.norm(Wh90_p1 - Wh90_p2)

    Y1 = O - 0.50 * (O - A)
    points = vertical_intersections(contour.reshape(-1, 2), Y1)
    Wt50_p1, Wt50_p2 = np.array(points[0]), np.array(points[1])

    Wt50 = np.linalg.norm(Wt50_p1 - Wt50_p2)

    Y2 = O - 0.85 * (O - A)
    points = vertical_intersections(contour.reshape(-1, 2), Y2)
    Wt85_p1, Wt85_p2 = np.array(points[0]), np.array(points[1])

    Wt85 = np.linalg.norm(Wt85_p1 - Wt85_p2)

    Y3 = O - 0.90 * (O - A)
    points = vertical_intersections(contour.reshape(-1, 2), Y3)
    Wt90_p1, Wt90_p2 = np.array(points[0]), np.array(points[1])

    Wt90 = np.linalg.norm(Wt90_p1 - Wt90_p2)

    # ---------------------------------------------------------
    # region Area

    Sx = cv2.contourArea(contour)

    # Sxt and Sxh

    h, w = gray.shape

    # 1. Create full mask of the contour
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    # 2. Create mask for left side of O
    x0 = int(O[0])
    left_mask = np.zeros_like(mask)
    left_mask[:, :x0] = 255

    # 3. Intersect (keep only left part of contour)
    left_region = cv2.bitwise_and(mask, left_mask)

    # 4. Compute area
    Sxt = cv2.countNonZero(left_region)

     # 2. Right side mask
    x0 = int(O[0])
    right_mask = np.zeros_like(mask)
    right_mask[:, x0:] = 255   # keep right side

    # 3. Intersection
    right_region = cv2.bitwise_and(mask, right_mask)

    # 4. Area
    Sxh = cv2.countNonZero(right_region)

    # ------------------------------------------------------------------
    # region Area part 2

    # Draw the ellipse
    center = (( C + D) / 2).astype(int)

    # axes
    a = np.linalg.norm(C - D) / 2  # vertical radius
    b = abs(B[0] - center[0])        # horizontal radius

    # # draw
    cv2.ellipse(rotated_img, tuple(center), (int(b), int(a)), 0, 0, 360, (0,255,0), 2)
    pts = cv2.ellipse2Poly(center, (int(b), int(a)), 0, 0, 360, delta=2)
    ellipse_cnt = pts.reshape(-1, 1, 2)
    ellipse_area = cv2.contourArea(ellipse_cnt)

    Sxd = ellipse_area

    Seh = Sxh - ellipse_area/2
    Set = Sxt - ellipse_area/2

    # E - The symmetrical point of B on the ellipse:
    E = 2*O - B
    # Cal AB in the slide
    AE = np.linalg.norm(A - E)
    AB_phay = 0.9 * AE
    

    J, K = find_ApBp(contour, ellipse_cnt, AE, O)
    
    J = np.array(J)
    K = np.array(K)

    No303 = (J[1] - O[1])*2

    # --------------------------------------------------------------------
    # region Draw Image - 1
    cv2.circle(rotated_img, tuple(O.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(A.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(B.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(C.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(D.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(X1.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(Y1.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(J.astype(int)), 5, (0, 255, 255), -1)
    cv2.circle(rotated_img, tuple(K.astype(int)), 5, (0, 255, 255), -1)
    draw_label(rotated_img, O, "O")
    draw_label(rotated_img, A, "A")
    draw_label(rotated_img, B, "B")
    draw_label(rotated_img, C, "C")
    draw_label(rotated_img, D, "D")
    draw_label(rotated_img, X1, "X")
    draw_label(rotated_img, Y1, "Y1")
    draw_label(rotated_img, J, "J")
    draw_label(rotated_img, K, "K")


    # Long Axis in Red
    # cv2.line(rotated_img, long_p1_r, long_p2_r, (0, 0, 255), 2)
    cv2.line(rotated_img, tuple(C.astype(int)), tuple(D.astype(int)), (0, 0, 255), 2)

    cv2.line(
        rotated_img,
        (int(O_r[0]), int(O_r[1])),
        (int(A[0]), int(A[1])),
        (255, 0, 0),
        2
    )

    cv2.line(
        rotated_img,
        (int(O_r[0]), int(O_r[1])),
        (int(B[0]), int(B[1])),
        (0, 255, 0),
        2
    )

    cv2.line(rotated_img, tuple(Wh50_p1.astype(int)), tuple(Wh50_p2.astype(int)), (0, 0, 255), 2)
    draw_label(rotated_img, Wh50_p1, "Wh50")

    cv2.line(rotated_img, tuple(Wh85_p1.astype(int)), tuple(Wh85_p2.astype(int)), (0, 0, 255), 2)
    draw_label(rotated_img, Wh85_p1, "Wh85")

    cv2.line(rotated_img, tuple(Wh90_p1.astype(int)), tuple(Wh90_p2.astype(int)), (0, 0, 255), 2)
    draw_label(rotated_img, Wh90_p1, "Wh90")

    cv2.line(rotated_img, tuple(Wt50_p1.astype(int)), tuple(Wt50_p2.astype(int)), (0, 0, 255), 2)
    draw_label(rotated_img, Wt50_p2, "Wt50")

    cv2.line(rotated_img, tuple(Wt85_p1.astype(int)), tuple(Wt85_p2.astype(int)), (0, 0, 255), 2)
    draw_label(rotated_img, Wt85_p2, "Wt85")

    cv2.line(rotated_img, tuple(Wt90_p1.astype(int)), tuple(Wt90_p2.astype(int)), (0, 0, 255), 2)
    draw_label(rotated_img, Wt90_p2, "Wt90")

    # Save to folder
    # scale = 0.5  # shrink to 50%
    # rotated_img = cv2.resize(rotated_img, None, fx=scale, fy=scale)
    stored_path = output_dir 
    new_img_name = img_name 
    output_path = stored_path + "/" + new_img_name
    cv2.imwrite(output_path, rotated_img)

    print("Finish write img: ", img_name)

    return {
            "O": O,
            "Lx": abs(Lx),
            "Ly": abs(Ly),
            "Lxt": abs(Lxt),
            "Lxh": abs(Lxh),
            "Wh50": abs(Wh50),
            "Wh85": abs(Wh85),
            "Wh90": abs(Wh90),
            "Wt50": abs(Wt50),
            "Wt85": abs(Wt85),
            "Wt90": abs(Wt90),
            "Sx": abs(Sx),
            "Sxt": abs(Sxt),
            "Sxh": abs(Sxh),
            "Sxd": abs(Sxd),
            "Seh": abs(Seh),
            "Set": abs(Set),
            "No303": abs(No303)
        }

if __name__ == "__main__":
    start = time.time()

    img_dir = "D:/DATN/Feature Extraction NewData/recon/inference"

    output_dir = "D:/DATN/Feature Extraction NewData/processed"

    output_csv_file = Path("D:/DATN/Feature Extraction NewData/features_2.csv")
    file_exists = output_csv_file.exists()

    skip_count = 0

    # Read existing filenames from CSV
    processed_files = set()

    if file_exists:
        with open(output_csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                processed_files.add(row["img_path"])

    for i, filename in enumerate(os.listdir(img_dir)):
        # if i >= 20:
        #     break

        # Skip if already processed
        if filename in processed_files:
            print(f"Skipping already processed file: {filename}")
            skip_count += 1
            print(skip_count)
            continue
        
 
        id = re.search(r'egg_(\d+)', filename)
        dot = re.search(r'dot_(\d+)', filename)
        ngay = re.search(r'ngay_(\d+)', filename)
        side = re.search(r'side_([A-Za-z]+)', filename)
        if id:
            egg_id = int(id.group(1))
            # print(egg_id)  # 11
        if dot:
            egg_dot = int(dot.group(1))
        if ngay:
            egg_ngay = int(ngay.group(1))
        if side:
            egg_side = side.group(1)

        file_path = os.path.join(img_dir, filename)
        try:
            result = feature_extraction(file_path, filename, output_dir)

            result["egg_id"] = egg_id
            result["batch"] = egg_dot
            result["day"] = egg_ngay
            result["side"] = egg_side
            result["img_path"] = filename

            with open(output_csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())

                if not file_exists:
                    writer.writeheader()
                    file_exists = True  # only write header once

                writer.writerow(result)

        except Exception as e:
            print("Processing img ", filename, " is aborted with error: ", e)
            # save skipped filename
            with open("skipped_files_2.txt", "a") as f:
                f.write(f"{filename}\n")
            skip_count += 1
            continue

    print("Skip Count: ", skip_count)

    end = time.time()

    print("Time ellapsed:", end - start, "seconds")



    

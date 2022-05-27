"""
This has code to identify which shape (from the SUAS list of shapes) is present
this file has the function id_shape() that takes an image that has been cropped using a
bounding box algorithm. Refer to its documentation. All other functions are helper
functions to make stuff nicer, please see id_shape() docstring to see potential
ability to skip redundant img pre-processing. Also to provide better pre-processing because mine
sucks.
"""
from typing import Optional, Tuple, Dict, List
import cv2
import numpy as np
import numpy.typing as npt
from config_odlc import POSSIBLE_COLORS


ODLC_SHAPES: Dict[int, str] = {
    3: "TRIANGLE",
    4: "quad",
    5: "PENTAGON",
    6: "HEXAGON",
    7: "HEPTAGON",
    8: "OCTAGON",
}


def get_contours(
    img_param: npt.NDArray[np.uint8], edge_detected: bool = False
) -> Tuple[Tuple[npt.NDArray[np.intc], ...], npt.NDArray[np.intc]]:
    """
    this will use Laplacian edge detection to get all the contours of any shapes in the image and
    return them in a tuple with the hierarchy

    parameters
    ----------
    img : npt.NDArray[np.uint8]
        the given image should be a BGR integer numpy image, and it should ideally be blurred
        already OR if edge_detected is true, then the given image should have already had edge
        detection applied

    edge_detected : bool, default=False
        this is a bool flag that is defaulted to false, if true then this function will assume
        that the given image is a B&W image that has had edge detection applied already and will
        skip applying it itself

    note
    ----------
    contours from cv2 have the form of [[[a, b]], [[c, d]], [[x, y]]] for some reason
    """
    edges: npt.NDArray[np.uint8]
    if edge_detected:
        edges = img_param
    else:
        edges = cv2.Laplacian(
            img_param, ddepth=cv2.CV_16S, ksize=3
        )  # cv2.Canny(img_param, 100, 100)
        edges = cv2.convertScaleAbs(edges)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges = np.where(edges > 30, np.uint8(0), np.uint8(255))
        # edges = cv2.GaussianBlur(edges, (7, 7), 0)
        # cv2.imshow("edges", edges)
        # cv2.waitKey(0)

    cnts: Tuple[npt.NDArray[np.intc]]
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(
    #     hierarchy,
    #     type(hierarchy),
    #     type(hierarchy[0]),
    #     type(hierarchy[0, 0]),
    #     type(hierarchy[0, 0, 0]),
    # )
    return cnts, hierarchy


def pick_out_shape(cnts: Tuple[npt.NDArray[np.intc], ...]) -> Tuple[npt.NDArray[np.intc], int]:
    """
    this will go through given tuple of contours and return the contour with the largest area

    notes
    ----------------
    contours in opencv (cv2) have the form of: [[[a, b]], [[c, d]], [[x, y]]] for some reason

    warnings
    ---------
    this function will not be accurate with contours that are self-intersecting or not closed, etc
    """
    index = 0
    max_area: float = 0
    cnt: npt.NDArray[np.intc]
    area: float = 0
    for i, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > max_area:
            index = i
            max_area = area
    # print(f"{cnts}\n{cnts[index]}\n{index}")
    return cnts[index], index


def get_angle(
    start_point: npt.NDArray[np.intc],
    middle_point: npt.NDArray[np.intc],
    end_point: npt.NDArray[np.intc],
) -> np.float64:
    """
    this will take 3 points from some contour and will calculate and return the angle between them
    in degrees

    parameters
    ----------
    start_point, middle_point, end_point : npt.NDArray[np.intc]
        (expected in form of [[n, m]] because thats what opencv does for the points making up a
        contour) These are 3 points in 2d space. The intended purpose is that they are
        consecutive points in a contour
    """
    start_line: npt.NDArray[np.intc] = start_point - middle_point
    end_line: npt.NDArray[np.intc] = end_point - middle_point
    cosine_angle: np.float64 = np.dot(start_line, end_line) / (
        np.linalg.norm(start_line) * np.linalg.norm(end_line)
    )
    return np.degrees(np.arccos(cosine_angle))


def get_angles(approx: npt.NDArray[np.intc]) -> npt.NDArray[np.float64]:
    """
    this takes a contour that has has been approximated with cv2.approxPolyDP() then will return a
    np array of the angles at each of the points in the contour
    """
    angles: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
    for i in range(len(approx)):
        angles = np.append(
            angles, get_angle(approx[i - 1, 0], approx[i, 0], approx[(i + 1) % len(approx), 0])
        )
    return angles


def compare_angles(
    angles: npt.NDArray[np.float64], angle: float, thresh: float = 0
) -> npt.NDArray[np.bool8]:
    """
    this takes a np array of "angles" (intended to be from a contour) and will compare them for
    equality all to a given "angle" within a given "thresh"-hold. Then will return a np array of
    bools for each angle in "angles", true if within "thresh" of "angle," false otherwise

    parameters
    ----------
    angles : npt.NDArray[np.float64]
        a np array of angles in degrees
    angle : float
        an angle in degrees that you want to compare the angle(s) in angles to
    thresh : float, default=0
        the margin of error allowed between your angle(s) and the angle you are comparing them to
        this is 0 by default, but that is really dumb in any real world example so dont do that
    """
    bools: npt.NDArray[np.bool8] = np.array([], dtype=np.bool8)
    for value in angles:
        bools = np.append(bools, -thresh < value - angle < thresh)
    return bools


def get_lengths(approx: npt.NDArray[np.intc]) -> npt.NDArray[np.float64]:
    """
    will return a np array of floats that is the lengths of all of the sides of the given (assumed
    approximated) closed contour
    """
    lengths: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
    for i in range(len(approx)):
        lengths = np.append(
            lengths,
            (
                (approx[i, 0, 0] - approx[(i + 1) % len(approx), 0, 0]) ** 2
                + (approx[i, 0, 1] - approx[(i + 1) % len(approx), 0, 1]) ** 2
            )
            ** 0.5,
        )
    return lengths


def _check_convexity_defect_shapes(approx: npt.NDArray[np.intc]) -> Optional[str]:
    """
    this function takes the approximated shape (so that it will have the correct number of sides)
    and will check if it is a star or plus by seeing how many convexity defects it
    has (angle inside > 180)
    """
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    if defects is not None:
        if len(defects) == 5:
            return "STAR"
        if len(defects) == 4:
            return "PLUS"
    return None


def _check_circular_shapes(
    bw_denoised: npt.NDArray[np.uint8], shape: npt.NDArray[np.intc]
) -> Optional[str]:
    """
    this function will take a 1 channel smoothed image and the originally found shape contours and
    will check the image for circles (will ignore small and huge circles to minimize false
    positives) then it identifies what kind of circle it is (semi, quarter or full)
    """
    circles = cv2.HoughCircles(
        bw_denoised,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=int(len(bw_denoised) / 2.5),
        maxRadius=int(len(bw_denoised) * 1.2),
    )

    if circles is not None:
        circle = np.array([[0, 0, 0]])
        for circles_circle in circles:
            if circles_circle[0][2] > circle[0][2]:
                circle = circles_circle

        max_dist = 0
        dist = 0
        perimeter = cv2.arcLength(shape, True)
        for i, point in enumerate(shape):
            dist = cv2.arcLength(np.array([point, shape[(i + 1) % len(shape)]]), False)
            if dist > max_dist:
                max_dist = dist
        # 0.280 is ratio of radius to total perimeter of quarter circle = r/(r + r + pi*r/2)
        if abs((max_dist / perimeter) - 0.280) < 0.05:
            return "QUARTER_CIRCLE"
        # 0.388 is ratio of diameter to total perimeter of semi circle = 2r/(2r + pi*2)
        if abs((max_dist / perimeter) - 0.388) < 0.05:
            return "SEMI_CIRCLE"
        return "CIRCLE"
    return None


def _check_polygons(approx: npt.NDArray[np.intc]) -> Optional[str]:
    """
    this will check for all of the easy shapes once all of the difficult ones have been eliminated
    """
    shape_name: Optional[str] = None

    shape_name = ODLC_SHAPES.get(len(approx))

    if shape_name == "quad":
        angles = get_angles(approx)
        if np.all(compare_angles(angles, 90, 5)):
            lengths = get_lengths(approx)
            if (lengths[0]) / (np.sum(lengths) / 4) < 0.05:
                shape_name = "SQUARE"
            else:
                shape_name = "RECTANGLE"
        else:
            shape_name = "TRAPEZOID"
    return shape_name


def find_color_in_shape(
    img_param: npt.NDArray[np.uint8],
    shape: npt.NDArray[np.intc],
    cnts: Tuple[npt.NDArray[np.intc], ...],
    hierarchy: npt.NDArray[np.intc],
    shape_index: int,
) -> npt.NDArray[np.uint8]:
    """
    function to get the BGR (RGB but backwards bc opencv says so) color of a shape, then converts
    to hsv
    Quick explanation because i'm proud of it: It makes a mask based off of the contour, then
    will subtract all areas where there are contours inside of the shape, then take all of the
    masked pixels and find their average to get the color of the shape

    parameters
    ----------
    img_param: npt.NDArray[np.uint8]
        the original image
    shape: npt.NDArray[np.intc]
        original contour of the shape
    cnts: Tuple[npt.NDArray[np.intc], ...]
        the originally found list of contours
    hierarchy: npt.NDArray[np.intc]
        the hierarchy matrix found in get_contours()
    shape_index: int
        the index of the shape in cnts/hierarchy, provided by pick_out_shape()
    """
    mask = np.zeros(img_param.shape[:2], dtype=np.int8)
    mask = cv2.drawContours(mask, [shape], -1, (255), cv2.FILLED)

    current_index: int = hierarchy[0, shape_index, 2]
    while True:
        if cv2.contourArea(cnts[current_index]) / cv2.contourArea(shape) < 0.8:
            mask = cv2.drawContours(mask, [cnts[current_index]], -1, (0), cv2.FILLED)
        current_index = hierarchy[0, current_index, 0]
        if current_index == -1:
            break

    mask = np.where(mask > 0, True, False)

    shape_partition = np.array([(img_param[mask]).reshape(-1, 3)])

    color = np.array(np.uint8(np.average(shape_partition, 1)))

    color = cv2.cvtColor(np.array([color]), cv2.COLOR_BGR2HSV)
    # print(color)
    return color


def parse_color(color: npt.NDArray[np.uint8]) -> Optional[str]:
    """
    takes in a 1 pixel hsv image and will compare it to a list of
    color ranges in ODLC_COLORS and will return the matching string in ODLC_COLORS
    or None if no colors match
    NOTE: Largely copied from _parse_color in standard_obj_text.py
    """
    matched: List[str] = []
    for col, ranges in POSSIBLE_COLORS.items():
        if len(ranges) > 2:  # red has 2 ranges
            if (cv2.inRange(color, np.array([ranges[1]]), np.array([ranges[0]]))[0, 0] == 255) or (
                cv2.inRange(color, np.array([ranges[3]]), np.array([ranges[2]]))[0, 0] == 255
            ):
                matched.append(col)
        elif cv2.inRange(color, np.array([ranges[1]]), np.array([ranges[0]]))[0, 0] == 255:
            matched.append(col)

    ## Determine distance to center to choose color if falls in multiple ##
    color_name = None  # returns None if no match
    if len(matched) > 1:  # 2+ matched colors
        # find color with min dist to color value
        best_dist: float = float("inf")

        for col in matched:
            dist: float = float("inf")
            # get midpoint value of color range
            if len(POSSIBLE_COLORS[col]) > 2:  # handle red's 2 ranges
                mid1: float = np.mean(POSSIBLE_COLORS[col][:2])  # midpoint of range 1
                mid2: float = np.mean(POSSIBLE_COLORS[col][2:])  # midpoint of range 2
                dist = min(  # min dist of color to range mid
                    np.sum(np.abs(color - mid1)),
                    np.sum(np.abs(color - mid2)),
                )
            else:  # any color except red
                mid: float = np.mean(POSSIBLE_COLORS[col])  # midpoint of range
                dist = np.sum(np.abs(color - mid))  # dist of color to range mid

            if dist < best_dist:  # color with min distance is the color chosen
                best_dist = dist
                color_name = col
    elif len(matched) == 1:  # single matched color
        color_name = matched[0]
    return color_name


def id_shape(
    img_param: npt.NDArray[np.uint8],
    procd_img_param: Optional[npt.NDArray[np.uint8]] = None,
    edge_detected: Optional[npt.NDArray[np.uint8]] = None,
    cnts_param: Optional[Tuple[npt.NDArray[np.intc], ...]] = None,
    hierarchy_param: Optional[npt.NDArray[np.intc]] = None,
    find_color: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    this is the main driver function for finding out what a given shape is. It assumes that the
    given image is a cropped image of just the bounding box arround the shape to be identified
    As in the whole shape is in the frame, but it comes as close to possible to filling the whole
    frame (but you dont need to do any weird image rotation, just an upright bounding box)

    parameters
    -----------
    img_param : npt.NDArray[np.uint8]
        this is the unaltered (except for being cropped to just the shape in question) image
    procd_img_param : npt.NDArray[np.uint8], Optional
        If the image has already been denoised/blurred earlier in the
        pipeline, then it can be passed in as a parameter to avoid doing the same work twice
        Also it is supposed to be processed_image_parameter but I suck at abbrev.
    edge_detected : npt.NDArray[np.uint8], Optional
        If the image has already had edge detection applied to it earlier in the pipeline, then it
        can be passed in to avoid doing work twice
    cnts_param : Tuple[npt.NDArray[np.intc], ...], Optional
        If the contour of the shape has already been found, then it can be passed in as a
        parameter. The contour should be passed inside of a tuple, so that if multiple contours
        are within the bounding box for the shape then my code can pick out the most relevant
        (biggest) one.
    hierarchy_param: Optional[Tuple[npt.NDArray[np.intc], ...]]
        This the data for the relations between contours that is provided by cv2.findCountours()
        NOTE: If the contours are provided then the hierarchy MUST also be provided
    find_color: bool
        A flag to determine whether the function to identify the color of the shape should be used
        True is default and will find the color.

    returns
    ---------
    Will return a tuple of 2 strings in the form of
    (shape_name, shape_color) unless find_color is set to False then will be (shape_name, None)
    will return one of: "STAR" "PLUS" "QUARTER_CIRCLE" "SEMI_CIRCLE" "CIRCLE" "TRIANGLE" "SQUARE"
    "RECTANGLE" "TRAPEZOID" "PENTAGON" "HEXAGON" "HEPTAGON" or "OCTOGON" depending on the shape
    The color element in the returned tuple will also be None if the color could not be matched

    raises
    -------
    ValueError
        with message "If contours are provided than the hierarchy from cv2.findContours() must
        also be provided" If you already found the contours then you must also provide the
        hierarchy provided from cv2.findContours must also be provided for my stuff to work
    """
    shape_name: Optional[str] = None
    color_name: Optional[str] = None
    procd_img: npt.NDArray[np.uint8] = np.zeros(1, np.uint8)

    if procd_img_param is None:
        procd_img = cv2.GaussianBlur(img_param, (3, 3), 0)

        procd_img = cv2.cvtColor(procd_img, cv2.COLOR_BGR2GRAY)
    else:
        procd_img = procd_img_param

    cv2.imshow("pre-processing", procd_img)
    cv2.waitKey(0)

    cnts: Tuple[npt.NDArray[np.intc], ...]
    hierarchy: npt.NDArray[np.intc]

    if cnts_param is None:
        if edge_detected is None:
            cnts, hierarchy = get_contours(procd_img)
        else:
            cnts, hierarchy = get_contours(img_param=edge_detected, edge_detected=True)
    elif hierarchy_param is not None:
        cnts = cnts_param
        hierarchy = hierarchy_param
    else:
        raise ValueError("If contours are provided than the hierarchy from cv2.findContours()")
    # print(cnts)
    bw_denoised: npt.NDArray[np.uint8] = procd_img  # cv2.cvtColor(procd_img, cv2.COLOR_BGR2GRAY)

    shape, shape_index = pick_out_shape(cnts)
    peri = cv2.arcLength(shape, True)
    approx = cv2.approxPolyDP(shape, 0.02 * peri, True)

    img_copy = np.copy(img_param)
    cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 1)
    cv2.drawContours(img_copy, [shape], -1, (255, 0, 0), 1)
    cv2.imshow("test", img_copy)
    cv2.waitKey(0)

    shape_name = _check_convexity_defect_shapes(approx=approx)

    if shape_name is None:
        shape_name = _check_circular_shapes(bw_denoised=bw_denoised, shape=shape)

    if shape_name is None:
        shape_name = _check_polygons(approx=approx)

    if shape_name is not None:
        if find_color:
            color_name = parse_color(
                find_color_in_shape(img_param, shape, cnts, hierarchy, shape_index)
            )
    return shape_name, color_name


if __name__ == "__main__":
    img: npt.NDArray[np.uint8] = cv2.imread(
        "C:/Users/natem/code/multirotor/standard-object-detection-testing/odlc_test/DJI_0413.JPG"
    )
    img = img[2005:2375, 1910:2290]
    cv2.imshow("img", img)
    cv2.waitKey(0)
    # if you have already done pre-processing you can add it, see id_shape() docs
    print(id_shape(img))

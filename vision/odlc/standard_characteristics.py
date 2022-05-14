"""
This has code to identify which shape (from the SUAS list of shapes) is present
this file has the function id_shape() that takes an image that has been cropped using a
bounding box algorithm. Refer to its documentation. All other functions are helper
functions to make stuff nicer, please see id_shape() docstring to see potential
ability to skip redundant img pre-processing
"""
from typing import Optional, Tuple, Dict
import cv2
import numpy as np
import numpy.typing as npt


ODLC_SHAPES: Dict[int, str] = {
    3: "TRIANGLE",
    4: "quad",
    5: "PENTAGON",
    6: "HEXAGON",
    7: "HEPTAGON",
    8: "OCTAGON",
}

# in BGR
# values for colors were obtained by finding the RGB (BGR)
# color for the color that is widely accepted (ie 0, 0, 0 for black)
# then with gray in the middle I have a sphere that i move all of the color
# points onto (still in the same vector direction away from gray, but now all
# the same distance from gray)
# https://www.desmos.com/calculator/zokjuf8rm9
# link to the desmos page where I did some formulas
# can put in a color and get out the new color that is a set distance from gray
# but still in the same direction as the original
ODLC_COLORS: Tuple[Tuple[npt.NDArray[np.int64], str], ...] = (
    (np.array([185, 185, 185]), "WHITE"),
    (np.array([69, 69, 69]), "BLACK"),
    (np.array([127, 127, 127]), "GRAY"),
    (np.array([69, 69, 185]), "RED"),
    (np.array([185, 69, 69]), "BLUE"),
    (np.array([69, 185, 69]), "GREEN"),
    (np.array([198, 57, 127]), "PURPLE"),
    (np.array([70, 185, 185]), "YELLOW"),
    (np.array([58, 148, 196]), "ORANGE"),
    (np.array([42, 76, 110]), "BROWN"),
)


def get_contours(
    img_param: npt.NDArray[np.uint8], edge_detected: bool = False
) -> Tuple[Tuple[npt.NDArray[np.intc], ...], npt.NDArray[np.intc]]:
    """
    this will use canny edge detection to get all the contours of any shapes in the image and
    return them in a tuple

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
        edges = cv2.Canny(img_param, 100, 100)

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
    for i, cnt in enumerate(cnts):
        if cv2.contourArea(cnt) > max_area:
            index = i
            max_area = cv2.contourArea(cnt)
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
            # print("Star detected")
            return "STAR"
        if len(defects) == 4:
            # print("Plus detected")
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
    function to get the BGR (RGB but backwards bc opencv says so) color of a shape.
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
        the hierarchy matrix provided by get_contours()
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

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    mask = np.where(mask > 0, True, False)

    shape_partition = np.array([(img_param[mask]).reshape(-1, 3)])

    color = np.array(np.uint8(np.average(shape_partition, 1)))
    # print(color)
    return color


def parse_color(color: npt.NDArray[np.uint8]) -> Optional[str]:
    """
    takes in a 1 pixel bgr image and will convert to hsv to compare it to a list of
    color ranges in ODLC_COLORS and will return the matching string in ODLC_COLORS
    or none if no colors match
    """
    dist = 450.0  # bigger then the longest possible distance
    best_color: Optional[str] = None
    for pair in ODLC_COLORS:
        if np.linalg.norm(color - pair[0]) < dist:
            dist = float(np.linalg.norm(color - pair[0]))
            best_color = pair[1]
    return best_color


def id_shape(
    img_param: npt.NDArray[np.uint8],
    procd_img_param: Optional[npt.NDArray[np.uint8]] = None,
    edge_detected: Optional[npt.NDArray[np.uint8]] = None,
    cnts: Optional[Tuple[npt.NDArray[np.intc], ...]] = None,
    find_color: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    this is the main driver function for finding out what a given shape is. It assumes that the
    given image is a cropped image of just the bounding box arround the shape to be identified
    As in the whole shape is in the frame, but it comes as close to possible to filling the whole
    frame

    parameters
    -----------
    img_param : npt.NDArray[np.uint8]
        this is the unaltered (except for being cropped to just the shape in question) image
    procd_img_param : npt.NDArray[np.uint8], Optional
        If the image has already been denoised/blurred (Some testing suggested that denoised
        images yield better results with Canny edge detection than blurred ones) earlier in the
        pipeline, then it can be passed in as a parameter to avoid doing the same work twice
        I ended up doing both denoising, and blurring for best results in my included
        preproccessing. Also it is supposed to be processed_image_parameter but I suck at abbrev.
    edge_detected : npt.NDArray[np.uint8], Optional
        If the image has already had edge detection applied to it earlier in the pipeline, then it
        can be passed in to avoid doing work twice
    cnts : Tuple[npt.NDArray[np.intc], ...], Optional
        If the contour of the shape has already been found, then it can be passed in as a
        parameter. The contour should be passed inside of a tuple, so that if multiple contours
        are within the bounding box for the shape then my code can pick out the most relevant
        (biggest) one.
    find_color: bool
        A flag to determine whether the function to identify the color of the shape should be used
        True is default and will find the color.

    returns
    ---------
    Will return a tuple of 2 strings (unless it raises an error, see below) in the form of
    (shape_name, shape_color) unless find_color is set to False then will be (shape_name, None)
    will return one of: "STAR" "PLUS" "QUARTER_CIRCLE" "SEMI_CIRCLE" "CIRCLE" "TRIANGLE" "SQUARE"
    "RECTANGLE" "TRAPEZOID" "PENTAGON" "HEXAGON" "HEPTAGON" or "OCTOGON" depending on the shape
    The color element in the returned tuple will also be false if the color could not be matched

    raises
    -------
    RuntimeError
        with message "Shape failed all checks, make sure it is not a tree or something"
        this means that the shape could not be matched to any of the possible shapes specified
        by the rules, or that my code doesn't work
    """
    shape_name: Optional[str] = None
    procd_img: npt.NDArray[np.uint8]

    if procd_img_param is None:
        procd_img = cv2.fastNlMeansDenoisingColored(
            src=img_param, dst=procd_img, templateWindowSize=7, searchWindowSize=21, h=10, hColor=10
        )
        procd_img = cv2.GaussianBlur(procd_img, ksize=(5, 5), sigmaX=5, sigmaY=5)
    else:
        procd_img = procd_img_param

    if cnts is None:
        if edge_detected is None:
            cnts, hierarchy = get_contours(procd_img)
        else:
            cnts, hierarchy = get_contours(img_param=edge_detected, edge_detected=True)
    # print(cnts)
    bw_denoised: npt.NDArray[np.uint8] = cv2.cvtColor(procd_img, cv2.COLOR_BGR2GRAY)

    shape, shape_index = pick_out_shape(cnts)
    peri = cv2.arcLength(shape, True)
    approx = cv2.approxPolyDP(shape, 0.02 * peri, True)

    # img_copy = np.copy(img_param)
    # cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 1)
    # cv2.imshow("test", img_copy)
    # cv2.waitKey(0)

    try:
        shape_name = _check_convexity_defect_shapes(approx=approx)
    except Exception as problem:
        raise RuntimeError(
            "Contour detection error: physically impossible self-intersections present"
        ) from problem

    if shape_name is None:
        shape_name = _check_circular_shapes(bw_denoised=bw_denoised, shape=shape)

    if shape_name is None:
        shape_name = _check_polygons(approx=approx)

    if shape_name is not None:
        color_name: Optional[str] = None
        if find_color:
            color_name = parse_color(
                find_color_in_shape(img_param, shape, cnts, hierarchy, shape_index)
            )
        return shape_name, color_name
    raise RuntimeError("Shape failed all checks, make sure it is not a tree or something")


if __name__ == "__main__":
    img: npt.NDArray[np.uint8] = cv2.imread(
        "C:/Users/natem/code/multirotor/standard-object-detection-testing/odlc_test/DJI_0408.JPG"
    )
    img = img[1250:1400, 3220:3390]
    cv2.imshow("img", img)
    cv2.waitKey(0)
    # if you have already done pre-processing you can add it, see id_shape() docs
    print(id_shape(img))

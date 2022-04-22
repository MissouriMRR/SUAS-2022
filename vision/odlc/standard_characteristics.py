"""
This has code to identify which shape (from the SUAS list of shapes) is present
this file has the function id_shape() that takes an image that has been cropped using a
bounding box algorithm. Refer to its documentation. All other functions are helper
functions to make stuff nicer, maybe they could be useful to you??
"""
from typing import Optional, Tuple, Dict
import cv2
import numpy as np
import numpy.typing as npt


ODLC_COLORS: Tuple[Tuple[npt.NDArray[np.int64], str], ...] = (
    (np.array([[180, 18, 255], [0, 0, 231]]), "WHITE"),
    (np.array([[180, 255, 30], [0, 0, 0]]), "BLACK"),
    (np.array([[180, 18, 230], [0, 0, 40]]), "GRAY"),
    (np.array([[180, 255, 255], [159, 50, 70]]), "RED"),
    (np.array([[9, 255, 255], [0, 50, 70]]), "RED"),
    (np.array([[128, 255, 255], [90, 50, 70]]), "BLUE"),
    (np.array([[89, 255, 255], [36, 50, 70]]), "GREEN"),
    (np.array([[158, 255, 255], [129, 50, 70]]), "PURPLE"),
    (np.array([[35, 255, 255], [25, 50, 70]]), "YELLOW"),
    (np.array([[24, 255, 255], [10, 50, 70]]), "ORANGE"),
    (np.array([[20, 255, 180], [10, 100, 120]]), "BROWN"),
)


def get_contours(
    img_param: npt.NDArray[np.uint8], edge_detected: bool = False
) -> Tuple[npt.NDArray[np.intc], ...]:
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
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def pick_out_shape(cnts: Tuple[npt.NDArray[np.intc], ...]) -> npt.NDArray[np.intc]:
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
    return cnts[index]


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
    shape_names: Dict[int, str] = {
        3: "TRIANGLE",
        4: "quad",
        5: "PENTAGON",
        6: "HEXAGON",
        7: "HEPTAGON",
        8: "OCTAGON",
    }

    shape_name = shape_names.get(len(approx))

    if shape_name == "quad":
        angles = get_angles(approx)
        if np.all(compare_angles(angles, 90, 5)):
            lengths = get_lengths(approx)
            if (lengths[0] - np.sum(lengths) / 4) / (np.sum(lengths) / 4) < 0.05:
                shape_name = "SQUARE"
            else:
                shape_name = "RECTANGLE"
        else:
            shape_name = "TRAPEZOID"
    return shape_name


def find_colors_in_shape(
    img_param: npt.NDArray[np.uint8],
    # cnts: Tuple[npt.NDArray[np.intc], ...],
    shape: npt.NDArray[np.intc],
) -> None:
    """
    Not written yet, i just need pylint to shut up for a sec.
    """
    height: int
    width: int
    height, width, _ = img_param.shape

    start: int = -1
    end: int = -1

    for x in range(width):
        point = [x, height // 2]
        if cv2.pointPolygonTest(shape, point, False) == 0:
            if start == -1:
                start = x + 4
            else:
                end = x - 4
                break

    line: npt.NDArray[np.uint8] = img_param[start:end, height // 2, :]

    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    floaty_line: npt.NDArray[np.float32] = np.float32(line)

    _2, label, center = cv2.kmeans(
        floaty_line, 2, bestLabels=None, criteria=term_criteria, attempts=10, flags=0
    )

    clusters = np.uint8(center)[label.flatten()]
    clusters = clusters.reshape((line.shape))


def id_shape(
    img_param: npt.NDArray[np.uint8],
    denoised: Optional[npt.NDArray[np.uint8]] = None,
    edge_detected: Optional[npt.NDArray[np.uint8]] = None,
    cnts: Optional[Tuple[npt.NDArray[np.intc], ...]] = None,
) -> str:
    """
    this is the main driver function for finding out what a given shape is. It assumes that the
    given image is a cropped image of just the bounding box arround the shape to be identified
    As in the whole shape is in the frame, but it comes as close to possible to filling the whole
    frame

    parameters
    -----------
    img : npt.NDArray[np.uint8]
        this is the unaltered (except for being cropped to just the shape in question) image
    denoised : npt.NDArray[np.uint8], Optional
        If the image has already been denoised/blurred (Some testing suggested that denoised
        images yield better results with Canny edge detection than blurred ones) earlier in the
        pipeline, then it can be passed in as a parameter to avoid doing the same work twice
    edge_detected : npt.NDArray[np.uint8], Optional
        If the image has already had edge detection applied to it earlier in the pipeline, then it
        can be passed in to avoid doing work twice
    cnts : Tuple[npt.NDArray[np.intc], ...], Optional
        If the contour of the shape has already been found, then it can be passed in as a
        parameter. The contour should be passed inside of a tuple, so that if multiple contours
        are within the bounding box for the shape then my code can pick out the most relevant
        (biggest) one.

    returns
    ---------
    will return one of: "star" "plus" "quarter-circle" "semi-circle" "circle" "triangle" "square"
    "rectangle" "trapezoid" "pentagon" "hexagon" "heptagon" or "octogon" depending on the shape
    (unless it raises an error, see below)

    raises
    -------
    RuntimeError
        with message "Shape failed all checks, make sure it is not a tree or something"
        this means that the shape could not be matched to any of the possible shapes specified
        by the rules, or that my code doesn't work
    """
    shape_name: Optional[str] = None

    if denoised is None:
        denoised = cv2.fastNlMeansDenoisingColored(
            src=img_param, dst=denoised, templateWindowSize=7, searchWindowSize=21, h=5, hColor=10
        )

    if cnts is None:
        if edge_detected is None:
            cnts = get_contours(denoised)
        else:
            cnts = get_contours(img_param=edge_detected, edge_detected=True)

    bw_denoised: npt.NDArray[np.uint8] = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    shape = pick_out_shape(cnts)
    peri = cv2.arcLength(shape, True)
    approx = cv2.approxPolyDP(shape, 0.01 * peri, True)

    shape_name = _check_convexity_defect_shapes(approx=approx)

    if shape_name is None:
        shape_name = _check_circular_shapes(bw_denoised=bw_denoised, shape=shape)

    if shape_name is None:
        shape_name = _check_polygons(approx=approx)
    find_colors_in_shape(img_param=img_param, shape=shape)
    if shape_name is not None:
        return shape_name
    raise RuntimeError("Shape failed all checks, make sure it is not a tree or something")


if __name__ == "__main__":
    img: npt.NDArray[np.uint8] = cv2.imread(
        "C:/Users/natem/code/multirotor/standard-object-detection-testing/quartercircle.jpg"
    )
    cv2.imshow("img", img)
    cv2.waitKey(0)

    print(id_shape(img))

import cv2
import numpy as np
import imutils


def resize(img, scale):
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    dimensions = (new_width, new_height)
    return cv2.resize(img, dimensions)


# This gets the actual angle of the edge of the camera view; this can be derived using a square pyramid with height 1
def edge_angle(horizontal_angle, vertical_angle):
    return 2 * np.arctan(np.tan(horizontal_angle / 2) * np.cos(vertical_angle / 2))


def calculate_length(horizontal_angle, vertical_angle):
    return 2 * 1/(np.cos(vertical_angle / 2)) * np.tan(horizontal_angle / 2)


def partial_deskew(image, camera_data):
    # Save the original size and aspect ratio for later
    og_width = image.shape[1]
    og_height = image.shape[0]
    og_aspect_ratio = og_width / og_height

    camera_pitch = camera_data.get('camera_pitch')

    # If only one FOV is given, calculate the missing FOV using the aspect ratio and the existing FOV
    if 'fov_horizontal' not in camera_data:
        fov_vertical = camera_data.get('fov_vertical')

        fov_horizontal = 2 * np.arctan(og_aspect_ratio * np.tan(fov_vertical / 2))
    elif 'fov_vertical' not in camera_data:
        fov_horizontal = camera_data.get('fov_horizontal')

        fov_vertical = 2 * np.arctan(np.tan(fov_horizontal / 2) / og_aspect_ratio)
    else:
        fov_horizontal = camera_data.get('fov_horizontal')
        fov_vertical = camera_data.get('fov_vertical')

    # Calculate the actual top and bottom FOV of the camera
    fov_edge = edge_angle(fov_horizontal, fov_vertical)

    # Calculate the ratio of the lengths of the top and bottom of the image
    top = np.cos(camera_pitch + fov_vertical/2)
    bottom = np.cos(camera_pitch - fov_vertical/2)

    # You can use either bottom/top or top/bottom depending on which way you want to transform
    distance_ratio = bottom / top

    # Add padding to the image so you can distort outside the image
    padding_size = int(((distance_ratio * image.shape[1]) - image.shape[1])/2)
    image = cv2.copyMakeBorder(image, 0, 0, padding_size, padding_size, cv2.BORDER_CONSTANT, None, value=0)

    # Find the length of the top of the image and the distance between the top and the bottom
    top_length = 2 * 1/(np.cos(camera_pitch + fov_vertical/2)) * np.tan(fov_edge/2)
    center_length = np.tan(camera_pitch + fov_vertical/2) - np.tan(camera_pitch - fov_vertical/2)

    # Divide the lengths to get the aspect ratio of the final image
    aspect_ratio = top_length / center_length

    # Set the dimensions to match the aspect ratio
    # height = int(image.shape[1] / aspect_ratio)
    # width = image.shape[1]

    height = image.shape[0]
    width = int(image.shape[0] * aspect_ratio)

    # Define the input and output points for the transformation
    input_pts = np.float32([[padding_size, 0],
                            [padding_size + og_width, 0],
                            [0, image.shape[0] - 1],
                            [image.shape[1], image.shape[0] - 1]])
    output_pts = np.float32([[0, 0],
                            [width - 1, 0],
                            [0, height - 1],
                            [width - 1, height - 1]])

    # Use the points to get the transformation matrix and use the matrix to transform the image
    matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    result = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)

    return result


def deskew(image, camera_data):
    aspect_ratio = image.shape[1] / image.shape[0]

    camera_pitch = camera_data.get('camera_pitch')
    camera_pitch = np.deg2rad(camera_pitch)

    # If only one FOV is given, calculate the missing FOV using the aspect ratio and the existing FOV
    if 'fov_horizontal' not in camera_data:
        fov_vertical = camera_data.get('fov_vertical')
        fov_vertical = np.deg2rad(fov_vertical)

        fov_horizontal = 2 * np.arctan(aspect_ratio * np.tan(fov_vertical / 2))
    elif 'fov_vertical' not in camera_data:
        fov_horizontal = camera_data.get('fov_horizontal')
        fov_horizontal = np.deg2rad(fov_horizontal)

        fov_vertical = 2 * np.arctan(np.tan(fov_horizontal / 2) / aspect_ratio)
    else:
        fov_horizontal = camera_data.get('fov_horizontal')
        fov_vertical = camera_data.get('fov_vertical')

        fov_horizontal = np.deg2rad(fov_horizontal)
        fov_vertical = np.deg2rad(fov_vertical)

    # Do a partial deskew of the image to deskew based on the pitch, leaving only the roll
    image = partial_deskew(image, {'camera_pitch': camera_pitch, 'fov_horizontal': fov_horizontal})

    if 'camera_roll' in camera_data and camera_data.get('camera_roll') != 0:
        camera_roll = camera_data.get('camera_roll')
        camera_roll = np.deg2rad(camera_roll)

        # Rotate 90 degrees and add padding to create a new perspective view with the original roll as the pitch
        image = imutils.rotate_bound(image, 90)

        # Calculate the padding needed to create the new perspective image - this is based on trig distances
        top_distance = np.tan(camera_pitch + fov_vertical / 2)
        bottom_distance = np.tan(camera_pitch - fov_vertical / 2)
        image_length = top_distance - bottom_distance
        conversion_rate = image.shape[1] / image_length

        bottom_distance = int(conversion_rate * bottom_distance)

        padding_size = 2 * bottom_distance + image.shape[1]

        flipped = cv2.flip(image, 1)

        image = cv2.copyMakeBorder(image, 0, 0, padding_size, 0, cv2.BORDER_CONSTANT, None, value=0)

        # image[0:flipped.shape[0], 0:flipped.shape[1]] = flipped

        # cv2.imwrite("output.png", image)

        image = partial_deskew(image, {'camera_pitch': camera_roll, 'fov_horizontal': camera_pitch * 2 + fov_horizontal})

        image = imutils.rotate_bound(image, -90)
        # image = image[0:int(image.shape[0] / 2) + 1]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        image = image[y:y + h, x:x + w]  # Crop the image

    return image

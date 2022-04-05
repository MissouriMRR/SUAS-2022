"""
Bounding box objects represent an area in an image and
are used to convey information between flight and vision processes.
"""

from enum import Enum

from typing import Dict, Tuple, Union


class ObjectType(Enum):
    """
    Type of object that a BoundingBox represents.
    NOTE: The string of an enum is accessed with the name property.
    """

    STD_OBJECT = 0
    EMG_OBJECT = 1
    TEXT = 2


class BoundingBox:
    """
    A set of 4 coordinates that distinguish a region of an image.
    The order of the coordinates is (top-left, top-right, bottom-right, bottom-left).

    Parameters
    ----------
    vertices : Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        The main structure of the BoundingBox. Denotes the 4 coordinates
        representing a box in an image. Vertices is a tuple of 4 coordinates. Each
        coordinate consists of a tuple 2 integers.
    obj_type : ObjectType
        Enumeration that denotes what type of object the BoundingBox represents.
    attributes : Dict[str, Union[int, float, str, None]]
        Any additional attributes to convey about the object in the BoundingBox.
    """

    def __init__(
        self,
        vertices: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        obj_type: ObjectType,
        attributes: Dict[str, Union[int, float, str, None]],
    ) -> None:
        self.vertices = vertices
        self.obj_type = obj_type
        self.attributes = attributes

    def __repr__(self) -> str:
        """
        Returns a string representation of the BoundingBox
        that contains its id, object type, and vertices.

        Returns
        -------
        str
            The string representation of the BoundingBox object.
        """
        return f"BoundingBox[{id(self)}, {self.obj_type}]: {str(self.vertices)}"


if __name__ == "__main__":
    pass

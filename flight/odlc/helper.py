from typing import Iterable

def coord_conversion(coord: Iterable[int], system_dim: Iterable[int], start: str ="ij") -> Iterable[int]:
    """
    Converts between 'xy' coordinates to 'ij'

    Parameters
    ----------
    coord: Iterable[int]
        the coordinate to be converted
    system_dim: Iterable[int]
        The dimensions of the coordinate system
    start: str
        The coordinate system 'coord' is already in
    """

    return (coord[1] + 1, system_dim[1] - coord[0])

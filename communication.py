"""Shared communication object between flight and vision code"""


class Communication:
    """
    Object to communicate between flight and vision

    Methods
    -------
        __init__() -> None
            Initializes current state
    """
    def __init__(self) -> None:
        """
        Initializer for communication object
        """
        self.__state: str = "Start State"

    @property
    def current_state(self):
        """
        Getter to return string of current state

        Returns
        -------
            str
                Name of current state
        """
        return self.__state

    # Set member variable state to new state
    @current_state.setter
    def current_state(self, new_state: str):
        """
        Sets the string name for the current state

        Parameters
        ----------
            new_state: str
                Name of new state to update communication object
        """
        self.__state: str = new_state

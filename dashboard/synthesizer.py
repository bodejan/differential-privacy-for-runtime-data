import abc

class Synthesizer(abc.ABC):
    """
    Abstract base class for synthesizer implementations.

    Args:
        input_data (str): Path to the input CSV file containing the original dataset.
        session_id (str): Identifier for the session.

    Methods:
        request(num_tuples, dataset, **kwargs):
            Abstract method to generate synthetic data based on specific implementation.

    Attributes:
        _input_data (str): Path to the input CSV file containing the original dataset.
        _session_id (str): Identifier for the session.
    """
    def __init__(self, input_data: str, session_id: str):
        self._input_data = input_data
        self._session_id = session_id

    @abc.abstractmethod
    def request(self, num_tuples: int = 1000, dataset: str = "sort", **kwargs):
        """
        Abstract method to generate synthetic data based on specific implementation.

        Args:
            num_tuples (int, optional): Number of synthetic tuples to generate (default: 1000).
            dataset (str, optional): Name of the dataset being used (default: "sort").
            **kwargs: Additional keyword arguments specific to the implementation.

        Returns:
            Implementation-specific results.
        """
        raise NotImplementedError()

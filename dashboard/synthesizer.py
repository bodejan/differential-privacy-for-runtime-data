import abc


class Synthesizer(abc.ABC):

    def __init__(self, input_data: str, session_id: str):
        self._input_data = input_data
        self._session_id = session_id

    @abc.abstractmethod
    def request(self, num_tuples: int = 1000, dataset: str = "sort", **kwargs):
        raise NotImplementedError()

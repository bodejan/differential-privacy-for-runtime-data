import dataclasses
import json


@dataclasses.dataclass
class MetaInformation:
    """
    Represents metadata information associated with a session.

    Attributes:
        id (str): Unique identifier for the session.
        dataset_name (str): Name of the dataset associated with the session.
    """

    id: str
    dataset_name: str

    def save(self) -> None:
        """
        Saves the metadata information to a JSON file.

        The metadata information is serialized to a JSON file named '<id>.meta' in the 'temp' directory.

        Returns:
            None
        """
        with open(f'temp/{self.id}.meta', 'w+') as file:
            json.dump(dataclasses.asdict(self), file)

    @staticmethod
    def from_id_file(id: str) -> 'MetaInformation':
        """
        Loads metadata information from a JSON file based on the session ID.

        Args:
            id (str): The session ID.

        Returns:
            MetaInformation: An instance of MetaInformation loaded from the JSON file.
        """
        with open(f'temp/{id}.meta', 'r') as file:
            return MetaInformation(**json.load(file))

import dataclasses
import json


@dataclasses.dataclass
class MetaInformation:
    id: str
    dataset_name: str

    def save(self) -> None:
        with open(f'temp/{self.id}.meta', 'w+') as file:
            json.dump(dataclasses.asdict(self), file)

    @staticmethod
    def from_id_file(id: str) -> 'MetaInformation':
        with open(f'temp/{id}.meta', 'r') as file:
            return MetaInformation(**json.load(file))

from typing import Any


class MultilayerPerceptron:
    def __init__(self) -> None:
        ...

    def predict(self, attributes: tuple) -> Any:
        ...

    def predict_all(self, all_attributes: list[tuple]) -> list[Any]:
        ...

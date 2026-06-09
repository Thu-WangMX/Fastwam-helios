from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..schema import DatasetMetadata


class ModalityTransform(BaseModel, ABC):
    apply_to: list[str] = Field(..., description="The keys to apply the transform to.")
    training: bool = Field(default=True)
    _dataset_metadata: DatasetMetadata | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        assert self._dataset_metadata is not None, "Dataset metadata is not set. Call set_metadata() first."
        return self._dataset_metadata

    @dataset_metadata.setter
    def dataset_metadata(self, value: DatasetMetadata):
        self._dataset_metadata = value

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        self.dataset_metadata = dataset_metadata

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.apply(data)

    @abstractmethod
    def apply(self, data: dict[str, Any]) -> dict[str, Any]: ...

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class InvertibleModalityTransform(ModalityTransform):
    @abstractmethod
    def unapply(self, data: dict[str, Any]) -> dict[str, Any]: ...


class ComposedModalityTransform(ModalityTransform):
    transforms: list[ModalityTransform] = Field(...)
    apply_to: list[str] = Field(default_factory=list)
    training: bool = Field(default=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        for transform in self.transforms:
            transform.set_metadata(dataset_metadata)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(self.transforms):
            try:
                data = transform(data)
            except Exception as e:
                raise ValueError(f"Error applying transform {i} to data: {e}") from e
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(reversed(self.transforms)):
            if isinstance(transform, InvertibleModalityTransform):
                try:
                    data = transform.unapply(data)
                except Exception as e:
                    step = len(self.transforms) - i - 1
                    raise ValueError(f"Error unapplying transform {step} to data: {e}") from e
        return data

    def train(self):
        for transform in self.transforms:
            transform.train()

    def eval(self):
        for transform in self.transforms:
            transform.eval()

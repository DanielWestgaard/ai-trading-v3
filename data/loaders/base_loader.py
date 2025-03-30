from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    @abstractmethod
    def load(self, source, **kwargs):
        pass
from abc import ABC, abstractmethod
from torch import nn


class BaseModel(nn.Module, ABC):
    """
    Base class for all models in the project. 

    Methods:
        forward: Performs the forward pass of the model.
        _forward: Abstract method to be implemented by subclasses.
       
    """

    def forward(
        self, batch
    ):
        """
        Returns a tensor of shape (BATCH, EMB_DIM). This method handles the concatenation
        of the two embeddings in case the model processes the text pairs disjointly.

        Returns:
            torch.Tensor: Tensor of shape (BATCH, EMB_DIM).

        """

        x = self._forward(
            batch
        )  # (BATCH, EMB_DIM)
        return x

    @abstractmethod
    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Should return a tensor of shape (BATCH, EMB_DIM), a single embedding for
        the whole input.

        Returns:
            torch.Tensor: Tensor of shape (BATCH, EMB_DIM).

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        """

        raise NotImplementedError("forward not implemented")
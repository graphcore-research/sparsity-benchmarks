# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Union, TYPE_CHECKING
import pva

if TYPE_CHECKING:
    from reptil import Reptil


class ReptilNamespace:
    """
    Namespace class for seperating methods in the Reptil object.
    """

    def __init__(self, parent: Union["Reptil", "ReptilNamespace"]):
        """
        Namespace class for seperating methods in the Reptil object.

        Parameters
        ----------
        parent
            parent ReptilNamespace or Reptil object that owns this.

        Raises
        ------
        TypeError
            If the parent is not an instantiation of Reptil or
            ReptilNamespace
        """

        from reptil import Reptil

        if not isinstance(parent, (Reptil, ReptilNamespace)):
            raise TypeError(
                "Namespace constructor takes a Reptil or "
                "ReptilNamespace object"
            )

        self._parent = parent
        self._report = parent._report
        self._steps = parent._steps

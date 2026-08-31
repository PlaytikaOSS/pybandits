# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Independent per-action meta-model (the smab / per-action cmab dispatch pattern).

``SmabMetaModel[T]`` wraps a ``dict[ActionId, T]`` and routes each ``sample_proba`` /
``update`` / ``reset`` call to the relevant action's own model — no coordinated cross-action state.
This is the historical smab/cmab pattern lifted into the meta-model abstraction without any
behavioural change. Shared-state algorithms (e.g. the neural-linear cmab) live in
``cmab_meta_model.py`` instead.
"""

from typing import Any, Generic

import numpy as np

from pybandits.base import ActionId, BinaryReward
from pybandits.meta_model.base import ActionModelType, BaseMetaModel, SampleProbaResult
from pybandits.model import Beta, BetaCC, BetaDP, BetaMO, BetaMOCC
from pybandits.quantitative_model import Zooming, ZoomingCC, ZoomingDP


class SmabMetaModel(BaseMetaModel, Generic[ActionModelType]):
    """Meta-model that dispatches independently to per-action models.

    Wraps a ``dict[ActionId, ActionModelType]`` and routes each ``sample_proba`` /
    ``update`` / ``reset`` call to the relevant action's model. This is the
    historical cmab/smab pattern, lifted into the meta-model abstraction without
    any behavioural change.

    The ``actions`` dict is a regular Pydantic field. Pydantic v2 with its
    default ``revalidate_instances='never'`` does not deep-copy nested model
    instances during validation, so the original model references are preserved.
    This guarantees that mutations via ``update()`` are visible to callers that
    hold references to the original per-action model objects.

    Domain invariants (e.g. same number of objectives for smab; consistent
    input size / update method for cmab) are validated on the respective manager
    subclasses, not here.
    """

    actions: dict[ActionId, ActionModelType]  # type: ignore[valid-type]

    def sample_proba(
        self,
        rng: np.random.Generator,
        valid_action_ids: set[ActionId] | None = None,
        **kwargs: Any,
    ) -> dict[ActionId, SampleProbaResult]:
        return {
            action_id: model.sample_proba(rng=rng, **kwargs)
            for action_id, model in self.actions.items()
            if valid_action_ids is None or action_id in valid_action_ids
        }

    def update(
        self,
        actions: list[ActionId],
        rewards: list[BinaryReward] | list[list[BinaryReward]],
        quantities: list[float | list[float] | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Group rows by action and dispatch to each action's ``update``.

        ``quantities`` is row-aligned with ``actions`` by the actions manager (``_validate_update_params``
        validates equal lengths) before calling here. ``smab`` has no per-sample context (unlike cmab).
        """
        self._dispatch_per_action_update(actions=actions, rewards=rewards, quantities=quantities)

    def reset(self) -> None:
        for model in self.actions.values():
            model.reset()


# Module-level aliases for concrete parameterisations.
# These are required for pickling: Python's pickle resolves a class by looking
# up ``cls.__qualname__`` on ``sys.modules[cls.__module__]``.  Pydantic sets the
# qualname of a parameterised generic to e.g.
# ``SmabMetaModel[Beta | Zooming]``; defining the alias here (co-located with
# ``SmabMetaModel``) makes that attribute accessible on this module.
SmabMetaModelSO = SmabMetaModel[Beta | Zooming]
SmabMetaModelCC = SmabMetaModel[BetaCC | ZoomingCC]
SmabMetaModelDP = SmabMetaModel[BetaDP | ZoomingDP]
SmabMetaModelMO = SmabMetaModel[BetaMO]
SmabMetaModelMOCC = SmabMetaModel[BetaMOCC]

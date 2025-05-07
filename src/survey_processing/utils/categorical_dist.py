from typing import Dict, cast, Optional, Tuple, Sequence

import torch
from gluonts.torch.distributions import DistributionOutput
from torch.distributions import Categorical


class DiscreteCategorical(Categorical):
    """
    defines categorical distribution
    """

    def __init__(
        self,
        values: torch.Tensor,
        logits: torch.Tensor,
        validate_args: Optional[bool] = None,
    ):
        super().__init__(logits=logits, validate_args=validate_args)
        self.values = values

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        draw sample from distribution
        """
        return self.values[super().sample(sample_shape=sample_shape)]

    def log_prob(self, value):
        """
        compute log probability of value
        """
        transformed = torch.repeat_interleave(
            input=value.unsqueeze(-1), repeats=self.values.shape[0], dim=-1
        )
        transformed_values = self.values.reshape(
            shape=[1] * (transformed.ndim - 1) + [-1]
        )
        transformed = transformed - transformed_values
        transformed = transformed.abs().argmin(dim=-1)

        return super().log_prob(transformed)


class CategoricalOutput(DistributionOutput):
    """
    defines categorical distribution output for sequence model
    """

    distr_cls = DiscreteCategorical
    values: torch.Tensor

    def __init__(self, beta: float, values: Sequence) -> None:
        super().__init__(beta)
        self.values = (
            torch.tensor(values, dtype=torch.float32).unique().squeeze().cuda()
        )
        self.args_dim = cast(
            Dict[str, int],
            {"logit_%s" % i: 1 for i in range(self.values.shape[0])},
        )

    @classmethod
    def domain_map(cls, *logits) -> tuple:  # type: ignore

        return tuple(torch.abs(logit).squeeze(-1) for logit in logits)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> DiscreteCategorical:
        distr = torch.cat([d.unsqueeze(2) for d in distr_args], dim=2)
        return self.distr_cls(
            values=self.values,
            logits=distr,
        )

    @property
    def event_shape(self) -> Tuple:
        return ()

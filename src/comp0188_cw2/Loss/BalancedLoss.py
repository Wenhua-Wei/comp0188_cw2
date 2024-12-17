import torch
from typing import Dict, Optional

from ..Metric.WandBMetricOrchestrator import WandBMetricOrchestrator


class TrackerBalancedLoss:

    def __init__(
        self,
        loss_lkp:Dict,
        mo:Optional[WandBMetricOrchestrator]=None,
        name:Optional[str] = ""
        ):
        """Class for handling MultiLoss outputs

        Args:
            loss_lkp (Dict[str:torch.nn.modules.loss]): Dictionary containing 
            the required losses and lookup values. The lookup values should
            match those passed to the call method in act and pred parameters
        """
        self.loss_lkp = loss_lkp
        self.mo = mo
        self.__step = 1
        self.name = name

    def __call__(
        self, 
        pred:Dict[str, torch.Tensor],
        act:Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        """Evaluates the input values against the loss functions specified in
        the init.

        Args:
            act (Dict[str:torch.Tensor]): Dictionary of the
            form {"name_of_loss": actual_values}
            The keys should match the keys provided in the loss_lkp parameter,
            specified in the init
            pred (Dict[str:torch.Tensor]): Dictionary of the
            form {"name_of_loss": predicted_values}
            The keys should match the keys provided in the loss_lkp parameter,
            specified in the init

        Returns:
            Dict[str:Any]: Dictionary of evaluated results. The keys will match
            those provided in the multi_loss parameter
        """
        print(f"Prediction 'grp' shape: {pred['grp'].shape}")
        print(f"Target 'grp' shape: {act['grp'].shape}")

        loss = 0
        _metric_value_dict = {}
        for key in self.loss_lkp.keys():
            _loss = self.loss_lkp[key](pred[key], act[key])
            _metric_value_dict[f"{key}_{self.name}_loss"] = {
                "label":f"step_{self.__step}",
                "value":_loss
            }
            loss += _loss
        if self.mo is not None:
            self.mo.update_metrics(metric_value_dict=_metric_value_dict)
        out_loss = torch.mean(loss)
        self.__step += 1
        return out_loss

    # def __call__(
    #     self, 
    #     pred: Dict[str, torch.Tensor],
    #     act: Dict[str, torch.Tensor]
    # ) -> torch.Tensor:
    #     """Evaluates the input values against the loss functions specified in init."""
    #     loss = 0
    #     _metric_value_dict = {}

    #     print("///////////////////////////")

    #     # Debug: Check for NaNs or Infs in inputs
    #     for key in pred.keys():
    #         if torch.isnan(pred[key]).any() or torch.isinf(pred[key]).any():
    #             print(f"NaN/Inf detected in predictions for '{key}'")
    #             print("Predicted values:", pred[key])
    #             raise ValueError("NaN or Inf detected in predictions.")
    #         if torch.isnan(act[key]).any() or torch.isinf(act[key]).any():
    #             print(f"NaN/Inf detected in targets for '{key}'")
    #             print("Target values:", act[key])
    #             raise ValueError("NaN or Inf detected in targets.")

    #     for key in self.loss_lkp.keys():
    #         _loss = self.loss_lkp[key](pred[key], act[key])

    #         # Debug: Print loss per component
    #         print(f"Loss for {key}: {_loss.item()}")

    #         _metric_value_dict[f"{key}_{self.name}_loss"] = {
    #             "label": f"step_{self.__step}",
    #             "value": _loss.item()  # Ensure loss is printed as a scalar
    #         }
    #         loss += _loss

    #     if self.mo is not None:
    #         self.mo.update_metrics(metric_value_dict=_metric_value_dict)

    #     # Debug: Check for NaN/Inf in the final loss
    #     if torch.isnan(loss).any() or torch.isinf(loss).any():
    #         print("NaN/Inf detected in cumulative loss.")
    #         raise ValueError("NaN or Inf detected in cumulative loss.")

    #     out_loss = torch.mean(loss)  # Ensure final loss is scalar
    #     print(f"Step {self.__step}: Total Loss = {out_loss.item()}")
    #     self.__step += 1
    #     return out_loss

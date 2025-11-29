import logging
from functools import partial
from typing import Optional, Union

import torch
from torch.fx.proxy import TraceError

from ..utils import recursive_apply

logger = logging.getLogger(__name__)


class FeatureExtractorWithHook(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, train_parsed: dict, eval_parsed: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.train_parsed = train_parsed
        self.eval_parsed = eval_parsed
        self._features = {}
        self._hooks = []

        self.name_to_module = {}
        for path, module in model.named_modules():
            if path in train_parsed or path in eval_parsed:
                self.name_to_module[path] = module

    def _register_hooks(self, parsed: dict):
        self._remove_hooks()
        for name, info in parsed.items():
            module = self.name_to_module[name]

            def hook_fn(m, inp, out, req_grad, target):
                if not req_grad:
                    out = recursive_apply(out, torch.Tensor.detach)
                self._features[target] = out

            hook = partial(hook_fn, req_grad=info["requires_grad"], target=info["target_key"])
            self._hooks.append(module.register_forward_hook(hook))

    def _remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def forward(self, x: Union[torch.Tensor, tuple[torch.Tensor]]):
        self._features.clear()
        parsed = self.train_parsed if self.training else self.eval_parsed
        self._register_hooks(parsed)
        self.model(*x) if isinstance(x, (list, tuple)) else self.model(x)
        expected = {info["target_key"] for info in parsed.values()}
        if missing := expected - self._features.keys():
            raise RuntimeError(f"Some unexpected errors happened. Missing feature nodes: {missing}")
        return self._features.copy()

    def __del__(self):
        self._remove_hooks()


def _parse_for_hook(nodes: Optional[Union[list[str], dict[str, str]]]) -> Optional[dict[str, dict]]:
    if nodes is None:
        return None
    parsed = {}
    if isinstance(nodes, list):
        for name in nodes:
            parsed[name] = {"target_key": name, "requires_grad": False}
    elif isinstance(nodes, dict):
        for name, val in nodes.items():
            if isinstance(val, str):
                parsed[name] = {"target_key": val, "requires_grad": False}
            elif isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], dict):
                target_key, config = val
                parsed[name] = {
                    "target_key": target_key,
                    "requires_grad": config.get("requires_grad", False)
                }
            else:
                raise ValueError(f"Unsupported format in `create_feature_extractor` with hooks: {val}")
    return parsed


def create_feature_extractor(
    model: torch.nn.Module,
    mode: str = "fx",
    return_nodes: Optional[Union[list[str], dict[str, str]]] = None,
    train_return_nodes: Optional[Union[list[str], dict[str, str]]] = None,
    eval_return_nodes: Optional[Union[list[str], dict[str, str]]] = None,
) -> torch.nn.Module:
    """
    Create a feature extractor with fx.GraphModule or hooks.

    For further information on FX see the docs at
    `torchvision.orionvis.feature_extraction.create_feature_extractor`_.

    Args:
        model (torch.nn.Module): model on which we will extract the features
        return_nodes (list or dict, optional): either a ``List`` or a ``Dict``
            containing the names (or partial names - see note above)
            of the nodes for which the activations will be returned. If it is
            a ``Dict``, the keys are the node names, and the values
            are the user-specified keys for the graph module's returned
            dictionary. If it is a ``List``, it is treated as a ``Dict`` mapping
            node specification strings directly to output names. In the case
            that ``train_return_nodes`` and ``eval_return_nodes`` are specified,
            this should not be specified.
        mode (str, optional): either "fx" or "hook". Default is "fx".
        train_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``eval_return_nodes`` must also be specified,
            and ``return_nodes`` should not be specified.
        eval_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``train_return_nodes`` must also be specified,
            and `return_nodes` should not be specified.

    Examples::

        >>> # Feature extraction with resnet
        >>> import torchvision
        >>> backbone = torchvision.orionvis.resnet18()
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> feature_extractor = create_feature_extractor(backbone, {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = feature_extractor(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])), ('feat2', torch.Size([1, 256, 14, 14]))]

    """
    if all(arg is None for arg in [return_nodes, train_return_nodes, eval_return_nodes]):
        raise ValueError(
            "Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified"
        )

    if (train_return_nodes is None) ^ (eval_return_nodes is None):
        raise ValueError(
            "If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"
        )

    if not ((return_nodes is None) ^ (train_return_nodes is None)):
        raise ValueError("If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified")

    if mode not in ["fx", "hook"]:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'fx' and 'hook'.")

    if mode == "hook":
        base_parsed = _parse_for_hook(return_nodes)
        train_parsed = _parse_for_hook(train_return_nodes) or base_parsed
        eval_parsed = _parse_for_hook(eval_return_nodes) or base_parsed
        extractor = FeatureExtractorWithHook(model, train_parsed, eval_parsed)
        return extractor

    try:
        from torchvision.models.feature_extraction import create_feature_extractor as fx_create_feature_extractor
        extractor = fx_create_feature_extractor(
            model,
            return_nodes=return_nodes,
            train_return_nodes=train_return_nodes,
            eval_return_nodes=eval_return_nodes,
        )
        logger.info(f"Created feature extractor with FX.")
        return extractor
    except TraceError as e:
        base_parsed = _parse_for_hook(return_nodes)
        train_parsed = _parse_for_hook(train_return_nodes) or base_parsed
        eval_parsed = _parse_for_hook(eval_return_nodes) or base_parsed
        extractor = FeatureExtractorWithHook(model, train_parsed, eval_parsed)
        logger.warning(f"FX TraceError: {e}\nFallback to created feature extractor with hooks.")
        return extractor
    except Exception as e:
        raise e
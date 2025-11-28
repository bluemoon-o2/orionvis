import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module=__name__)
import torch
import torch.nn as nn
from .util.registry import Registry
MODEL = Registry('Model')
from .mobilemamba import MobileMamba_T2, MobileMamba_T4, MobileMamba_S6, MobileMamba_B1, MobileMamba_B2, MobileMamba_B4


__all__ = ['get_model']


def get_model(cfg_model: dict):
	model_name = cfg_model["name"]
	model_kwargs = {k: v for k, v in cfg_model["model_kwargs"].items()}
	model_fn = MODEL.get_module(model_name)
	checkpoint_path = model_kwargs.pop('checkpoint_path')
	ema = model_kwargs.pop('ema')
	strict = model_kwargs.pop('strict')
	device_map = model_kwargs.pop('device_map')
	dtype = model_kwargs.pop('dtype')

	model = model_fn(**model_kwargs)
	if checkpoint_path:
		ckpt = torch.load(checkpoint_path, map_location='cpu')
		if 'net' in ckpt.keys() or 'net_E' in ckpt.keys():
			state_dict = ckpt['net_E' if ema else 'net']
		else:
			state_dict = ckpt
		if not strict:
			no_ft_keywords = model.no_ft_keywords()
			for no_ft_keyword in no_ft_keywords:
				del state_dict[no_ft_keyword]
			ft_head_keywords, num_classes = model.ft_head_keywords()
			for ft_head_keyword in ft_head_keywords:
				if state_dict[ft_head_keyword].shape[0] <= num_classes:
					del state_dict[ft_head_keyword]
				elif state_dict[ft_head_keyword].shape[0] == num_classes:
					continue
				else:
					state_dict[ft_head_keyword] = state_dict[ft_head_keyword][:num_classes]

		if isinstance(model, nn.Module):
			model.load_state_dict(state_dict, strict=strict)
		else:
			for sub_model_name, sub_state_dict in state_dict.items():
				sub_model = getattr(model, sub_model_name, None)
				if sub_model:
					sub_model.load_state_dict(sub_state_dict, strict=strict)
	if device_map == "auto":
		device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return model.to(device=device_map, dtype=dtype)

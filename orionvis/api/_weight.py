import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, List, Tuple
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

from .register import get_transform

logger = logging.getLogger(__name__)


class TransformConfig(BaseModel):
    name: str = Field(..., description="Registered Transform name(must be in models.transforms)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Transform parameters")


class WeightMeta(BaseModel):
    num_classes: int = Field(..., description="Model output class number")
    params: Optional[float] = Field(None, description="Model parameter number(scale: 1e6)")
    flops: Optional[float] = Field(None, description="Model flops number(scale: 1e9)")
    training_data: Optional[str] = Field(None, description="Training dataset name")
    training_epoches: Optional[int] = Field(None, description="Training epochs number")
    accuracy: Optional[float] = Field(None, description="Model accuracy on validation set")
    release: Optional[str] = Field(None, description="Model release version")
    cfg_path: Optional[str] = Field(None, description="Path to model config file(relative/absolute)")


class WeightConfig(BaseModel):
    version: str = Field(..., description="Weight version(unique in model, like IMAGENET1K_V1)")
    url: str = Field(..., description="Weight file URL or local path(support env variables)")
    source: Optional[str] = Field(None, description="Weight file source provider")
    transform: TransformConfig = Field(..., description="Associated preprocessing Transform config")
    meta: WeightMeta = Field(..., description="Weight metadata")
    deprecated: Dict[bool, str] = Field(default_factory=dict, description="Whether this weight version is deprecated")
    md5: Optional[str] = Field(None, description="Weight file MD5 checksum(used for integrity verification)")


class ModelConfig(BaseModel):
    model_name: str = Field(..., description="Model unique identifier(global unique, like resnet50)")
    default: str = Field(..., description="Default weight version for this model")
    weights: List[WeightConfig] = Field(..., min_length=1, description="All weight configs for this model")


@dataclass(frozen=True)
class Weight:
    model_name: str          # Model unique identifier
    version: str             # Weight version
    url: str                 # Resolved URL/path
    transform: Callable      # Bound Transform function with params
    meta: Dict[str, Any]     # Flattened metadata (Pydantic model to dict)
    deprecated: Dict[bool, str]  # Whether this weight version is deprecated and why
    md5: Optional[str]       # Checksum

    def get_transform(self) -> Callable:
        return self.transform


class WeightConfigLoader:
    def __init__(self, weights_dir: str = None, num_workers: int = 2):
        package_root = Path(__file__).resolve().parents[1]
        self.weight_dir = Path(weights_dir) if weights_dir else package_root / "configs" / "weights"
        self.num_workers = num_workers
        if not self.weight_dir.exists():
            raise FileNotFoundError(f"Weight directory not found: {self.weight_dir}")
        self._index: Optional[Dict[str, Tuple[str, List[Weight]]]] = None

    def load_all(self) -> List[Tuple[str, List[Weight]]]:
        yaml_files = list(self.weight_dir.glob("*.yaml")) + list(self.weight_dir.glob("*.yml"))
        results: List[Tuple[str, List[Weight]]] = []
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    raw_docs = list(yaml.safe_load_all(f))
            except Exception as e:
                logger.error(f"Error reading weight config {yaml_file}: {e}")
                continue
            for raw_config in raw_docs:
                if raw_config is None:
                    continue
                try:
                    model_config = ModelConfig(**raw_config)
                except Exception as e:
                    logger.error(f"Invalid model config in {yaml_file}: {e}")
                    continue
                model_name = model_config.model_name
                default_version, weights = self._from_model_config(model_config)
                results.append((model_name, weights))
        return results

    def _load_single_weight_safe(self, yaml_file: Path) -> Optional[Tuple[str, List[Weight]]]:
        try:
            return self._load_single_weight(yaml_file)
        except Exception as e:
            logger.error(f"Error loading weight config {yaml_file}: {e}")
            return None

    @staticmethod
    def _from_model_config(model_config: ModelConfig) -> Tuple[str, List[Weight]]:
        model_name = model_config.model_name
        weights: List[Weight] = []
        for weight_config in model_config.weights:
            transform_func = get_transform(weight_config.transform.name)
            bound_transform = transform_func(**weight_config.transform.params)
            meta_dict = weight_config.meta.model_dump(exclude_none=True)
            weight = Weight(
                model_name=model_name,
                version=weight_config.version,
                url=weight_config.url,
                transform=bound_transform,
                meta=meta_dict,
                deprecated=weight_config.deprecated,
                md5=weight_config.md5
            )
            if weight.deprecated.get(True):
                logger.warning(f"Model {model_name} weight {weight.version} is deprecated: {weight.deprecated.get(True)}")
            weights.append(weight)
        return model_config.default, weights

    def _build_index(self) -> None:
        if self._index is not None:
            return
        yaml_files = list(self.weight_dir.glob("*.yaml")) + list(self.weight_dir.glob("*.yml"))
        index: Dict[str, Tuple[str, List[Weight]]] = {}
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    raw_docs = list(yaml.safe_load_all(f))
            except Exception as e:
                logger.error(f"Error indexing weight config {yaml_file}: {e}")
                continue
            for raw_config in raw_docs:
                if raw_config is None:
                    continue
                try:
                    model_config = ModelConfig(**raw_config)
                except Exception as e:
                    logger.error(f"Invalid model config in {yaml_file}: {e}")
                    continue
                model_name = model_config.model_name
                default_version, weights = self._from_model_config(model_config)
                index[model_name] = (default_version, weights)
        self._index = index

    def get_weight_by_version(self, model_name: str, version: Optional[str] = None) -> Weight:
        self._build_index()
        if self._index is None or model_name not in self._index:
            raise KeyError(f"Model {model_name} not found in {self.weight_dir}")
        default_version, weights = self._index[model_name]
        target_version = version or default_version
        for weight in weights:
            if weight.version == target_version:
                return weight
        available = [w.version for w in weights]
        raise KeyError(f"Model {model_name} has no weight version {target_version}, available versions: {available}")


class WeightsRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.loader = WeightConfigLoader()
            cls._instance._weights = {}
        return cls._instance

    def get(self, model_name: str, version: Optional[str] = None) -> Weight:
        return self.loader.get_weight_by_version(model_name, version)

    def list_models(self) -> List[str]:
        self.loader._build_index()
        if self.loader._index is None:
            return []
        return sorted(list(self.loader._index.keys()))

    def list_versions(self, model_name: str) -> List[str]:
        self.loader._build_index()
        if model_name in self.loader._index:
            return [w.version for w in self.loader._index[model_name][1]]
        raise KeyError(f"Model {model_name} not found. Available models: {self.list_models()}")

    def get_transform(self, model_name: str, version: Optional[str] = None) -> Callable:
        weight = self.get(model_name, version)
        return weight.get_transform()

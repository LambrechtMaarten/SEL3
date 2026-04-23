import os
from pathlib import Path
import numpy as np
import wandb
from configs.subconfigurations.logger.logger import Logger
from src.jax_extra.jax_extra import jarr
from src.render.render import save_video

def serialize(obj):
    """
    Recursively convert a configuration object to a serializable dict.
    - dataclasses and SubConfiguration objects are converted recursively.
    - lists/tuples are converted element-wise.
    - non-serializable objects are replaced by their class name.
    """
    import dataclasses

    if dataclasses.is_dataclass(obj):
        result = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = serialize(value)
        return result
    elif hasattr(obj, "to_dict"):
        # For SubConfiguration classes
        return {k: serialize(v) for k, v in vars(obj).items() if k != "configuration"}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # Fallback for non-serializable objects: store class name
        return type(obj).__name__


class WandbLogger(Logger):
    """
    Logger implementation using Weights & Biases.
    Automatically serializes nested Configuration objects.
    """

    def __init__(self, name: str = "wandb", project: str = "brittle-star-training"):
        super().__init__(name)
        self.project = project
        self.run = None
        self.generation = 0

    def init_logger(self):
        os.makedirs(self.base_folder, exist_ok=True)

        config_dict = None
        if hasattr(self, "configuration"):
            config_dict = serialize(self.configuration)

        self.run = wandb.init(
            project=self.project,
            name=self.name,
            dir=self.base_folder,
            config=config_dict
        )

    def log_configuration(self):
        if hasattr(self, "configuration"):
            wandb.config.update(serialize(self.configuration))

    def log(self, logging: dict):
        wandb.log(logging)

    def log_genetic_generation(self, population: jarr, selections: jarr, evaluations: jarr):
        evals = np.array(evaluations)

        metrics = {
            "generation": self.generation,
            "fitness/best": float(np.max(evals)),
            "fitness/mean": float(np.mean(evals)),
            "fitness/min": float(np.min(evals)),
            "fitness/std": float(np.std(evals)),
            "population/size": int(len(evals)),
        }

        wandb.log(metrics)
        self.generation += 1

    def log_video(self, frames, name: str, fps: int = 30):
        frames = np.array(frames)

        video_folder = Path(self.base_folder) / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        video_path = video_folder / f"{name}.mp4"

        save_video(frames, str(video_path))

        wandb.log({name: wandb.Video(str(video_path), fps=fps, format="mp4")})

    def log_model(self, model_path: str):
        artifact = wandb.Artifact(name="model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def log_controller(self, cpg_map):
        path = Path(self.base_folder) / "controller"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a") as f:
            if isinstance(cpg_map, dict):
                for direction, params in cpg_map.items():
                    line = f"{direction}: {np.array2string(np.array(params), precision=4)}\n"
                    f.write(line)
            else:
                line = np.array2string(np.array(cpg_map), precision=4) + "\n"
                f.write(line)

        artifact = wandb.Artifact(name="controller", type="controller")
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)

    def finish(self):
        if self.run:
            self.run.finish()
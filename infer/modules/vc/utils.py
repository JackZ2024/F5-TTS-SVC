import os

# from fairseq import checkpoint_utils

from transformers import HubertModel
from torch import nn


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

def load_hubert(config):

    hubert_model = HubertModelWithFinalProj.from_pretrained("./infer/assets/hubert")
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


# def load_hubert(config):
#     models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
#         ["infer/assets/hubert/hubert_base.pt"],
#         suffix="",
#     )
#     hubert_model = models[0]
#     hubert_model = hubert_model.to(config.device)
#     if config.is_half:
#         hubert_model = hubert_model.half()
#     else:
#         hubert_model = hubert_model.float()
#     return hubert_model.eval()

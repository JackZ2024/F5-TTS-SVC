import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.lib.audio import load_audio
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *
from infer.modules.vc.split_audio import process_audio, merge_audio


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        # logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        # to_return_protect1 = {
        #     "visible": self.if_f0 != 0,
        #     "value": (
        #         to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
        #     ),
        #     "__type__": "update",
        # }

        
        # person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {sid}")

        self.cpt = torch.load(sid, map_location="cpu")
        # print(self.cpt)
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        # n_spk = self.cpt["config"][-3]
        # index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        # logger.info("Select index: " + index["value"])

        # return (
        #     (
        #         {"visible": True, "maximum": n_spk, "__type__": "update"},
        #         to_return_protect0,
        #         # to_return_protect1,
        #         index,
        #         index,
        #     )
        #     if to_return_protect
        #     else {"visible": True, "maximum": n_spk, "__type__": "update"}
        # )

    def vc_single(
        self,
        spk,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        split_audio=True
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            if split_audio:
                chunks, intervals = process_audio(audio, 16000)
                print(f"Audio split into {len(chunks)} chunks for processing.")
            else:
                chunks = []
                chunks.append(audio)

            converted_chunks = []
            for c in chunks:
                audio_opt = self.pipeline.pipeline(
                    self.hubert_model,
                    self.net_g,
                    spk,
                    c,
                    input_audio_path,
                    times,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    self.if_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    f0_file,
                )

                converted_chunks.append(audio_opt)
                if split_audio:
                    print(f"Converted audio chunk {len(converted_chunks)}")

            if split_audio:
                audio_opt = merge_audio(chunks, converted_chunks, intervals, 16000, self.tgt_sr)
            else:
                audio_opt = converted_chunks[0]

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr

            return tgt_sr, audio_opt
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return None, None

    def vc_multi(
        self,
        spk,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                tgt_sr, audio_opt = self.vc_single(
                    spk,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if tgt_sr is not None:
                    try:
                        sf.write("%s/%s.%s" % (opt_root, os.path.basename(path), "wav"), audio_opt, tgt_sr)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()

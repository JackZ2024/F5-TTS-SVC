from __future__ import annotations

import gc
import os.path
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

import huggingface_hub
from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from collections.abc import Callable


class SelfDisposingModel:
    def __init__(
            self, model_id: str, load_fn: Callable[[], WhisperModel], ttl: int,
            unload_fn: Callable[[str], None] | None = None
    ) -> None:
        self.model_id = model_id
        self.load_fn = load_fn
        self.ttl = ttl
        self.unload_fn = unload_fn

        self.ref_count: int = 0
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.model: WhisperModel | None = None

    def unload(self) -> None:
        with self.rlock:
            if self.model is None:
                raise ValueError(f"Model {self.model_id} is not loaded. {self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"Model {self.model_id} is still in use. {self.ref_count=}")
            if self.expire_timer:
                self.expire_timer.cancel()
            self.model = None
            # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992
            gc.collect()
            print(f"Model {self.model_id} unloaded")
            if self.unload_fn is not None:
                self.unload_fn(self.model_id)

    def _load(self) -> None:
        with self.rlock:
            assert self.model is None
            print(f"Loading model {self.model_id}")
            start = time.perf_counter()
            self.model = self.load_fn()
            print(f"Model {self.model_id} loaded in {time.perf_counter() - start:.2f}s")

    def _increment_ref(self) -> None:
        with self.rlock:
            self.ref_count += 1
            if self.expire_timer:
                print(f"Model was set to expire in {self.expire_timer.interval}s, cancelling")
                self.expire_timer.cancel()
            print(f"Incremented ref count for {self.model_id}, {self.ref_count=}")

    def _decrement_ref(self) -> None:
        with self.rlock:
            self.ref_count -= 1
            print(f"Decremented ref count for {self.model_id}, {self.ref_count=}")
            if self.ref_count <= 0:
                if self.ttl > 0:
                    print(f"Model {self.model_id} is idle, scheduling offload in {self.ttl}s")
                    self.expire_timer = threading.Timer(self.ttl, self.unload)
                    self.expire_timer.start()
                elif self.ttl == 0:
                    print(f"Model {self.model_id} is idle, unloading immediately")
                    self.unload()
                else:
                    print(f"Model {self.model_id} is idle, not unloading")

    def __enter__(self) -> WhisperModel:
        with self.rlock:
            if self.model is None:
                self._load()
            self._increment_ref()
            assert self.model is not None
            return self.model

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        self._decrement_ref()


def get_whisper_folder():
    if os.path.exists("./whisper_models/whisper-medium-zh-ct2"):
        return "./whisper_models/whisper-medium-zh-ct2"
    else:
        return "deepdml/faster-whisper-large-v3-turbo-ct2"


class WhisperModelManager:
    def __init__(self) -> None:
        self.loaded_models: OrderedDict[str, SelfDisposingModel] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def download_model(repo_id):
        allow_patterns = [
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ]

        kwargs = {
            "allow_patterns": allow_patterns,
        }

        return huggingface_hub.snapshot_download(repo_id, **kwargs)

    def _load_fn(self, model_id: str) -> WhisperModel:
        if os.path.exists(model_id):
            return WhisperModel(model_id)
        else:
            return WhisperModel(self.download_model(model_id))

    def _handle_model_unload(self, model_name: str) -> None:
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]

    def unload_model(self, model_name: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_name)
            if model is None:
                raise KeyError(f"Model {model_name} not found")
            self.loaded_models[model_name].unload()

    def load_model(self) -> SelfDisposingModel:
        model_name = get_whisper_folder()
        print(f"Loading model {model_name}")
        with self._lock:
            print("Acquired lock")
            if model_name in self.loaded_models:
                print(f"{model_name} model already loaded")
                return self.loaded_models[model_name]
            self.loaded_models[model_name] = SelfDisposingModel(
                model_name,
                load_fn=lambda: self._load_fn(model_name),
                ttl=300,
                unload_fn=self._handle_model_unload,
            )
            return self.loaded_models[model_name]

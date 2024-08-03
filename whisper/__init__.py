# 2. after reading the README.md file, the next step is to read the __init__.py file.
# This script is a part of the Whisper ASR (Automatic Speech Recognition) model
# This file typically initializes the module and can provide insights into what components are included when the whisper package is imported.
# The code and the model weights of Whisper are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.

import hashlib           # Provides hashing algorithms.
import io                # Core tools for working with streams.
import os                # Provides a way to work with the operating system.
import urllib            # Provides a high-level interface for fetching data across the World Wide Web.
import warnings
from typing import List, Optional, Union   # The typing module defines a standard interface for type hints and annotations.

import torch
from tqdm import tqdm

from .whisper_audio_processing import load_audio, log_mel_spectrogram, pad_or_trim
from .others.decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import Whisper, ModelDimensions
from .others.transcribe_old import transcribe
from .others.transcribe_old import transcribe_audio
from .others.version import __version__

# The _MODELS dictionary contains the URLs for the pre-trained models. Dictionary mapping model names to their download URLs
_MODELS = {
    # "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    # "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://huggingface.co/openai/whisper-large-v3/resolve/main/pytorch_model.bin"
}

def download_model(model_name):
    if model_name in _MODELS:
        url = _MODELS[model_name]
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{model_name}.pt", 'wb') as f:
                f.write(response.content)
            print(f"{model_name} model downloaded.")
        else:
            print(f"Failed to download {model_name}. Status code: {response.status_code}")
    else:
        print(f"Model {model_name} not found.")

# _download: Downloads the model file from the specified URL to the given root directory. 
# It verifies the integrity of the downloaded file using SHA256 checksum. If in_memory is True, the model is loaded into memory; otherwise, the file path is returned.
def _download(url: str, root: str, in_memory: bool) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.")

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())

#load_model: This function loads a Whisper ASR model. It takes several parameters:
# name: The name of the model to load, which should be one of the official model names or a path to a model checkpoint.
# device: The PyTorch device (e.g., CPU or GPU) where the model will be loaded.
# download_root: The path to download the model files. By default, it uses the ~/.cache/whisper directory.
# in_memory: Whether to load the model weights into host memory.
# The function first determines the device
def load_model(name: str, device: Optional[Union[str, torch.device]] = None, download_root: str = None, in_memory: bool = False) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        download_root = os.getenv(
            "XDG_CACHE_HOME", 
            os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        )

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with (io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model.to(device)

import zlib
from typing import Iterator, TextIO

# This function performs exact division, ensuring there are no remainders. It is used to calculate the number of frames per output token.
def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

# This function converts an optional integer argument from a string, providing a default value if the argument is not provided.
def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string): # This function converts an optional float argument from a string, providing a default value if the argument is not provided.
    return None if string == "None" else float(string)

# This function calculates the compression ratio of a given text string using the zlib library. 
# The compression ratio is the ratio of the length of the compressed text to the length of the original text.
def compression_ratio(text) -> float: 
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

# This function formats timestamps into a human-readable string, which is useful for displaying the start and end times of transcribed segments.
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def write_txt(transcript: Iterator[dict], file: TextIO):# This function writes a transcript to a file in plain text format.
    for segment in transcript:
        print(segment['text'].strip(), file=file, flush=True)


def write_vtt(transcript: Iterator[dict], file: TextIO):# This function writes a transcript to a file in WebVTT format.
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def write_srt(transcript: Iterator[dict], file: TextIO):
    """
    Write a transcript to a file in SRT format.

    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt

        result = transcribe(model, audio_path, temperature=temperature, **args)

        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

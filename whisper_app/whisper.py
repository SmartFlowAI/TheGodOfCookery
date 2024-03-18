import whisper


def run_whisper(
        model_scale: str,
        device: str,
        audio_path: str):
    """
    使用Whisper模型对音频进行转录。

    Args:
        model_scale (str): 模型大小，可选值为 "tiny","base","small","medium", "large"。
        device (str): 设备类型，可选值为 "cpu","cuda"。
        audio_path (str): 音频文件路径。

    Returns:
        result (str): 转录结果。

    """
    assert model_scale in ["tiny", "base", "small", "medium",
                           "large"], "model must be one of tiny, base, small, medium or large"
    assert device in ["cpu", "cuda"], "device must be one of cpu or cuda"
    model = whisper.load_model(model_scale, device=device)
    result = model.transcribe(audio_path)
    return result['text']

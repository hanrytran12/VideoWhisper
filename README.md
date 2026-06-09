# VideoWhisper 🎥🗣️

Pipeline phân tích video tự động với AI: chuyển đổi video thành transcript có context và speaker detection.

## ✨ Features

- 🎤 **Transcription** - Whisper (faster-whisper) với word-level timestamps
- 🎭 **Speaker Detection** - Active Speaker Detection (ASD) qua MediaPipe FaceMesh
- 🔗 **AV Matching** - Ghép audio với face tracks
- 📢 **Speech Classification** - onscreen/offscreen/voiceover
- 🎬 **Frame Extraction** - Frames đại diện cho mỗi segment
- 🤖 **VLM Context** - Mô tả ngữ cảnh hình ảnh (Ollama + MiniCPM-V)
- 🎵 **Audio Separation** (optional) - Tách vocals bằng Demucs

## 🚀 Quick Start

### 1. Setup (lần đầu)
```cmd
setup.bat
```

### 2. Cài thêm FFmpeg và Ollama
- **FFmpeg**: https://www.gyan.dev/ffmpeg/builds/ (hoặc `winget install ffmpeg`)
- **Ollama**: https://ollama.com/download

### 3. Start Ollama và pull VLM model
```cmd
ollama serve
ollama pull minicpm-v
```

### 4. Kiểm tra setup
```cmd
python check_setup.py
```

### 5. Chạy!
```cmd
python run.py sample.mp4
```

## 📊 Output

Kết quả trong `output/<video_name>/final_output.json`:
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "こんにちは、皆さん",
      "speaker_id": "face_00",
      "speech_type": "onscreen",
      "face_boxes": [[100, 50, 200, 150]],
      "is_active_speaker": true,
      "context_description": "一人の男性、30歳くらい、白いシャツを着て...",
      "frame_paths": ["frames/frame_00002600.jpg"]
    }
  ],
  "speakers": [
    {
      "id": "face_00",
      "appearance_timestamps": [[0.0, 45.2], [50.0, 85.3]]
    }
  ]
}
```

## 🎯 Pipeline Stages

1. **Separate** (optional) - Demucs tách vocals (CPU, ~1-2 phút)
2. **Transcribe** - Whisper chuyển giọng nói → text (GPU, ~30-60s)
3. **ASD** - Active Speaker Detection qua face + lip movement (CPU, ~15-30s)
4. **AV Match** - Ghép audio với face tracks (CPU, <1s)
5. **Offscreen** - Phân loại speech_type (CPU, <1s)
6. **Frames** - Trích xuất frame đại diện (CPU, ~5s)
7. **VLM Context** - Ollama mô tả ngữ cảnh (GPU, ~2-5 phút)
8. **Assemble** - Gộp tất cả thành output cuối (CPU, <1s)

## ⚙️ Configuration

Chỉnh `config.yaml` để tùy chỉnh:
- Whisper model size: `small` | `medium` | `large-v3-turbo`
- VLM model: `minicpm-v` | `llava`
- GPU settings cho RTX 3050 4GB

## 📖 Chi tiết

Xem `SETUP.md` để hiểu rõ hơn về setup và troubleshooting.

## 🛠️ Requirements

- Python 3.9-3.11
- NVIDIA GPU với CUDA 12.1
- FFmpeg
- Ollama

## 📝 Usage

```cmd
# Chạy full pipeline (không separate)
python run.py video.mp4

# Với audio separation (chất lượng cao hơn, chậm hơn)
python run.py video.mp4 --stages separate,transcribe,asd,av_match,offscreen,frames,vlm_context,assemble

# Chỉ transcribe + ASD
python run.py video.mp4 --stages transcribe,asd

# Chạy lại (bỏ qua cache)
python run.py video.mp4 --force

# Test với script tự động
python test_full_pipeline.py video.mp4
```

## 🔧 Tối ưu cho GPU yếu

Nếu gặp OOM (Out of Memory), giảm trong `config.yaml`:
```yaml
transcribe:
  compute_type: int8
  batch_size: 4

vlm:
  num_gpu: 10
```

---

Made with ❤️ for Vietnamese content creators

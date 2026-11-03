"""Configuração central do ClipRush Gamer AI — valores para Ryzen 5 multi-core/16 GB."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRS = {
    "uploads": ROOT / "uploads", "output": ROOT / "output", "temp": ROOT / "temp",
    "fonts": ROOT / "fonts", "models": ROOT / "models", "logs": ROOT / "logs",
    "signatures": ROOT / "assinaturas", "library": ROOT / "biblioteca",
    "data": ROOT / "data",
    "music": ROOT / "musicas", "sfx": ROOT / "sfx", "narration": ROOT / "narracao",
    "cache": ROOT / "temp" / "cache", "sessions": ROOT / "output" / "sessoes",
}

# Processamento térmico. Não usa aceleração GPU/AMF/OpenCL/DirectML.
PERFORMANCE = {
    "threads": 2,
    "whisper_cpu_threads": 4,
    "whisper_num_workers": 1,
    "priority_below_normal": True,
    "economy_mode": False,
    "economy_threads": 1,
    "economy_pause_seconds": 4.0,
    "ffmpeg_preset": "veryfast",
    "draft_preset": "superfast",
    "one_job_at_a_time": True,
}

ANALYSIS = {
    "sample_rate": 16000,
    "audio_block_seconds": 1.0,
    "highpass_hz": 100,
    "adaptive_threshold_mad": 2.7,
    "peak_distance_seconds": 1.5,
    "candidate_limit": 300,
    "quick_visual_candidates": 80,
    "visual_fps": 5,
    "visual_width": 320,
    "scene_threshold": 0.30,
    "combo_window_seconds": 15.0,
    "clip_min_seconds": 20,
    "clip_max_seconds": 90,
    "clip_target_count": 10,
    "min_clip_gap_seconds": 75,
    "max_same_event_type": 3,
    "start_lead_seconds": 8,
    "end_tail_seconds": 8,
    "debug_csv_name": "calibracao.csv",
    "cache_schema_version": 6,
    "specific_event_min_confidence": 0.82,
    "input_speech_model": "tiny",
    "speech_language": "pt",
}

SCENE_MODE = "manter"  # manter | priorizar-gameplay | ignorar
SCENES = {
    "max_share": 0.30,
    "hud_threshold": 0.33,
    "camera_cut_diff_threshold": 0.28,
    "camera_cut_min_count": 2,
    "speech_min_rms": 0.004,
    "speech_voiced_threshold": 0.45,
    "speech_min_seconds": 3.0,
    "speech_min_ratio": 0.50,
    "speech_window_seconds": 9.0,
    "speech_vad_enabled": True,
    "vad_chunk_seconds": 60,
    "vad_threshold": 0.45,
    "vad_min_speech_duration_ms": 250,
    "vad_min_silence_duration_ms": 500,
    "vad_speech_pad_ms": 100,
    "probe_interval_seconds": 120.0,
    "probe_limit": 30,
    "gameplay_priority_bonus": 0.08,
    "score_weights": {"speech": 0.50, "music": 0.25, "tension": 0.25},
}

# IA local. Amostragem limitada para manter inferência visual sequencial em CPU.
AI = {
    "vision_model": "llava:7b",
    "text_model": "llama3.2:3b",
    "vision_max_candidates": 18,
    "vision_batch_size": 3,
    "vision_timeout_seconds": 90,
    "vision_frame_width": 320,
    "vision_frame_height": 180,
    "vision_score_weight": 0.65,
    "audio_score_weight": 0.25,
    "local_signal_weight": 0.10,
}

EVENT_WEIGHTS = {
    "escopeta": 1.35, "headshot": 1.30, "parry_faca": 1.35,
    "motosserra": 1.38, "explosao": 1.32, "boss_combate": 1.25,
    "susto_grito": 1.22, "pistola_rifle": 1.10, "combo": 1.35,
    "impacto": 1.0, "morno": 0.55, "cena": 0.85,
}
SCORE_WEIGHTS = {"audio": 0.58, "scene": 0.22, "movement": 0.12, "event_type": 0.08}

RENDER = {
    "width": 1080, "height": 1920, "fps": 30, "crf": 23,
    "audio_bitrate": "192k", "blur_small_width": 270, "blur_small_height": 480,
    "blur_sigma": 15, "background_brightness": -0.04, "background_saturation": 0.92,
    "foreground_height": 608, "hook_seconds": 1.8, "hook_before_peak": 0.5,
    "intro_fade_seconds": 0.20, "zoom_factor": 1.15, "zoom_duration": 0.75,
    "zoom_min_interval": 1.5, "zoom_focus_x": 0.5, "zoom_focus_y": 0.5,
    "safe_title_y_min": 150, "safe_title_y_max": 400, "safe_caption_y_min": 1150,
    "safe_caption_y_max": 1350, "safe_x_margin": 120, "safe_max_width": 840,
    "caption_font_size": 66, "title_font_size": 68, "outline_px": 5,
    "dark_brightness_threshold": 0.16, "max_brightness_correction": 0.18,
    "loudnorm_i": -14.0, "loudnorm_tp": -1.5, "loudnorm_lra": 11.0,
    "audio_fade_seconds": 0.25, "music_volume_db": -18,
    "music_enabled": False, "narration_enabled": False, "sfx_enabled": False,
    "reactions_enabled": True, "reaction_max_per_clip": 3,
    "watermark_enabled": False, "watermark_text": "",
    "loop_end_enabled": False, "cover_format": "jpg",
    "draft_width": 540, "draft_height": 960,
}

PLATFORMS = {
    "shorts": {"label": "YouTube Shorts", "min_seconds": 1, "max_seconds": 59,
               "title_max_chars": 100, "hashtags": ["#shorts", "#gaming", "#gameplay"]},
    "reels": {"label": "Instagram Reels", "min_seconds": 1, "max_seconds": 59,
              "title_max_chars": 100, "hashtags": ["#reels", "#gaming", "#gameplay"]},
    "tiktok": {"label": "TikTok monetizável", "min_seconds": 61, "max_seconds": 90,
               "title_max_chars": 150, "hashtags": ["#fyp", "#gaming", "#gameplay"]},
}

LOUDNESS = {"integrated_lufs": -14.0, "true_peak_db": -1.5, "lra": 11.0}
MONETIZATION_GOALS = {
    "youtube": {"views_30_days": 0, "subscribers": 0, "posts": 0},
    "tiktok": {"views_30_days": 0, "followers": 0, "posts": 0},
}

REQUIRED_FFMPEG_FILTERS = (
    "adelay", "afade", "alimiter", "amix", "aresample", "ass", "atrim", "boxblur",
    "concat", "crop", "eq", "fade", "fps", "loudnorm", "overlay", "pad",
    "scale", "setsar", "setpts", "sidechaincompress", "tpad",
)
WHISPER_MODEL = "tiny"
WHISPER_COMPUTE_TYPE = "int8"
FONT_FILE = DIRS["fonts"] / "DejaVuSans-Bold.ttf"

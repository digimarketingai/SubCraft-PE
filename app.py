# ==============================================================================
# 🎬 SubCraft-PE: AI Video Subtitle & Bilingual Post-Editing Workbench
#    GitHub: https://github.com/digimarketingai/SubCraft-PE
#
# Key Highlights in This Edition:
#   1. ⚡ `deepdml/faster-distil-whisper-large-v3.5` (756M Params) is the DEFAULT
#      ASR model, upgraded with Anti-Degradation Sequential 15s Windowing
#      (`chunk_length=15`), Audio Peak Normalization, Temperature Fallback
#      `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`, `hallucination_silence_threshold=1.5`,
#      and `repetition_penalty=1.1` so accuracy stays sharp from start to end!
#   2. 🧠 Whole-Text Context-Aware Translation (`translate_segments_full_context`):
#      - Combines subtitle cues into coherent multi-sentence passages, translates
#        with full context, and splits/aligns back to each exact subtitle cue!
#   3. 🔥 Guaranteed CJK + Multilingual Hard-Burn Video Output (`burn_subtitles_to_mp4`):
#      - Automatically verifies/installs/downloads a true CJK font (`Noto Sans CJK` /
#        `WenQuanYi Zen Hei`) and passes `fontsdir` to FFmpeg `ass`/`subtitles`
#        so Translated Chinese/CJK & English text NEVER disappears!
#      - Uses explicit ASS override tags (`\fs`, `\c`, `\b`) with `BorderStyle=1`
#        high-contrast outline + shadow so `libass` never clips multi-line boxes.
#      - 100% aligned with user's selected Mode (`bilingual_target_top`,
#        `bilingual_source_top`, `target_only`, `source_only`) and Font Size!
#   4. 🎓 Built-in Classroom Sample Clips (`samples/` directory + Auto-Generator).
#   5. 🧭 Comprehensive C-E & E-C Audiovisual Post-Editing Skills (Tab 2 + PDF).
# ==============================================================================

import concurrent.futures
import copy
import ctypes
import difflib
import gc
import glob
import html
import importlib.util
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import time
import urllib.parse
from typing import Any, Dict, Generator, List, Optional, Tuple


def ensure_environment() -> None:
    """Installs required packages, FFmpeg, and CJK fonts (even when FFmpeg is already installed!) and preloads CUDA libs."""
    pip_packages = {
        "gradio": "gradio",
        "faster_whisper": "faster-whisper",
        "opencc": "opencc-python-reimplemented",
        "requests": "requests",
        "numpy": "numpy",
        "reportlab": "reportlab",
        "deep_translator": "deep-translator",
        "translators": "translators",
        "gtts": "gTTS",
        "nvidia.cublas": "nvidia-cublas-cu12",
        "nvidia.cudnn": "nvidia-cudnn-cu12",
    }
    missing_pip = [
        pkg for mod, pkg in pip_packages.items()
        if importlib.util.find_spec(mod) is None
    ]
    if missing_pip:
        print(f"📦 Installing missing packages: {', '.join(missing_pip)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing_pip]
        )

    # IMPORTANT: Check for FFmpeg AND CJK fonts independently!
    # Previously, if ffmpeg was already pre-installed on Colab, `fonts-noto-cjk` was skipped,
    # causing Chinese/CJK translated subtitles to be invisible in FFmpeg hard-burn!
    need_ffmpeg = shutil.which("ffmpeg") is None
    cjk_font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    has_system_cjk_font = any(os.path.exists(p) for p in cjk_font_candidates)

    if (need_ffmpeg or not has_system_cjk_font) and shutil.which("apt-get") is not None:
        print("🛠️ Ensuring FFmpeg, libass, and CJK fonts (fonts-noto-cjk / fonts-wqy-zenhei) are installed...")
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        pkgs_to_install = ["fonts-noto-cjk", "fonts-wqy-zenhei", "fontconfig"]
        if need_ffmpeg:
            pkgs_to_install.insert(0, "ffmpeg")
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", *pkgs_to_install],
            check=False,
        )
        subprocess.run(["fc-cache", "-f"], check=False)

    # Preload libcublas.so.12 and libcudnn*.so.9 so ctranslate2 finds them on Colab T4 GPU
    lib_dirs: List[str] = []
    search_bases = [
        *site.getsitepackages(),
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    for base in search_bases:
        for pattern in ("**/nvidia/*/lib", "**/lib64", ""):
            target_glob = os.path.join(base, pattern) if pattern else base
            for d in glob.glob(target_glob, recursive=True):
                if os.path.isdir(d) and d not in lib_dirs:
                    lib_dirs.append(d)

    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = (
            ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        for d in lib_dirs:
            for pattern in (
                "libcublas.so*",
                "libcublasLt.so*",
                "libcudnn*.so*",
            ):
                for so_file in sorted(glob.glob(os.path.join(d, pattern))):
                    try:
                        ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    # Patch PyAV av.open in case PyAV >= 14 removed `metadata_errors` keyword arg
    try:
        import av
        import inspect

        if "metadata_errors" not in inspect.signature(av.open).parameters:
            _orig_av_open = av.open

            def _patched_av_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return _orig_av_open(*args, **kwargs)

            av.open = _patched_av_open
    except Exception:
        pass


ensure_environment()

import gradio as gr
import numpy as np
import requests
from faster_whisper import BatchedInferencePipeline, WhisperModel
from opencc import OpenCC

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ==============================================================================
# Global State, Directories, Fonts, Sample Clips & Verified T4 GPU ASR / MT Models
# ==============================================================================

APP_ROOT_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
SAMPLES_DIR = os.path.join(APP_ROOT_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

SHARED_MEDIA_DIR = os.path.join(
    tempfile.gettempdir(), "gradio_bilingual_media_cache"
)
os.makedirs(SHARED_MEDIA_DIR, exist_ok=True)

FONTS_CACHE_DIR = os.path.join(SHARED_MEDIA_DIR, "subcraft_fonts")
os.makedirs(FONTS_CACHE_DIR, exist_ok=True)

AUTO_BACKUP_PATH = os.path.join(
    SHARED_MEDIA_DIR, "latest_student_autosave_project.json"
)

WHISPER_MODEL: Optional[WhisperModel] = None
WHISPER_BATCHED: Optional[BatchedInferencePipeline] = None
WHISPER_KEY: Optional[Tuple[str, str]] = None
FORCE_CPU: bool = False
NVENC_AVAILABLE: Optional[bool] = None
PDF_CJK_FONT_NAME: Optional[str] = None
SUBTITLE_FONT_INFO: Optional[Tuple[str, str]] = None

OPENCC_S2TWP = OpenCC("s2twp")
OPENCC_T2S = OpenCC("t2s")

DEFAULT_ASR_MODEL = "deepdml/faster-distil-whisper-large-v3.5"

ASR_MODEL_CHOICES = [
    (
        "⚡ deepdml/faster-distil-whisper-large-v3.5 (DEFAULT • SOTA Distilled English v3.5 • 756M Params • Anti-Drift 15s Window) / 最新蒸餾英文霸主 v3.5（預設首選）",
        "deepdml/faster-distil-whisper-large-v3.5",
    ),
    (
        "🏆 large-v3 (Maximum Studio Multilingual Accuracy • 1.55B Params • 32 Layers) / 最高精準度多語模型",
        "large-v3",
    ),
    (
        "🔥 deepdml/faster-whisper-large-v3-turbo-ct2 (Fast Multilingual Turbo SOTA • 809M Params • 6x Faster on T4) / 高速多語霸主 Turbo",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
    ),
    (
        "🈯 flateon/Belle-whisper-large-v3-turbo-zh-ct2 (Chinese & ZH-EN Code-Switching Punctuation SOTA) / 中文與中英夾雜標點強化模型",
        "flateon/Belle-whisper-large-v3-turbo-zh-ct2",
    ),
    (
        "⚡ Systran/faster-distil-whisper-large-v3 (Ultra-Fast English Distilled v3 • 6x Faster) / 高速英文蒸餾模型 v3",
        "Systran/faster-distil-whisper-large-v3",
    ),
    (
        "🎯 large-v2 (Pro Subtitler Favorite for ZH/EN Code-Switching • 1.55B Params) / 專業字幕組最愛穩定版 v2",
        "large-v2",
    ),
    (
        "⚡ distil-medium.en (Lightweight English Distilled) / 輕量英文模型",
        "distil-medium.en",
    ),
    ("⚖️ medium (Balanced Multilingual) / 中型多語模型", "medium"),
    ("🪶 small (Lightweight & Fast) / 小型快速模型", "small"),
    ("🪶 base (Very Lightweight) / 基礎模型", "base"),
]

MODEL_FALLBACK_MIRRORS: Dict[str, List[str]] = {
    "deepdml/faster-distil-whisper-large-v3.5": [
        "deepdml/faster-distil-whisper-large-v3.5",
        "distil-whisper/distil-large-v3.5-ct2",
        "Systran/faster-distil-whisper-large-v3",
    ],
    "large-v3": [
        "large-v3",
        "Systran/faster-whisper-large-v3",
    ],
    "deepdml/faster-whisper-large-v3-turbo-ct2": [
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3-turbo",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "Systran/faster-whisper-large-v3",
    ],
    "flateon/Belle-whisper-large-v3-turbo-zh-ct2": [
        "flateon/Belle-whisper-large-v3-turbo-zh-ct2",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3",
    ],
    "Systran/faster-distil-whisper-large-v3": [
        "Systran/faster-distil-whisper-large-v3",
        "distil-large-v3",
    ],
    "large-v2": [
        "large-v2",
        "Systran/faster-whisper-large-v2",
    ],
}

ENGLISH_ONLY_MODELS = {
    "deepdml/faster-distil-whisper-large-v3.5",
    "distil-whisper/distil-large-v3.5-ct2",
    "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3",
    "distil-medium.en",
}

DISTIL_SHORT_WINDOW_MODELS = {
    "deepdml/faster-distil-whisper-large-v3.5",
    "distil-whisper/distil-large-v3.5-ct2",
    "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3",
    "distil-medium.en",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "flateon/Belle-whisper-large-v3-turbo-zh-ct2",
}

MT_ENGINE_CHOICES = [
    (
        "🌐 Google Translate (Whole-Passage Context Neural GTX • Best Quality, No Key) / Google 全文語境神經翻譯（推薦）",
        "google_free",
    ),
    (
        "🈶 Alibaba Translate (Whole-Passage Context via `translators` • Great for C-E / E-C, No Key) / 阿里雲全文語境免費翻譯",
        "alibaba_free",
    ),
    (
        "🟦 Bing Microsoft Translator (Whole-Passage Context via `translators` • Natural Fluency, No Key) / 微軟 Bing 全文語境免費翻譯",
        "bing_free",
    ),
    (
        "📚 MyMemory Translated.net (Free REST API • Good for MTPE Error Comparison) / MyMemory 免費翻譯",
        "mymemory_free",
    ),
]

TRANSLATION_MODE_CHOICES = [
    (
        "🧠 Full-Passage Context Translation → Split Back to Cues (DEFAULT • Best Context & Coherence) / 全文語境連貫翻譯後自動切回對應字幕（預設推薦）",
        "full_context_passage",
    ),
    (
        "⚡ Line-by-Line Cue Translation (Independent Cues) / 逐行獨立翻譯",
        "line_by_line",
    ),
]

SOURCE_LANG_CHOICES = [
    ("Auto Detect / 自動偵測", "auto"),
    ("English / 英文", "en-US"),
    ("Traditional Chinese / 繁體中文", "zh-TW"),
    ("Simplified Chinese / 簡體中文", "zh-CN"),
    ("Japanese / 日文", "ja-JP"),
    ("Korean / 韓文", "ko-KR"),
    ("Spanish / 西班牙文", "es-ES"),
    ("French / 法文", "fr-FR"),
    ("German / 德文", "de-DE"),
]

TARGET_LANG_CHOICES = [
    ("Traditional Chinese / 繁體中文", "zh-TW"),
    ("Simplified Chinese / 簡體中文", "zh-CN"),
    ("English / 英文", "en-US"),
    ("Japanese / 日文", "ja-JP"),
    ("Korean / 韓文", "ko-KR"),
    ("Spanish / 西班牙文", "es-ES"),
    ("French / 法文", "fr-FR"),
    ("German / 德文", "de-DE"),
]

PRESENTATION_MODE_CHOICES = [
    (
        "Bilingual: Target Top + Source Bottom / 雙語：譯文在上＋原文在下",
        "bilingual_target_top",
    ),
    (
        "Bilingual: Source Top + Target Bottom / 雙語：原文在上＋譯文在下",
        "bilingual_source_top",
    ),
    ("Translated Only (Monolingual Target) / 僅顯示譯文（單語）", "target_only"),
    ("Original Only (Monolingual Source) / 僅顯示原文（單語）", "source_only"),
]

PRESENTATION_MODE_LABELS = {val: label for label, val in PRESENTATION_MODE_CHOICES}

WHISPER_ISO_MAP = {
    "auto": None,
    "en-US": "en",
    "zh-TW": "zh",
    "zh-CN": "zh",
    "ja-JP": "ja",
    "ko-KR": "ko",
    "es-ES": "es",
    "fr-FR": "fr",
    "de-DE": "de",
}

DETECTED_TO_LANG_CODE = {
    "en": "en-US",
    "zh": "zh-TW",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
}

GOOGLE_LANG_MAP = {
    "auto": "auto",
    "en-US": "en",
    "zh-TW": "zh-TW",
    "zh-CN": "zh-CN",
    "ja-JP": "ja",
    "ko-KR": "ko",
    "es-ES": "es",
    "fr-FR": "fr",
    "de-DE": "de",
}

TRANSLATORS_LIB_LANG_MAP = {
    "auto": "auto",
    "en-US": "en",
    "zh-TW": "zh-CHT",
    "zh-CN": "zh",
    "ja-JP": "ja",
    "ko-KR": "ko",
    "es-ES": "es",
    "fr-FR": "fr",
    "de-DE": "de",
}


# ==============================================================================
# Guaranteed CJK + Multilingual Font Resolver for FFmpeg / libass Hard-Burn
# ==============================================================================


def ensure_cjk_font_for_subtitles() -> Tuple[str, str]:
    """
    Guarantees that a real CJK + Latin font file exists in `FONTS_CACHE_DIR` and returns
    `(font_family_name, fonts_dir_path)` so FFmpeg's `ass`/`subtitles` filter NEVER
    drops or hides Chinese/Japanese/Korean translated text!
    """
    global SUBTITLE_FONT_INFO
    if SUBTITLE_FONT_INFO is not None:
        return SUBTITLE_FONT_INFO

    system_candidates = [
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "Noto Sans CJK TC",
        ),
        (
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "Noto Sans CJK TC",
        ),
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
            "Noto Sans CJK TC",
        ),
        (
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "WenQuanYi Zen Hei",
        ),
        (
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "WenQuanYi Micro Hei",
        ),
        (
            "/System/Library/Fonts/PingFang.ttc",
            "PingFang TC",
        ),
        (
            "C:/Windows/Fonts/msjh.ttc",
            "Microsoft JhengHei",
        ),
    ]

    for src_path, fam_name in system_candidates:
        if os.path.exists(src_path):
            dst_path = os.path.join(
                FONTS_CACHE_DIR, os.path.basename(src_path)
            )
            if not os.path.exists(dst_path):
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception:
                    pass
            SUBTITLE_FONT_INFO = (fam_name, FONTS_CACHE_DIR)
            return SUBTITLE_FONT_INFO

    # Fallback: If no system CJK font is installed (e.g., restricted container without apt),
    # download NotoSansTC-Bold.otf directly into FONTS_CACHE_DIR!
    downloaded_font = os.path.join(FONTS_CACHE_DIR, "NotoSansTC-Regular.otf")
    if not os.path.exists(downloaded_font) or os.path.getsize(downloaded_font) < 100_000:
        font_urls = [
            "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf",
            "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf",
        ]
        for url in font_urls:
            try:
                print(f"📥 Downloading CJK font for FFmpeg subtitle rendering from {url} ...")
                resp = requests.get(url, timeout=25)
                if resp.status_code == 200 and len(resp.content) > 500_000:
                    with open(downloaded_font, "wb") as f:
                        f.write(resp.content)
                    break
            except Exception as e:
                print(f"⚠️ CJK font download attempt failed: {e}")

    if os.path.exists(downloaded_font) and os.path.getsize(downloaded_font) > 100_000:
        SUBTITLE_FONT_INFO = ("Noto Sans CJK TC", FONTS_CACHE_DIR)
        return SUBTITLE_FONT_INFO

    SUBTITLE_FONT_INFO = ("sans-serif", FONTS_CACHE_DIR)
    return SUBTITLE_FONT_INFO


# ==============================================================================
# Built-in Classroom Practice Sample Clips Discovery & Auto-Bootstrap
# ==============================================================================


def ensure_builtin_sample_clips() -> List[Tuple[str, str]]:
    """
    Discovers all `.mp4`, `.mov`, `.mkv`, `.webm` files inside `samples/`.
    If `samples/` has no videos yet, automatically creates 2 classroom practice clips
    (1 E-C English speech clip & 1 C-E Chinese speech clip) so students can test immediately!
    """
    existing = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
        existing.extend(glob.glob(os.path.join(SAMPLES_DIR, ext)))

    if not existing:
        try:
            from gtts import gTTS

            en_mp4 = os.path.join(
                SAMPLES_DIR, "sample_01_EC_tech_keynote_practice.mp4"
            )
            zh_mp4 = os.path.join(
                SAMPLES_DIR, "sample_02_CE_cultural_interview_practice.mp4"
            )

            if not os.path.exists(en_mp4):
                en_script = (
                    "Welcome to our audiovisual translation workshop. "
                    "When a sentence is translated word by word without context, "
                    "the meaning is often lost by the machine. "
                    "However, by reading the whole passage first and polishing the subtitle, "
                    "we can make every line sound natural and concise for viewers."
                )
                en_mp3 = os.path.join(SHARED_MEDIA_DIR, "temp_en_sample.mp3")
                gTTS(text=en_script, lang="en").save(en_mp3)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=#0f172a:s=1280x720:r=24",
                        "-i",
                        en_mp3,
                        "-shortest",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        en_mp4,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

            if not os.path.exists(zh_mp4):
                zh_script = (
                    "歡迎來到影視字幕譯後編輯實作課堂。"
                    "機器翻譯在處理中文流水句與四字成語時，往往容易產生結構混亂或漏譯主詞的問題。"
                    "透過全篇語境分析與人工斷句潤飾，我們能讓英文字幕既精簡又道地。"
                )
                zh_mp3 = os.path.join(SHARED_MEDIA_DIR, "temp_zh_sample.mp3")
                gTTS(text=zh_script, lang="zh-tw").save(zh_mp3)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=#1e293b:s=1280x720:r=24",
                        "-i",
                        zh_mp3,
                        "-shortest",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        zh_mp4,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
        except Exception as e:
            print(f"⚠️ Note: Auto sample clip synthesis skipped ({e}).")

    discovered: List[Tuple[str, str]] = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
        for fpath in sorted(glob.glob(os.path.join(SAMPLES_DIR, ext))):
            fname = os.path.basename(fpath)
            if "sample_01_EC" in fname:
                label = f"🎓 E-C Practice Clip: Tech Keynote ({fname}) / 英譯中課堂練習片段"
            elif "sample_02_CE" in fname:
                label = f"🎓 C-E Practice Clip: Cultural Interview ({fname}) / 中譯英課堂練習片段"
            else:
                label = f"🎬 Classroom Sample: {fname}"
            discovered.append((label, fpath))
    return discovered


def has_nvenc_support() -> bool:
    """Checks whether NVIDIA GPU hardware video encoding (h264_nvenc) works."""
    global NVENC_AVAILABLE
    if NVENC_AVAILABLE is not None:
        return NVENC_AVAILABLE
    try:
        test_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=256x256:d=0.1",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        res = subprocess.run(
            test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8
        )
        NVENC_AVAILABLE = res.returncode == 0
    except Exception:
        NVENC_AVAILABLE = False
    return NVENC_AVAILABLE


def get_video_codec_name(video_path: str) -> str:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        return (
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            .strip()
            .lower()
        )
    except Exception:
        return ""


def make_gradio_file_url(file_path: str) -> str:
    """Returns a fast HTTP streaming URL served by Gradio (no Base64 lag!)."""
    abs_path = os.path.abspath(file_path)
    mtime = (
        int(os.path.getmtime(abs_path))
        if os.path.exists(abs_path)
        else int(time.time())
    )
    quoted = urllib.parse.quote(abs_path, safe="/:")
    return f"/gradio_api/file={quoted}?v={mtime}"


# ==============================================================================
# Dynamic Loading Banner (Animated Spinner + Progress Bar + Live Step View)
# ==============================================================================


def build_dynamic_status_banner(
    state: str,
    title: str,
    subtitle: str,
    pct: int = 0,
    elapsed_sec: float = 0.0,
) -> str:
    pct = max(0, min(100, int(pct)))
    if state == "running":
        border_col = "#0284c7"
        bg_grad = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
        icon_html = """
        <div style="position:relative;width:36px;height:36px;flex-shrink:0;">
            <div style="width:36px;height:36px;border:4px solid rgba(56,189,248,0.2);border-top-color:#38bdf8;border-right-color:#fde047;border-radius:50%;animation:pe-spin 0.75s linear infinite;"></div>
            <div style="position:absolute;inset:8px;background:#38bdf8;border-radius:50%;animation:pe-pulse 1.2s ease-in-out infinite;"></div>
        </div>
        """
        badge_bg = "#0284c7"
        badge_txt = f"⚡ RUNNING • {elapsed_sec:.1f}s ({pct}%)"
        bar_bg = "linear-gradient(90deg, #0284c7, #38bdf8, #fde047, #38bdf8)"
    elif state == "done":
        border_col = "#10b981"
        bg_grad = "linear-gradient(135deg, #064e3b 0%, #0f172a 100%)"
        icon_html = """
        <div style="width:36px;height:36px;border-radius:50%;background:#10b981;color:#022c22;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:900;flex-shrink:0;">✓</div>
        """
        badge_bg = "#059669"
        badge_txt = f"✅ COMPLETED IN {elapsed_sec:.1f}s"
        bar_bg = "linear-gradient(90deg, #10b981, #34d399)"
    else:
        border_col = "#334155"
        bg_grad = "#0f172a"
        icon_html = """
        <div style="width:36px;height:36px;border-radius:50%;background:#1e293b;border:2px solid #475569;color:#38bdf8;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🎬</div>
        """
        badge_bg = "#334155"
        badge_txt = "IDLE / READY"
        bar_bg = "#38bdf8"

    return f"""
    <style>
        @keyframes pe-spin {{ to {{ transform: rotate(360deg); }} }}
        @keyframes pe-pulse {{ 0%, 100% {{ transform: scale(0.65); opacity: 0.5; }} 50% {{ transform: scale(1.05); opacity: 1; }} }}
        @keyframes pe-shimmer {{ 0% {{ background-position: 0% 50%; }} 100% {{ background-position: 200% 50%; }} }}
    </style>
    <div style="background:{bg_grad};border:1.5px solid {border_col};border-radius:12px;padding:14px 18px;color:#f8fafc;font-family:system-ui,-apple-system,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,0.25);">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:14px;">
                {icon_html}
                <div>
                    <div style="font-size:15px;font-weight:700;color:#f8fafc;">{html.escape(title)}</div>
                    <div style="font-size:13px;color:#cbd5e1;margin-top:2px;">{html.escape(subtitle)}</div>
                </div>
            </div>
            <span style="background:{badge_bg};color:#fff;font-weight:700;font-size:12px;padding:5px 12px;border-radius:999px;letter-spacing:0.4px;">
                {badge_txt}
            </span>
        </div>
        <div style="margin-top:10px;width:100%;height:7px;background:#020617;border-radius:999px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{bar_bg};background-size:200% 100%;animation:pe-shimmer 1.5s linear infinite;transition:width 0.25s ease;"></div>
        </div>
    </div>
    """


# ==============================================================================
# Helper Functions: Time, Multi-Line Normalization, SRT/VTT/ASS, Project Backup
# ==============================================================================


def normalize_multiline_text(text: Any) -> str:
    """Preserves intentional line breaks (`Enter` / `\\n`) for dual-speaker subtitles."""
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    lines = [line.strip() for line in s.split("\n") if line.strip()]
    return "\n".join(lines)


def seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    return seconds_to_srt_time(seconds).replace(",", ".")


def seconds_to_ass_time(seconds: float) -> str:
    """Formats seconds into ASS timestamp `H:MM:SS.cc` (centiseconds)."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{centis:02d}"


def parse_flexible_time(val: Any) -> float:
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    s = str(val).strip()
    if not s:
        return 0.0
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    s = s.replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return max(
                0.0,
                int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]),
            )
        if len(parts) == 2:
            return max(0.0, int(parts[0]) * 60 + float(parts[1]))
    except ValueError:
        return 0.0
    return 0.0


def compose_presentation_lines(
    src_text: str, tgt_text: str, mode: str
) -> List[str]:
    src_lines = [
        line.strip()
        for line in normalize_multiline_text(src_text).split("\n")
        if line.strip()
    ]
    tgt_lines = [
        line.strip()
        for line in normalize_multiline_text(tgt_text).split("\n")
        if line.strip()
    ]
    if mode == "bilingual_target_top":
        return tgt_lines + src_lines
    if mode == "bilingual_source_top":
        return src_lines + tgt_lines
    if mode == "source_only":
        return src_lines
    # For target_only, fallback to src_lines only if tgt_lines is empty
    return tgt_lines if tgt_lines else src_lines


def write_srt_file(
    segments: List[Dict[str, Any]],
    path: str,
    mode: str = "target_only",
) -> str:
    lines: List[str] = []
    idx = 1
    for seg in segments:
        start_s = parse_flexible_time(seg.get("start", 0.0))
        end_s = max(
            start_s + 0.15, parse_flexible_time(seg.get("end", start_s + 0.8))
        )
        sub_lines = compose_presentation_lines(
            seg.get("source", ""), seg.get("target", ""), mode
        )
        if not sub_lines:
            continue
        lines.append(str(idx))
        lines.append(
            f"{seconds_to_srt_time(start_s)} --> {seconds_to_srt_time(end_s)}"
        )
        lines.extend(sub_lines)
        lines.append("")
        idx += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_vtt_file(
    segments: List[Dict[str, Any]],
    path: str,
    mode: str = "bilingual_target_top",
) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for idx, seg in enumerate(segments, start=1):
        start_s = parse_flexible_time(seg.get("start", 0.0))
        end_s = max(
            start_s + 0.15, parse_flexible_time(seg.get("end", start_s + 0.8))
        )
        sub_lines = compose_presentation_lines(
            seg.get("source", ""), seg.get("target", ""), mode
        )
        if not sub_lines:
            continue
        lines.append(str(idx))
        lines.append(
            f"{seconds_to_vtt_time(start_s)} --> {seconds_to_vtt_time(end_s)}"
        )
        lines.extend(sub_lines)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def escape_ass_text(text: str) -> str:
    """Escapes special characters for ASS subtitle dialogue lines and converts `\n` to `\\N`."""
    clean = normalize_multiline_text(text)
    if not clean:
        return ""
    clean = clean.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]
    return "\\N".join(lines)


def write_ass_file(
    segments: List[Dict[str, Any]],
    path: str,
    mode: str = "bilingual_target_top",
    font_size: int = 20,
    font_family: str = "Noto Sans CJK TC",
) -> str:
    """
    Generates an Advanced SubStation Alpha (.ASS) subtitle file for FFmpeg `libass`.
    Why this never hides Translated CJK text:
      1. Uses the verified `font_family` from `ensure_cjk_font_for_subtitles()`.
      2. Uses `BorderStyle=1` (high-contrast thick dark outline + drop shadow) instead of
         `BorderStyle=3`, avoiding the `libass` bug where multi-line opaque boxes clip or
         hide the top/bottom line.
      3. Uses explicit inline ASS override tags (`{\\fs...\\c&H...&\\b1}`) instead of `{\\r...}`
         style resets, guaranteeing both Target (Bold Yellow #FDE047) and Source (Crisp White #F8FAFC)
         render reliably in all 4 modes!
    """
    main_fs = max(20, int(round(int(font_size or 20) * 1.75)))
    sub_fs = max(15, int(round(main_fs * 0.76)))

    # ASS color format in inline override tags is &HBBGGRR& (Blue, Green, Red)
    # Yellow (#FDE047 -> R=FD, G=E0, B=47) => &H47E0FD&
    # White  (#F8FAFC -> R=F8, G=FA, B=FC) => &HFCFAF8&
    target_tag = f"{{\\fn{font_family}\\fs{main_fs}\\b1\\1c&H47E0FD&\\3c&H170602&\\bord3.2\\shad1.5}}"
    source_sub_tag = f"{{\\fn{font_family}\\fs{sub_fs}\\b0\\1c&HFCFAF8&\\3c&H170602&\\bord2.6\\shad1.2}}"
    source_main_tag = f"{{\\fn{font_family}\\fs{main_fs}\\b1\\1c&HFCFAF8&\\3c&H170602&\\bord3.2\\shad1.5}}"

    ass_header = f"""[Script Info]
Title: SubCraft-PE Hard-Burn Subtitles ({mode})
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{main_fs},&H0047E0FD,&H0047E0FD,&H00170602,&H80000000,-1,0,0,0,100,100,0,0,1,3.2,1.5,2,40,40,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: List[str] = []
    for seg in segments:
        start_s = parse_flexible_time(seg.get("start", 0.0))
        end_s = max(
            start_s + 0.15, parse_flexible_time(seg.get("end", start_s + 0.8))
        )
        start_ass = seconds_to_ass_time(start_s)
        end_ass = seconds_to_ass_time(end_s)

        src_ass = escape_ass_text(seg.get("source", ""))
        tgt_ass = escape_ass_text(seg.get("target", ""))

        if mode == "bilingual_target_top":
            if tgt_ass and src_ass:
                line_text = f"{target_tag}{tgt_ass}\\N{source_sub_tag}{src_ass}"
            elif tgt_ass:
                line_text = f"{target_tag}{tgt_ass}"
            elif src_ass:
                line_text = f"{source_main_tag}{src_ass}"
            else:
                continue

        elif mode == "bilingual_source_top":
            if src_ass and tgt_ass:
                line_text = f"{source_sub_tag}{src_ass}\\N{target_tag}{tgt_ass}"
            elif src_ass:
                line_text = f"{source_main_tag}{src_ass}"
            elif tgt_ass:
                line_text = f"{target_tag}{tgt_ass}"
            else:
                continue

        elif mode == "source_only":
             chosen = src_ass if src_ass else tgt_ass
             if not chosen:
                 continue
             line_text = f"{source_main_tag}{chosen}"

        else:  # target_only (Monolingual Translated Only)
            chosen = tgt_ass if tgt_ass else src_ass
            if not chosen:
                continue
            line_text = f"{target_tag}{chosen}"

        events.append(
            f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{line_text}"
        )

    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(ass_header + "\n".join(events) + "\n")
    return path


def save_project_checkpoint_json(
    current_segments: List[Dict[str, Any]],
    baseline_segments: List[Dict[str, Any]],
    presentation_mode: str,
    font_size: int,
    out_dir: str,
) -> str:
    payload = {
        "version": "3.2",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "presentation_mode": presentation_mode,
        "font_size": int(font_size or 20),
        "segments": current_segments,
        "baseline_segments": baseline_segments,
    }
    checkpoint_path = os.path.join(
        out_dir, "student_post_editing_project_backup.json"
    )
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    try:
        shutil.copy2(checkpoint_path, AUTO_BACKUP_PATH)
    except Exception:
        pass
    return checkpoint_path


def get_video_duration(video_path: str) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True
        ).strip()
        return round(max(0.0, float(out)), 2)
    except Exception:
        return 0.0


def prepare_h264_clip(
    input_video: str, start_sec: float, end_sec: float, log_fn=None
) -> str:
    stamp = int(time.time() * 1000)
    out_path = os.path.join(SHARED_MEDIA_DIR, f"prepared_clip_{stamp}.mp4")
    total_dur = get_video_duration(input_video)
    codec = get_video_codec_name(input_video)

    need_trim = (start_sec > 0.05) or (
        end_sec > start_sec
        and total_dur > 0
        and abs(end_sec - total_dur) > 0.25
    )

    if (
        not need_trim
        and codec == "h264"
        and input_video.lower().endswith(".mp4")
    ):
        if log_fn:
            log_fn(
                "⚡ Instant Stream Copy (Full H.264 MP4 detected — 0s re-encode overhead!)"
            )
        shutil.copy2(input_video, out_path)
        return out_path

    if need_trim and codec == "h264":
        if log_fn:
            log_fn("⚡ Fast H.264 stream-copy trimming (-c copy)...")
        cmd_copy = ["ffmpeg", "-y"]
        if start_sec > 0.01:
            cmd_copy.extend(["-ss", f"{start_sec:.2f}"])
        if end_sec > start_sec:
            cmd_copy.extend(["-to", f"{end_sec:.2f}"])
        cmd_copy.extend(
            [
                "-i",
                input_video,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )
        res_copy = subprocess.run(
            cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if (
            res_copy.returncode == 0
            and os.path.exists(out_path)
            and os.path.getsize(out_path) > 1024
        ):
            return out_path

    use_nvenc = has_nvenc_support()
    if log_fn:
        enc_label = (
            "NVIDIA GPU (h264_nvenc)"
            if use_nvenc
            else "CPU (libx264 ultrafast)"
        )
        log_fn(f"Video encoder selected: {enc_label}")

    cmd = ["ffmpeg", "-y"]
    if start_sec > 0.01:
        cmd.extend(["-ss", f"{start_sec:.2f}"])
    if end_sec > start_sec and need_trim:
        cmd.extend(["-to", f"{end_sec:.2f}"])

    v_codec = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
        if use_nvenc
        else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
    )
    cmd.extend(
        [
            "-i",
            input_video,
            *v_codec,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            out_path,
        ]
    )
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 and use_nvenc:
        cmd_cpu = ["ffmpeg", "-y"]
        if start_sec > 0.01:
            cmd_cpu.extend(["-ss", f"{start_sec:.2f}"])
        if end_sec > start_sec and need_trim:
            cmd_cpu.extend(["-to", f"{end_sec:.2f}"])
        cmd_cpu.extend(
            [
                "-i",
                input_video,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )
        subprocess.run(
            cmd_cpu, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    return out_path


def decode_audio_ffmpeg_np(
    media_path: str, sample_rate: int = 16000
) -> np.ndarray:
    """Decodes 16kHz mono PCM audio and applies Peak Normalization + High-Pass Filter so volume never drops later in the video."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        media_path,
        "-af",
        "highpass=f=70,lowpass=f=7800",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((sample_rate,), dtype=np.float32)
    audio = (
        np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32)
        / 32768.0
    )
    if audio.size > 0:
        peak = float(np.max(np.abs(audio)))
        if 1e-4 < peak < 0.85:
            audio = np.clip(audio * (0.92 / peak), -1.0, 1.0)
    return audio


# ==============================================================================
# T4-Maximized ASR (`distil-large-v3.5` Default + Anti-Drift 15s Sequential Mode)
# ==============================================================================


def release_whisper_vram(log_fn=None) -> None:
    """Frees GPU/CPU memory when switching ASR models so T4 VRAM stays clean."""
    global WHISPER_MODEL, WHISPER_BATCHED, WHISPER_KEY
    if WHISPER_MODEL is not None or WHISPER_BATCHED is not None:
        old_name = WHISPER_KEY[0] if WHISPER_KEY else "previous model"
        if log_fn:
            log_fn(f"🧹 Unloading previous ASR model '{old_name}' from memory...")
        WHISPER_BATCHED = None
        WHISPER_MODEL = None
        WHISPER_KEY = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def load_whisper_model(
    model_size: str, log_fn
) -> Tuple[WhisperModel, Optional[BatchedInferencePipeline], str]:
    global WHISPER_MODEL, WHISPER_BATCHED, WHISPER_KEY, FORCE_CPU
    has_cuda = False
    if not FORCE_CPU:
        try:
            res = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            has_cuda = res.returncode == 0
        except Exception:
            has_cuda = False

    device = "cuda" if (has_cuda and not FORCE_CPU) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    if WHISPER_MODEL is not None and WHISPER_KEY == (model_size, device):
        log_fn(
            f"♻️ Using cached ASR Model '{model_size}' on {device.upper()} ({compute_type})"
        )
        return WHISPER_MODEL, WHISPER_BATCHED, device

    release_whisper_vram(log_fn)

    candidates = MODEL_FALLBACK_MIRRORS.get(model_size, [model_size, "large-v3"])
    cpu_threads = max(4, os.cpu_count() or 4)

    try:
        loaded_candidate = None
        last_err = None
        for candidate in candidates:
            try:
                log_fn(
                    f"Loading SOTA ASR Model '{candidate}' on {device.upper()} ({compute_type})..."
                )
                WHISPER_MODEL = WhisperModel(
                    candidate,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                    num_workers=2,
                )
                loaded_candidate = candidate
                break
            except Exception as err:
                last_err = err
                log_fn(
                    f"⚠️ Mirror '{candidate}' unavailable ({err}); trying next mirror..."
                )

        if WHISPER_MODEL is None:
            raise RuntimeError(f"All mirrors failed: {last_err}")

        WHISPER_BATCHED = (
            BatchedInferencePipeline(model=WHISPER_MODEL)
            if device == "cuda"
            else None
        )
        WHISPER_KEY = (model_size, device)
        log_fn(
            f"ASR GPU Ready: {loaded_candidate} ({device} / {compute_type} / Anti-Drift Sequential 15s Window Enabled)"
        )
        return WHISPER_MODEL, WHISPER_BATCHED, device
    except Exception as e:
        FORCE_CPU = True
        log_fn(f"GPU fallback to CPU int8: {e}")
        release_whisper_vram()
        fallback_id = candidates[0] if candidates else "large-v3"
        try:
            WHISPER_MODEL = WhisperModel(
                fallback_id, device="cpu", compute_type="int8"
            )
        except Exception:
            WHISPER_MODEL = WhisperModel(
                "large-v3", device="cpu", compute_type="int8"
            )
        WHISPER_BATCHED = None
        WHISPER_KEY = (model_size, "cpu")
        return WHISPER_MODEL, None, "cpu"


def fallback_split_text_without_words(
    txt: str, start_sec: float, end_sec: float, max_chars: int = 65
) -> List[Dict[str, Any]]:
    clean = normalize_multiline_text(txt)
    if not clean:
        return []
    total_dur = max(0.3, end_sec - start_sec)
    if len(clean) <= max_chars and total_dur <= 5.0:
        return [
            {
                "start": round(start_sec, 2),
                "end": round(max(start_sec + 0.25, end_sec), 2),
                "source": clean,
                "target": "",
            }
        ]

    raw_parts = re.split(r"(?<=[.?!。？！])\s+", clean)
    parts: List[str] = []
    cur = ""
    for p in raw_parts:
        if not p.strip():
            continue
        if cur and len(cur) + 1 + len(p) <= max_chars:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                parts.append(cur)
            cur = p.strip()
    if cur:
        parts.append(cur)

    if not parts:
        parts = [clean]

    total_chars = max(1, sum(len(p) for p in parts))
    out: List[Dict[str, Any]] = []
    cursor = start_sec
    for idx, p in enumerate(parts):
        frac = len(p) / total_chars
        seg_dur = max(0.35, total_dur * frac)
        seg_end = (
            end_sec
            if idx == len(parts) - 1
            else min(end_sec, cursor + seg_dur)
        )
        out.append(
            {
                "start": round(cursor, 2),
                "end": round(max(cursor + 0.2, seg_end), 2),
                "source": p,
                "target": "",
            }
        )
        cursor = seg_end
    return out


def split_segment_by_words(
    seg: Any,
    source_lang_code: str,
    max_pause_sec: float = 0.50,
    max_duration_sec: float = 4.5,
    max_chars_latin: int = 65,
    max_chars_cjk: int = 28,
) -> List[Dict[str, Any]]:
    words = getattr(seg, "words", None)

    def _convert_zh(s: str) -> str:
        if source_lang_code == "zh-TW":
            return OPENCC_S2TWP.convert(s)
        if source_lang_code == "zh-CN":
            return OPENCC_T2S.convert(s)
        return s

    if not words:
        txt = _convert_zh(normalize_multiline_text(seg.text or ""))
        return fallback_split_text_without_words(
            txt, float(seg.start), float(seg.end)
        )

    chunks: List[Dict[str, Any]] = []
    cur_words: List[str] = []
    chunk_start: Optional[float] = None
    chunk_end: Optional[float] = None

    sentence_end_re = re.compile(r'[.?!。？！…]+["\'\)\]”’»]*$')
    clause_end_re = re.compile(r'[,;:，；：—\-]+["\'\)\]”’»]*$')
    abbrev_re = re.compile(
        r"\b(mr|mrs|ms|dr|prof|sr|jr|st|vs|etc)\.$", re.IGNORECASE
    )

    def _flush_current() -> None:
        nonlocal cur_words, chunk_start, chunk_end
        if cur_words and chunk_start is not None and chunk_end is not None:
            raw_txt = _convert_zh(normalize_multiline_text("".join(cur_words)))
            if raw_txt:
                chunks.append(
                    {
                        "start": round(chunk_start, 2),
                        "end": round(max(chunk_start + 0.25, chunk_end), 2),
                        "source": raw_txt,
                        "target": "",
                    }
                )
        cur_words = []
        chunk_start = None
        chunk_end = None

    for w in words:
        w_text = getattr(w, "word", "") or ""
        if not w_text.strip():
            continue
        w_start = float(getattr(w, "start", seg.start))
        w_end = float(getattr(w, "end", w_start + 0.15))

        if w_end - w_start > 1.8:
            w_start = max(w_start, w_end - 0.9)

        if chunk_start is None:
            chunk_start = w_start
            chunk_end = w_end
            cur_words = [w_text]
        else:
            gap = w_start - (chunk_end or w_start)
            current_text_so_far = "".join(cur_words).strip()
            has_cjk = bool(
                re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", current_text_so_far)
            )
            char_limit = max_chars_cjk if has_cjk else max_chars_latin

            prospective_dur = w_end - chunk_start
            prospective_len = len(current_text_so_far) + len(w_text)

            if gap >= max_pause_sec or (
                len(cur_words) >= 2
                and (
                    prospective_dur > max_duration_sec
                    or prospective_len > char_limit
                )
            ):
                _flush_current()
                chunk_start = w_start
                chunk_end = w_end
                cur_words = [w_text]
            else:
                chunk_end = max(chunk_end or w_end, w_end)
                cur_words.append(w_text)

        joined_now = "".join(cur_words).strip()
        dur_now = (chunk_end or w_end) - (chunk_start or w_start)
        has_cjk_now = bool(
            re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", joined_now)
        )
        soft_sent_min = 5 if has_cjk_now else 12
        soft_cls_min = 12 if has_cjk_now else 34

        w_stripped = w_text.strip()
        is_sent_end = bool(sentence_end_re.search(w_stripped)) and not bool(
            abbrev_re.search(w_stripped)
        )
        is_clause_end = bool(clause_end_re.search(w_stripped))

        if is_sent_end and (len(joined_now) >= soft_sent_min or dur_now >= 1.1):
            _flush_current()
        elif is_clause_end and (
            len(joined_now) >= soft_cls_min or dur_now >= 2.6
        ):
            _flush_current()

    _flush_current()
    return chunks


# ==============================================================================
# Multi-Engine Free Machine Translation + Whole-Passage Context-Aware Engine
# ==============================================================================


def postprocess_target_zh(text: str, tgt_lang: str) -> str:
    if tgt_lang == "zh-TW":
        return OPENCC_S2TWP.convert(text)
    if tgt_lang == "zh-CN":
        return OPENCC_T2S.convert(text)
    return text


def translate_single_line_google_free(
    body: str, src_lang: str, tgt_lang: str
) -> str:
    sl = GOOGLE_LANG_MAP.get(src_lang, "auto")
    tl = GOOGLE_LANG_MAP.get(tgt_lang, "zh-TW")

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": sl,
            "tl": tl,
            "dt": "t",
            "q": body,
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                pieces = [
                    chunk[0]
                    for chunk in data[0]
                    if isinstance(chunk, list) and chunk and chunk[0]
                ]
                merged = "".join(pieces).strip()
                if merged:
                    return postprocess_target_zh(merged, tgt_lang)
    except Exception:
        pass

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source=sl, target=tl).translate(body)
        if translated:
            return postprocess_target_zh(str(translated).strip(), tgt_lang)
    except Exception:
        pass

    return postprocess_target_zh(body, tgt_lang)


def translate_single_line_translators_lib(
    body: str, src_lang: str, tgt_lang: str, provider: str = "alibaba"
) -> str:
    from_lang = TRANSLATORS_LIB_LANG_MAP.get(src_lang, "auto")
    to_lang = TRANSLATORS_LIB_LANG_MAP.get(tgt_lang, "zh-CHT")
    try:
        import translators as ts

        res = ts.translate_text(
            query_text=body,
            translator=provider,
            from_language=from_lang,
            to_language=to_lang,
            timeout=10.0,
        )
        if res and str(res).strip():
            return postprocess_target_zh(str(res).strip(), tgt_lang)
    except Exception:
        pass

    return translate_single_line_google_free(body, src_lang, tgt_lang)


def translate_single_line_mymemory(
    body: str, src_lang: str, tgt_lang: str
) -> str:
    url = "https://api.mymemory.translated.net/get"
    params = {"q": body, "langpair": f"{src_lang}|{tgt_lang}"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        translated = (
            data.get("responseData", {}).get("translatedText", "") or body
        )
        translated = html.unescape(translated).strip()
        if "MYMEMORY WARNING" in translated.upper():
            return translate_single_line_google_free(body, src_lang, tgt_lang)
    except Exception:
        return translate_single_line_google_free(body, src_lang, tgt_lang)

    return postprocess_target_zh(translated, tgt_lang)


def translate_raw_block(
    text_block: str,
    src_lang: str,
    tgt_lang: str,
    mt_engine: str = "google_free",
) -> str:
    """Translates a multi-sentence block at once so the MT engine has full discourse context."""
    if not text_block.strip() or src_lang == tgt_lang:
        return text_block
    if mt_engine == "google_free":
        return translate_single_line_google_free(text_block, src_lang, tgt_lang)
    if mt_engine == "alibaba_free":
        return translate_single_line_translators_lib(
            text_block, src_lang, tgt_lang, provider="alibaba"
        )
    if mt_engine == "bing_free":
        return translate_single_line_translators_lib(
            text_block, src_lang, tgt_lang, provider="bing"
        )
    return translate_single_line_mymemory(text_block, src_lang, tgt_lang)


def translate_cue_text(
    text: str,
    src_lang: str,
    tgt_lang: str,
    mt_engine: str = "google_free",
) -> str:
    clean = normalize_multiline_text(text)
    if not clean or src_lang == tgt_lang:
        return clean

    def _translate_line(line_str: str) -> str:
        prefix = ""
        body = line_str.strip()
        if body.startswith("- ") or body.startswith("— "):
            prefix = "- "
            body = body[2:].strip()
        if not body:
            return line_str
        trans = translate_raw_block(body, src_lang, tgt_lang, mt_engine)
        return f"{prefix}{trans}"

    lines = clean.split("\n")
    return "\n".join(_translate_line(ln) for ln in lines if ln.strip())


def split_translated_sentence_proportionally(
    translated_sentence: str, source_lengths: List[int]
) -> List[str]:
    """Splits a full-sentence translation back across the constituent subtitle cues proportional to each cue's length."""
    n = len(source_lengths)
    clean = translated_sentence.strip()
    if n <= 1 or not clean:
        return [clean] + [""] * max(0, n - 1)

    total_src = max(1, sum(max(1, L) for L in source_lengths))
    has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", clean))

    if has_cjk:
        total_chars = len(clean)
        results: List[str] = []
        cursor = 0
        acc_src = 0
        for i in range(n):
            if i == n - 1:
                part = clean[cursor:].strip(" ，、；：")
                results.append(part if part else clean[-1:])
                break
            acc_src += max(1, source_lengths[i])
            target_idx = int(round((acc_src / total_src) * total_chars))
            target_idx = max(
                cursor + 1, min(total_chars - (n - 1 - i), target_idx)
            )
            best_idx = target_idx
            for delta in range(0, 7):
                for cand in (target_idx + delta, target_idx - delta):
                    if (
                        cursor < cand < total_chars
                        and clean[cand - 1] in "，。！？；、,.;!? "
                    ):
                        best_idx = cand
                        break
                if best_idx != target_idx:
                    break
            part = clean[cursor:best_idx].strip(" ，、")
            results.append(part)
            cursor = best_idx
        return results
    else:
        words = clean.split()
        if len(words) < n:
            return [clean] + [""] * (n - 1)
        results = []
        w_cursor = 0
        acc_src = 0
        total_w = len(words)
        for i in range(n):
            if i == n - 1:
                results.append(" ".join(words[w_cursor:]))
                break
            acc_src += max(1, source_lengths[i])
            target_w = int(round((acc_src / total_src) * total_w))
            target_w = max(w_cursor + 1, min(total_w - (n - 1 - i), target_w))
            results.append(" ".join(words[w_cursor:target_w]))
            w_cursor = target_w
        return results


def translate_segments_full_context(
    segments: List[Dict[str, Any]],
    src_lang: str,
    tgt_lang: str,
    mt_engine: str = "google_free",
    translation_mode: str = "full_context_passage",
    log_fn=None,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    if src_lang == tgt_lang:
        for seg in segments:
            seg["target"] = seg.get("source", "")
        return segments

    if translation_mode == "line_by_line":
        if log_fn:
            log_fn(f"Running Line-by-Line translation via '{mt_engine}'...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = {
                pool.submit(
                    translate_cue_text,
                    seg.get("source", ""),
                    src_lang,
                    tgt_lang,
                    mt_engine,
                ): idx
                for idx, seg in enumerate(segments)
            }
            for fut in concurrent.futures.as_completed(futs):
                idx = futs[fut]
                segments[idx]["target"] = fut.result()
        return segments

    sent_end_re = re.compile(r'[.?!。？！…]+["\'\)\]”’»]*$')
    sentence_groups: List[List[int]] = []
    current_group: List[int] = []

    for idx, seg in enumerate(segments):
        src_txt = (
            normalize_multiline_text(seg.get("source", ""))
            .replace("\n", " ")
            .strip()
        )
        current_group.append(idx)
        is_end = bool(sent_end_re.search(src_txt))
        if is_end or len(current_group) >= 3:
            sentence_groups.append(current_group)
            current_group = []
    if current_group:
        sentence_groups.append(current_group)

    passage_batches: List[List[List[int]]] = []
    cur_batch: List[List[int]] = []
    cur_chars = 0
    for s_group in sentence_groups:
        s_text = " ".join(
            normalize_multiline_text(segments[i].get("source", ""))
            .replace("\n", " ")
            .strip()
            for i in s_group
        )
        if cur_batch and (
            len(cur_batch) >= 12 or cur_chars + len(s_text) > 1400
        ):
            passage_batches.append(cur_batch)
            cur_batch = []
            cur_chars = 0
        cur_batch.append(s_group)
        cur_chars += len(s_text) + 8
    if cur_batch:
        passage_batches.append(cur_batch)

    if log_fn:
        log_fn(
            f"🧠 Whole-Passage Context Translation: {len(segments)} cues grouped into {len(sentence_groups)} complete sentences across {len(passage_batches)} discourse passage(s)"
        )

    def _process_passage_batch(
        batch_idx_and_groups: Tuple[int, List[List[int]]]
    ) -> Dict[int, str]:
        _, s_groups = batch_idx_and_groups
        out_map: Dict[int, str] = {}

        tagged_lines: List[str] = []
        group_source_texts: List[str] = []
        for g_idx, cue_indices in enumerate(s_groups, start=1):
            merged_src = " ".join(
                normalize_multiline_text(segments[i].get("source", ""))
                .replace("\n", " ")
                .strip()
                for i in cue_indices
                if normalize_multiline_text(
                    segments[i].get("source", "")
                ).strip()
            )
            group_source_texts.append(merged_src)
            tagged_lines.append(f"[{g_idx}] {merged_src}")

        full_passage_query = "\n".join(tagged_lines)
        translated_passage = translate_raw_block(
            full_passage_query, src_lang, tgt_lang, mt_engine
        )

        parsed_groups: Dict[int, str] = {}
        pattern = re.compile(
            r"(?:^|\n)\s*[\[\【\(]\s*(\d+)\s*[\]\】\)]\s*(.*?)(?=(?:\n\s*[\[\【\(]\s*\d+\s*[\]\】\)])|\Z)",
            re.DOTALL,
        )
        for m in pattern.finditer(translated_passage):
            g_num = int(m.group(1))
            g_trans = m.group(2).strip()
            if g_trans:
                parsed_groups[g_num] = g_trans

        if len(parsed_groups) < len(s_groups):
            raw_lines = [
                re.sub(r"^\s*[\[\【\(]?\d+[\]\】\)\.]?\s*", "", ln).strip()
                for ln in translated_passage.split("\n")
                if ln.strip()
            ]
            if len(raw_lines) == len(s_groups):
                for g_idx, ln in enumerate(raw_lines, start=1):
                    parsed_groups.setdefault(g_idx, ln)

        for g_idx, cue_indices in enumerate(s_groups, start=1):
            trans_sent = parsed_groups.get(g_idx, "").strip()
            if not trans_sent:
                trans_sent = translate_raw_block(
                    group_source_texts[g_idx - 1], src_lang, tgt_lang, mt_engine
                ).strip()

            if len(cue_indices) == 1:
                out_map[cue_indices[0]] = trans_sent
            else:
                src_lengths = [
                    max(
                        1,
                        len(
                            normalize_multiline_text(
                                segments[i].get("source", "")
                            )
                        ),
                    )
                    for i in cue_indices
                ]
                split_parts = split_translated_sentence_proportionally(
                    trans_sent, src_lengths
                )
                for cue_i, part_txt in zip(cue_indices, split_parts):
                    out_map[cue_i] = (
                        part_txt.strip()
                        if part_txt.strip()
                        else translate_cue_text(
                            segments[cue_i].get("source", ""),
                            src_lang,
                            tgt_lang,
                            mt_engine,
                        )
                    )
        return out_map

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_process_passage_batch, (b_i, batch))
            for b_i, batch in enumerate(passage_batches)
        ]
        for fut in concurrent.futures.as_completed(futures):
            batch_map = fut.result()
            for cue_idx, tgt_val in batch_map.items():
                segments[cue_idx]["target"] = tgt_val

    return segments


# ==============================================================================
# Live Bilingual HTML5 Workbench + Add / Delete / Split Subtitle Functions
# ==============================================================================

COLLECT_WORKBENCH_JS = """
(wb_json, pres_mode, font_sz, current_json, baseline_json, video_path) => {
    const container = document.getElementById('pe-workbench-container');
    if (!container) {
        return [wb_json, pres_mode, font_sz, current_json, baseline_json, video_path];
    }
    const rows = container.querySelectorAll('.pe-wb-row');
    const collected = [];
    rows.forEach((row, idx) => {
        const startEl = row.querySelector('.pe-input-start');
        const endEl = row.querySelector('.pe-input-end');
        const srcEl = row.querySelector('.pe-input-source');
        const tgtEl = row.querySelector('.pe-input-target');
        const startVal = startEl ? parseFloat(startEl.value || '0') : 0.0;
        const endVal = endEl ? parseFloat(endEl.value || '0') : (startVal + 1.0);
        const srcVal = srcEl ? srcEl.value : '';
        const tgtVal = tgtEl ? tgtEl.value : '';
        collected.push({
            id: idx + 1,
            start: isNaN(startVal) ? 0.0 : startVal,
            end: isNaN(endVal) ? (startVal + 1.0) : endVal,
            source: srcVal,
            target: tgtVal
        });
    });
    const modeSelect = document.getElementById('pe-wb-mode-select');
    const initMode = container.getAttribute('data-init-mode') || '';
    let selectedMode = pres_mode;
    if (modeSelect) {
        if (pres_mode !== initMode) {
            selectedMode = pres_mode;
            modeSelect.value = pres_mode;
        } else if (modeSelect.getAttribute('data-user-changed') === '1') {
            selectedMode = modeSelect.value;
        }
    }
    const fontSlider = document.getElementById('pe-wb-fontsize-slider');
    const initFs = parseInt(container.getAttribute('data-init-fs') || '20', 10);
    let selectedFontSize = parseInt(font_sz || 20, 10);
    if (fontSlider) {
        if (selectedFontSize !== initFs) {
            fontSlider.value = String(selectedFontSize);
        } else if (fontSlider.getAttribute('data-user-changed') === '1') {
            selectedFontSize = parseInt(fontSlider.value, 10);
        }
    }
    const payloadObj = {
        segments: collected,
        presentation_mode: selectedMode,
        font_size: isNaN(selectedFontSize) ? 20 : selectedFontSize,
        ts: Date.now()
    };
    const payload = JSON.stringify(payloadObj);
    try {
        localStorage.setItem('pe_workbench_autosave_v2', payload);
        window.__peDirty = false;
    } catch (e) {}
    return [payload, selectedMode, isNaN(selectedFontSize) ? font_sz : selectedFontSize, current_json, baseline_json, video_path];
}
"""


def build_bilingual_workbench_html(
    video_path: Optional[str],
    segments: List[Dict[str, Any]],
    presentation_mode: str = "bilingual_target_top",
    font_size: int = 20,
    model_badge: str = "deepdml/faster-distil-whisper-large-v3.5 + google_free",
) -> str:
    if not video_path or not os.path.exists(video_path):
        return """
        <div style="padding:28px;border:2px dashed #475569;border-radius:14px;background:#0f172a;color:#e2e8f0;text-align:center;">
            <h3 style="margin:0 0 8px 0;color:#38bdf8;">🎬 SubCraft-PE: Interactive Bilingual Video & Student-Safe HTML Workbench</h3>
            <p style="margin:0;color:#94a3b8;font-size:14px;">
                Load an <b>In-Class Practice Sample Clip</b> (or upload your own video) and click <b>Step 1 (Transcribe & Translate)</b> to unlock the live dual-subtitle video player, real-time browser auto-save, <b>➕ Add Subtitle</b>, <b>🗑️ Delete Subtitle</b>, <b>✂️ Split Cue</b>, and C-E / E-C Strategy PDF Report generator.
            </p>
        </div>
        """

    run_stamp = int(time.time() * 1000)
    font_size = int(font_size or 20)
    sub_font_size = max(11, int(round(font_size * 0.76)))
    video_src_url = make_gradio_file_url(video_path)

    ontimeupdate_js = (
        "const t=this.currentTime;"
        "const root=document.getElementById('pe-workbench-container');"
        "if(!root)return;"
        "const modeEl=document.getElementById('pe-wb-mode-select');"
        "const mode=modeEl?modeEl.value:'bilingual_target_top';"
        "const fsEl=document.getElementById('pe-wb-fontsize-slider');"
        "const mainFs=fsEl?parseInt(fsEl.value||'20',10):20;"
        "const subFs=Math.max(11,Math.round(mainFs*0.76));"
        "const fsValLabel=document.getElementById('pe-wb-fontsize-val');"
        "if(fsValLabel){fsValLabel.textContent=mainFs+'px';}"
        "const rows=root.querySelectorAll('.pe-wb-row');"
        "let activeSrc='',activeTgt='',matched=false;"
        "rows.forEach(r=>{"
        "  const s=parseFloat(r.querySelector('.pe-input-start').value||'0');"
        "  const e=parseFloat(r.querySelector('.pe-input-end').value||'0');"
        "  if(!matched && t>=s && t<=e){"
        "    matched=true;"
        "    r.style.borderColor='#38bdf8';"
        "    r.style.background='rgba(56,189,248,0.13)';"
        "    activeSrc=(r.querySelector('.pe-input-source').value||'').replace(/\\r\\n/g,'\\n').trim();"
        "    activeTgt=(r.querySelector('.pe-input-target').value||'').replace(/\\r\\n/g,'\\n').trim();"
        "  }else{"
        "    r.style.borderColor='#334155';"
        "    r.style.background='#1e293b';"
        "  }"
        "});"
        "const l1=document.getElementById('pe-ov-line1');"
        "const l2=document.getElementById('pe-ov-line2');"
        "if(!l1||!l2)return;"
        "if(!matched || (!activeSrc && !activeTgt)){"
        "  l1.textContent=''; l2.textContent='';"
        "  l1.style.display='none'; l2.style.display='none';"
        "  return;"
        "}"
        "if(mode==='bilingual_target_top'){"
        "  l1.textContent=activeTgt||activeSrc; l1.style.display=(activeTgt||activeSrc)?'inline-block':'none';"
        "  l1.style.color='#fde047'; l1.style.fontSize=mainFs+'px';"
        "  l2.textContent=activeSrc; l2.style.display=(activeTgt&&activeSrc)?'inline-block':'none';"
        "  l2.style.color='#f8fafc'; l2.style.fontSize=subFs+'px';"
        "}else if(mode==='bilingual_source_top'){"
        "  l1.textContent=activeSrc||activeTgt; l1.style.display=(activeSrc||activeTgt)?'inline-block':'none';"
        "  l1.style.color='#f8fafc'; l1.style.fontSize=subFs+'px';"
        "  l2.textContent=activeTgt; l2.style.display=(activeSrc&&activeTgt)?'inline-block':'none';"
        "  l2.style.color='#fde047'; l2.style.fontSize=mainFs+'px';"
        "}else if(mode==='target_only'){"
        "  const showT=activeTgt||activeSrc;"
        "  l1.textContent=showT; l1.style.display=showT?'inline-block':'none';"
        "  l1.style.color='#fde047'; l1.style.fontSize=mainFs+'px';"
        "  l2.style.display='none';"
        "}else{"
        "  const showS=activeSrc||activeTgt;"
        "  l1.textContent=showS; l1.style.display=showS?'inline-block':'none';"
        "  l1.style.color='#f8fafc'; l1.style.fontSize=mainFs+'px';"
        "  l2.style.display='none';"
        "}"
    )

    autosave_and_refresh_js = (
        "const v=document.getElementById('pe-wb-video');"
        "if(v){v.dispatchEvent(new Event('timeupdate'));}"
        "try{"
        "  const root=document.getElementById('pe-workbench-container');"
        "  if(root){"
        "    const rows=root.querySelectorAll('.pe-wb-row');"
        "    const arr=[];"
        "    rows.forEach((r,i)=>{"
        "      const numBadge=r.querySelector('.pe-row-num');"
        "      if(numBadge)numBadge.textContent='#'+(i+1);"
        "      r.setAttribute('data-idx',String(i+1));"
        "      arr.push({"
        "        id:i+1,"
        "        start:parseFloat(r.querySelector('.pe-input-start').value||'0'),"
        "        end:parseFloat(r.querySelector('.pe-input-end').value||'1'),"
        "        source:r.querySelector('.pe-input-source').value||'',"
        "        target:r.querySelector('.pe-input-target').value||''"
        "      });"
        "    });"
        "    const totEl=document.getElementById('pe-total-cues-count');"
        "    if(totEl)totEl.textContent=String(rows.length);"
        "    const mEl=document.getElementById('pe-wb-mode-select');"
        "    const fEl=document.getElementById('pe-wb-fontsize-slider');"
        "    localStorage.setItem('pe_workbench_autosave_v2', JSON.stringify({"
        "      segments:arr,"
        "      presentation_mode:mEl?mEl.value:'bilingual_target_top',"
        "      font_size:fEl?parseInt(fEl.value||'20',10):20,"
        "      savedAt:new Date().toLocaleTimeString()"
        "    }));"
        "    window.__peDirty=true;"
        "    if(!window.__peUnloadBound){"
        "      window.addEventListener('beforeunload',function(e){if(window.__peDirty){e.preventDefault();e.returnValue='';}});"
        "      window.__peUnloadBound=true;"
        "    }"
        "    const badge=document.getElementById('pe-autosave-status');"
        "    if(badge){badge.textContent='💾 Auto-saved ('+rows.length+' cues) at '+new Date().toLocaleTimeString();badge.style.color='#34d399';}"
        "  }"
        "}catch(err){}"
    )

    restore_draft_btn_js = (
        "try{"
        "  const raw=localStorage.getItem('pe_workbench_autosave_v2');"
        "  if(!raw){alert('No browser auto-saved draft found yet!');return;}"
        "  const data=JSON.parse(raw);"
        "  if(!data||!data.segments||!data.segments.length){alert('Saved draft is empty.');return;}"
        "  if(!confirm('Restore '+data.segments.length+' subtitle segments from your browser auto-save ('+(data.savedAt||'recent')+')?'))return;"
        "  const root=document.getElementById('pe-workbench-container');"
        "  const listContainer=document.getElementById('pe-rows-scroll-box');"
        "  let rows=root.querySelectorAll('.pe-wb-row');"
        "  while(rows.length < data.segments.length && rows.length > 0){"
        "    const clone=rows[rows.length-1].cloneNode(true);"
        "    listContainer.appendChild(clone);"
        "    rows=root.querySelectorAll('.pe-wb-row');"
        "  }"
        "  while(rows.length > data.segments.length && rows.length > 1){"
        "    rows[rows.length-1].remove();"
        "    rows=root.querySelectorAll('.pe-wb-row');"
        "  }"
        "  rows.forEach((r,i)=>{"
        "    if(i<data.segments.length){"
        "      const seg=data.segments[i];"
        "      if(seg.start!==undefined)r.querySelector('.pe-input-start').value=Number(seg.start).toFixed(2);"
        "      if(seg.end!==undefined)r.querySelector('.pe-input-end').value=Number(seg.end).toFixed(2);"
        "      if(seg.source!==undefined)r.querySelector('.pe-input-source').value=seg.source;"
        "      if(seg.target!==undefined)r.querySelector('.pe-input-target').value=seg.target;"
        "    }"
        "  });"
        + autosave_and_refresh_js
        + "}catch(e){alert('Could not restore draft: '+e);}"
    )

    dual_speaker_btn_js = (
        "const row=this.closest('.pe-wb-row');"
        "['.pe-input-source','.pe-input-target'].forEach(sel=>{"
        "  const ta=row.querySelector(sel);"
        "  if(!ta)return;"
        "  let val=ta.value.replace(/\\r\\n/g,'\\n').trim();"
        "  if(!val)return;"
        "  if(val.includes('\\n')){"
        "    ta.value=val.split('\\n').map(l=>{l=l.trim();return (l.startsWith('-')||l.startsWith('—'))?l:'- '+l;}).join('\\n');"
        "  }else{"
        "    const pos=(ta.selectionStart&&ta.selectionStart>0&&ta.selectionStart<val.length)?ta.selectionStart:Math.floor(val.length/2);"
        "    const p1=val.slice(0,pos).trim();"
        "    const p2=val.slice(pos).trim();"
        "    ta.value='- '+p1+'\\n- '+p2;"
        "  }"
        "});"
        + autosave_and_refresh_js
    )

    insert_break_btn_js = (
        "const row=this.closest('.pe-wb-row');"
        "const ta=row.querySelector('.pe-input-target');"
        "if(ta){"
        "  const val=ta.value;"
        "  const pos=(ta.selectionStart!==undefined && ta.selectionStart>0 && ta.selectionStart<val.length)?ta.selectionStart:Math.floor(val.length/2);"
        "  ta.value=val.slice(0,pos).trimEnd()+'\\n'+val.slice(pos).trimStart();"
        "  ta.focus();"
        "}"
        + autosave_and_refresh_js
    )

    split_cue_btn_js = (
        "const row=this.closest('.pe-wb-row');"
        "if(!row)return;"
        "const sEl=row.querySelector('.pe-input-start');"
        "const eEl=row.querySelector('.pe-input-end');"
        "const srcEl=row.querySelector('.pe-input-source');"
        "const tgtEl=row.querySelector('.pe-input-target');"
        "const s=parseFloat(sEl.value||'0');"
        "const e=parseFloat(eEl.value||'1');"
        "const src=srcEl.value||'';"
        "const tgt=tgtEl.value||'';"
        "function smartSplit(str,ta){"
        "  if(!str)return ['',''];"
        "  if(ta && ta.selectionStart>0 && ta.selectionStart<str.length){"
        "    return [str.slice(0,ta.selectionStart).trim(), str.slice(ta.selectionStart).trim()];"
        "  }"
        "  if(str.includes('\\n')){"
        "    const lines=str.split('\\n').filter(Boolean);"
        "    if(lines.length>=2){"
        "      const midL=Math.ceil(lines.length/2);"
        "      return [lines.slice(0,midL).join('\\n').trim(), lines.slice(midL).join('\\n').trim()];"
        "    }"
        "  }"
        "  const mid=Math.floor(str.length/2);"
        "  let best=-1;"
        "  for(let d=0;d<Math.min(40,mid);d++){"
        "    const rIdx=mid+d, lIdx=mid-d;"
        "    if('.?!。？！,，;； '.includes(str[rIdx])){best=rIdx+1;break;}"
        "    if('.?!。？！,，;； '.includes(str[lIdx])){best=lIdx+1;break;}"
        "  }"
        "  if(best<=0||best>=str.length)best=mid;"
        "  return [str.slice(0,best).trim(), str.slice(best).trim()];"
        "}"
        "const [src1,src2]=smartSplit(src,srcEl);"
        "const [tgt1,tgt2]=smartSplit(tgt,tgtEl);"
        "const ratio=(tgt1.length&&tgt.length)?(tgt1.length/tgt.length):((src1.length&&src.length)?(src1.length/src.length):0.5);"
        "const clampedRatio=Math.max(0.2,Math.min(0.8,ratio));"
        "const midTime=Number((s+(e-s)*clampedRatio).toFixed(2));"
        "const clone=row.cloneNode(true);"
        "eEl.value=midTime.toFixed(2);"
        "srcEl.value=src1;"
        "tgtEl.value=tgt1;"
        "clone.querySelector('.pe-input-start').value=midTime.toFixed(2);"
        "clone.querySelector('.pe-input-end').value=e.toFixed(2);"
        "clone.querySelector('.pe-input-source').value=src2;"
        "clone.querySelector('.pe-input-target').value=tgt2;"
        "row.parentNode.insertBefore(clone,row.nextSibling);"
        + autosave_and_refresh_js
    )

    add_after_row_btn_js = (
        "const row=this.closest('.pe-wb-row');"
        "if(!row)return;"
        "const curEnd=parseFloat(row.querySelector('.pe-input-end').value||'0');"
        "const nextRow=row.nextElementSibling;"
        "let newStart=Number(curEnd.toFixed(2));"
        "let newEnd=Number((curEnd+2.0).toFixed(2));"
        "if(nextRow && nextRow.classList.contains('pe-wb-row')){"
        "  const nextStart=parseFloat(nextRow.querySelector('.pe-input-start').value||'0');"
        "  if(nextStart > curEnd + 0.3){"
        "    newEnd=Number(Math.min(curEnd+2.5, nextStart).toFixed(2));"
        "  }"
        "}"
        "const clone=row.cloneNode(true);"
        "clone.querySelector('.pe-input-start').value=newStart.toFixed(2);"
        "clone.querySelector('.pe-input-end').value=newEnd.toFixed(2);"
        "clone.querySelector('.pe-input-source').value='';"
        "clone.querySelector('.pe-input-target').value='';"
        "row.parentNode.insertBefore(clone, row.nextSibling);"
        "const tgtInput=clone.querySelector('.pe-input-target');"
        "if(tgtInput)tgtInput.focus();"
        + autosave_and_refresh_js
    )

    delete_row_btn_js = (
        "const row=this.closest('.pe-wb-row');"
        "const root=document.getElementById('pe-workbench-container');"
        "if(!row||!root)return;"
        "const allRows=root.querySelectorAll('.pe-wb-row');"
        "if(allRows.length<=1){"
        "  row.querySelector('.pe-input-source').value='';"
        "  row.querySelector('.pe-input-target').value='';"
        "}else{"
        "  row.remove();"
        "}"
        + autosave_and_refresh_js
    )

    add_at_video_time_btn_js = (
        "const root=document.getElementById('pe-workbench-container');"
        "const listBox=document.getElementById('pe-rows-scroll-box');"
        "const v=document.getElementById('pe-wb-video');"
        "if(!root||!listBox)return;"
        "const curT=v?Number(v.currentTime.toFixed(2)):0.0;"
        "const endT=Number((curT+2.0).toFixed(2));"
        "const rows=Array.from(root.querySelectorAll('.pe-wb-row'));"
        "if(!rows.length)return;"
        "const template=rows[rows.length-1].cloneNode(true);"
        "template.querySelector('.pe-input-start').value=curT.toFixed(2);"
        "template.querySelector('.pe-input-end').value=endT.toFixed(2);"
        "template.querySelector('.pe-input-source').value='';"
        "template.querySelector('.pe-input-target').value='';"
        "let inserted=false;"
        "for(let i=0;i<rows.length;i++){"
        "  const rStart=parseFloat(rows[i].querySelector('.pe-input-start').value||'0');"
        "  if(rStart > curT){"
        "    listBox.insertBefore(template, rows[i]);"
        "    inserted=true;"
        "    break;"
        "  }"
        "}"
        "if(!inserted){listBox.appendChild(template);}"
        "template.scrollIntoView({behavior:'smooth',block:'center'});"
        "const tgtEl=template.querySelector('.pe-input-target');"
        "if(tgtEl)tgtEl.focus();"
        + autosave_and_refresh_js
    )

    mode_options_html = []
    for label, val in PRESENTATION_MODE_CHOICES:
        sel = "selected" if val == presentation_mode else ""
        mode_options_html.append(
            f'<option value="{html.escape(val)}" {sel}>{html.escape(label)}</option>'
        )

    display_segments = (
        segments
        if segments
        else [{"id": 1, "start": 0.0, "end": 2.0, "source": "", "target": ""}]
    )

    rows_html: List[str] = []
    for idx, seg in enumerate(display_segments, start=1):
        start_v = f"{parse_flexible_time(seg.get('start', 0.0)):.2f}"
        end_v = f"{parse_flexible_time(seg.get('end', 1.0)):.2f}"
        raw_src = normalize_multiline_text(seg.get("source", ""))
        raw_tgt = normalize_multiline_text(seg.get("target", ""))
        src_v = html.escape(raw_src)
        tgt_v = html.escape(raw_tgt)

        char_len = len(raw_tgt.replace("\n", ""))
        line_cnt = len([ln for ln in raw_tgt.split("\n") if ln.strip()]) or 1
        dur = max(0.2, float(end_v) - float(start_v))
        cps = round(char_len / dur, 1)

        rows_html.append(
            f"""
            <div class="pe-wb-row" data-idx="{idx}" data-run="{run_stamp}" style="border:1px solid #334155;border-radius:10px;padding:10px 12px;margin-bottom:10px;background:#1e293b;transition:all 0.15s;">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
                    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                        <span class="pe-row-num" style="background:#0284c7;color:#fff;font-weight:700;font-size:12px;padding:2px 8px;border-radius:999px;">#{idx}</span>
                        <button type="button"
                            onclick="const v=document.getElementById('pe-wb-video');const s=parseFloat(this.closest('.pe-wb-row').querySelector('.pe-input-start').value||'0');if(v){{v.currentTime=s+0.01;v.play();}}"
                            style="background:#0f172a;color:#38bdf8;border:1px solid #0284c7;border-radius:6px;padding:3px 9px;font-size:12px;cursor:pointer;font-weight:600;">
                            ▶ Seek & Play
                        </button>
                        <button type="button"
                            title="Insert a new blank subtitle cue right below this row"
                            onclick="{add_after_row_btn_js}"
                            style="background:#0f172a;color:#4ade80;border:1px solid #16a34a;border-radius:6px;padding:3px 8px;font-size:11px;cursor:pointer;font-weight:700;">
                            ➕ Add After / 下方新增
                        </button>
                        <button type="button"
                            title="Split this cue into 2 separate timed subtitle cues at your cursor or midpoint"
                            onclick="{split_cue_btn_js}"
                            style="background:#0f172a;color:#f472b6;border:1px solid #db2777;border-radius:6px;padding:3px 8px;font-size:11px;cursor:pointer;font-weight:700;">
                            ✂️ Split Cue / 拆分
                        </button>
                        <button type="button"
                            title="Insert a line break (Enter) in Target subtitle at cursor or midpoint"
                            onclick="{insert_break_btn_js}"
                            style="background:#0f172a;color:#fde047;border:1px solid #ca8a04;border-radius:6px;padding:3px 8px;font-size:11px;cursor:pointer;font-weight:600;">
                            ↵ Line Break / 換行
                        </button>
                        <button type="button"
                            title="Format as Simultaneous Dual-Speaker Dialogue (- Speaker A / - Speaker B)"
                            onclick="{dual_speaker_btn_js}"
                            style="background:#0f172a;color:#34d399;border:1px solid #059669;border-radius:6px;padding:3px 8px;font-size:11px;cursor:pointer;font-weight:600;">
                            🎭 Dual Speaker (- A / - B)
                        </button>
                        <button type="button"
                            title="Delete this subtitle cue"
                            onclick="{delete_row_btn_js}"
                            style="background:#0f172a;color:#f87171;border:1px solid #dc2626;border-radius:6px;padding:3px 8px;font-size:11px;cursor:pointer;font-weight:700;">
                            🗑️ Delete / 刪除
                        </button>
                    </div>
                    <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#cbd5e1;">
                        <span>Start(s):</span>
                        <input type="number" step="0.05" name="start_{run_stamp}_{idx}" class="pe-input-start" value="{start_v}"
                            oninput="{autosave_and_refresh_js}"
                            style="width:72px;background:#0f172a;color:#f8fafc;border:1px solid #475569;border-radius:5px;padding:3px 6px;font-size:12px;">
                        <span>End(s):</span>
                        <input type="number" step="0.05" name="end_{run_stamp}_{idx}" class="pe-input-end" value="{end_v}"
                            oninput="{autosave_and_refresh_js}"
                            style="width:72px;background:#0f172a;color:#f8fafc;border:1px solid #475569;border-radius:5px;padding:3px 6px;font-size:12px;">
                        <span style="color:#94a3b8;font-size:11px;">({line_cnt}L | {char_len}c | {cps} CPS)</span>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                    <div>
                        <div style="font-size:11px;color:#94a3b8;margin-bottom:3px;font-weight:600;">🎤 Original Speech (Press Enter for Multi-Line / 原文支援換行)</div>
                        <textarea name="src_{run_stamp}_{idx}" class="pe-input-source" rows="2"
                            oninput="{autosave_and_refresh_js}"
                            onfocus="const v=document.getElementById('pe-wb-video');const s=parseFloat(this.closest('.pe-wb-row').querySelector('.pe-input-start').value||'0');if(v&&v.paused){{v.currentTime=s+0.01;v.dispatchEvent(new Event('timeupdate'));}}"
                            style="width:100%;background:#0f172a;color:#e2e8f0;border:1px solid #475569;border-radius:6px;padding:6px 8px;font-size:13px;line-height:1.4;resize:vertical;white-space:pre-wrap;">{src_v}</textarea>
                    </div>
                    <div>
                        <div style="font-size:11px;color:#fde047;margin-bottom:3px;font-weight:600;">🌐 Post-Edited Subtitle (Auto-Saved on Every Keystroke / 譯文即時自動暫存)</div>
                        <textarea name="tgt_{run_stamp}_{idx}" class="pe-input-target" rows="2"
                            oninput="{autosave_and_refresh_js}"
                            onfocus="const v=document.getElementById('pe-wb-video');const s=parseFloat(this.closest('.pe-wb-row').querySelector('.pe-input-start').value||'0');if(v&&v.paused){{v.currentTime=s+0.01;v.dispatchEvent(new Event('timeupdate'));}}"
                            style="width:100%;background:#0f172a;color:#fef08a;border:1px solid #ca8a04;border-radius:6px;padding:6px 8px;font-size:13px;line-height:1.4;resize:vertical;white-space:pre-wrap;">{tgt_v}</textarea>
                    </div>
                </div>
            </div>
            """
        )

    return f"""
    <div id="pe-workbench-container" data-run-stamp="{run_stamp}" data-init-mode="{html.escape(presentation_mode)}" data-init-fs="{font_size}" style="background:#0f172a;color:#f8fafc;padding:16px;border-radius:14px;border:1px solid #334155;font-family:system-ui,-apple-system,sans-serif;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #1e293b;">
            <div>
                <h3 style="margin:0;font-size:17px;color:#38bdf8;">🎬 SubCraft-PE: Live Bilingual Video & Student-Safe Post-Editing Workbench</h3>
                <div style="margin-top:6px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span style="font-size:11px;color:#38bdf8;font-weight:700;background:#0c4a6e;padding:3px 10px;border-radius:999px;border:1px solid #0284c7;">
                        🤖 Active Run: {html.escape(model_badge)}
                    </span>
                    <button type="button" onclick="{add_at_video_time_btn_js}"
                        style="background:#059669;color:#ffffff;border:1px solid #34d399;border-radius:6px;padding:4px 11px;font-size:12px;cursor:pointer;font-weight:700;box-shadow:0 2px 6px rgba(5,150,105,0.35);">
                        ➕ Add Subtitle at Video Time / 在目前影片時間新增字幕
                    </button>
                    <button type="button" onclick="{restore_draft_btn_js}"
                        style="background:#1e293b;color:#fde047;border:1px solid #ca8a04;border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;font-weight:700;">
                        ⏪ Restore Browser Draft / 還原瀏覽器暫存
                    </button>
                    <span id="pe-autosave-status" style="font-size:12px;color:#34d399;font-weight:600;background:#064e3b;padding:3px 10px;border-radius:999px;border:1px solid #059669;">
                        🛡️ Auto-Save Active (Add / Delete / Split / Edits Saved Instantly)
                    </span>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
                <div style="display:flex;align-items:center;gap:6px;background:#1e293b;padding:5px 10px;border-radius:8px;border:1px solid #475569;">
                    <label style="font-size:12px;color:#fde047;font-weight:700;">🔠 Font Size:</label>
                    <input id="pe-wb-fontsize-slider" type="range" min="12" max="38" step="1" value="{font_size}"
                        oninput="this.setAttribute('data-user-changed','1');{autosave_and_refresh_js}"
                        style="width:105px;cursor:pointer;accent-color:#38bdf8;">
                    <span id="pe-wb-fontsize-val" style="font-size:12px;font-weight:700;color:#38bdf8;min-width:34px;">{font_size}px</span>
                </div>
                <div style="display:flex;align-items:center;gap:6px;">
                    <label style="font-size:12px;color:#cbd5e1;font-weight:600;">📺 Mode (Live &amp; Hard-Burn):</label>
                    <select id="pe-wb-mode-select" onchange="this.setAttribute('data-user-changed','1');{autosave_and_refresh_js}"
                        style="background:#1e293b;color:#f8fafc;border:1px solid #38bdf8;border-radius:8px;padding:6px 10px;font-size:12px;font-weight:600;">
                        {''.join(mode_options_html)}
                    </select>
                </div>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:minmax(300px, 1.08fr) minmax(320px, 1.25fr);gap:16px;align-items:start;">
            <div style="position:sticky;top:12px;background:#020617;border-radius:12px;overflow:hidden;border:1px solid #334155;">
                <div style="position:relative;width:100%;background:#000;">
                    <video id="pe-wb-video" controls playsinline preload="metadata"
                        ontimeupdate="{ontimeupdate_js}"
                        onseeked="{ontimeupdate_js}"
                        onloadedmetadata="{ontimeupdate_js}"
                        style="width:100%;max-height:420px;display:block;background:#000;"
                        src="{video_src_url}">
                    </video>
                    <div id="pe-wb-overlay"
                        style="position:absolute;left:4%;right:4%;bottom:54px;pointer-events:none;text-align:center;display:flex;flex-direction:column;align-items:center;gap:5px;z-index:20;">
                        <div id="pe-ov-line1"
                            style="display:none;white-space:pre-line;text-align:center;background:rgba(2,6,23,0.84);color:#fde047;padding:5px 14px;border-radius:6px;font-size:{font_size}px;font-weight:700;line-height:1.35;text-shadow:0 2px 4px #000;max-width:96%;">
                        </div>
                        <div id="pe-ov-line2"
                            style="display:none;white-space:pre-line;text-align:center;background:rgba(2,6,23,0.76);color:#f8fafc;padding:3px 12px;border-radius:5px;font-size:{sub_font_size}px;font-weight:500;line-height:1.3;text-shadow:0 1px 3px #000;max-width:94%;">
                        </div>
                    </div>
                </div>
                <div style="padding:10px 14px;background:#0f172a;border-top:1px solid #1e293b;display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#94a3b8;flex-wrap:wrap;gap:6px;">
                    <span>➕ Add After | ✂️ Split | 🗑️ Delete | 🔥 Hard-Burn MP4 matches selected Mode &amp; Font Size</span>
                    <span>Total Spoken Cues: <b id="pe-total-cues-count" style="color:#38bdf8;">{len(display_segments)}</b></span>
                </div>
            </div>

            <div id="pe-rows-scroll-box" style="max-height:480px;overflow-y:auto;padding-right:6px;">
                {''.join(rows_html)}
            </div>
        </div>
    </div>
    """


# ==============================================================================
# C-E / E-C Post-Editing Strategy Detection, HTML Report & PDF Export Generator
# ==============================================================================


def detect_pe_strategies(
    src_text: str, mt_text: str, pe_text: str
) -> List[str]:
    mt_norm = normalize_multiline_text(mt_text)
    pe_norm = normalize_multiline_text(pe_text)
    if mt_norm == pe_norm:
        return ["No Edit Needed (MT Accepted / 保留機譯)"]

    strategies: List[str] = []
    has_cjk_target = bool(re.search(r"[\u4e00-\u9fff]", pe_norm + mt_norm))

    mt_lines = [ln for ln in mt_norm.split("\n") if ln.strip()]
    pe_lines = [ln for ln in pe_norm.split("\n") if ln.strip()]
    has_speaker_dash = any(
        ln.startswith("-") or ln.startswith("—") for ln in pe_lines
    )
    if len(pe_lines) > 1 and has_speaker_dash:
        strategies.append(
            "Dual-Speaker Dialogue Line Break (雙語者重疊對話換行 - A / - B)"
        )
    elif len(pe_lines) != len(mt_lines):
        strategies.append(
            "Two-Line AVT Syntactic Line Breaking (影視雙行語法斷行優化)"
        )

    len_mt = len(mt_norm.replace("\n", ""))
    len_pe = len(pe_norm.replace("\n", ""))

    if len_mt > 0 and (len_mt - len_pe) / max(len_mt, 1) >= 0.14:
        strategies.append(
            "CPS Reading-Speed Condensation & Omission (字幕閱讀速率精簡與刪減冗詞)"
        )
    elif len_pe > len_mt * 1.18:
        strategies.append(
            "Contextual Explicitation & Zero-Pronoun Restoration (語境顯化與零代詞主語補全)"
        )

    if has_cjk_target:
        passive_markers = ["被", "受到", "加以", "進行", "作出"]
        if sum(pe_norm.count(m) for m in passive_markers) < sum(
            mt_norm.count(m) for m in passive_markers
        ):
            strategies.append(
                "E-C Passive-to-Active & Nominal-to-Verbal Shift (英譯中：去被動式與抽象名詞轉動詞)"
            )

        translationese_markers = [
            "的",
            "了",
            "們",
            "一個",
            "這是一個",
            "它是",
            "他的",
            "她的",
        ]
        if sum(pe_norm.count(m) for m in translationese_markers) < sum(
            mt_norm.count(m) for m in translationese_markers
        ):
            strategies.append(
                "E-C De-Translationese & Redundant Pronoun/Particle Pruning (英譯中：去歐化翻譯腔與省譯代詞／虛詞)"
            )

        if pe_norm.count("，") != mt_norm.count("，") or pe_norm.endswith(
            ("吧", "呢", "啊", "喔", "嗎")
        ):
            strategies.append(
                "E-C Parataxis Bamboo Clauses & Spoken Particle Polish (英譯中：化整為零流水句與口語語氣詞潤飾)"
            )
    else:
        connectives = {
            "because",
            "although",
            "while",
            "since",
            "unless",
            "therefore",
            "however",
            "when",
            "if",
            "which",
            "that",
        }
        mt_words = set(re.findall(r"\b[a-z]+\b", mt_norm.lower()))
        pe_words = set(re.findall(r"\b[a-z]+\b", pe_norm.lower()))
        if len(pe_words & connectives) > len(mt_words & connectives):
            strategies.append(
                "C-E Hypotaxis & Logical Connective Explicitation (中譯英：意合轉形合與邏輯連接詞顯化)"
            )

        subj_pronouns = {"i", "we", "you", "he", "she", "they", "it", "this"}
        if len(pe_words & subj_pronouns) > len(mt_words & subj_pronouns):
            strategies.append(
                "C-E Topic-Comment → SVO & Subject Restoration (中譯英：話題鏈轉主謂結構與補全主詞)"
            )

    mt_punc = set(re.findall(r"[，。！？；：,.!?;:—\-]", mt_norm))
    pe_punc = set(re.findall(r"[，。！？；：,.!?;:—\-]", pe_norm))
    if mt_punc != pe_punc and not has_speaker_dash:
        strategies.append(
            "AVT Punctuation & Clause Segmentation Reform (影視字幕標點簡化與斷句重組)"
        )

    ratio = difflib.SequenceMatcher(None, mt_norm, pe_norm).ratio()
    if 0.45 <= ratio < 0.98 and not strategies:
        strategies.append(
            "Lexical Precision & Spoken Register Naturalization (詞彙精準化與影視口語語域潤飾)"
        )
    elif ratio < 0.45:
        strategies.append(
            "Syntactic Restructuring & Cultural/Idiom Transcreation (句法結構重組與成語／文化道地創譯)"
        )

    return strategies or ["Micro-Stylistic AVT Polish (影視字幕微幅語氣潤飾)"]


def generate_pe_comparison_report(
    baseline_segments: List[Dict[str, Any]],
    current_segments: List[Dict[str, Any]],
) -> str:
    if not current_segments:
        return """
        <div style="padding:16px;border-radius:10px;background:#0f172a;color:#94a3b8;border:1px solid #334155;">
            No subtitle segments available yet.
        </div>
        """

    base_map = {
        int(s.get("id", idx + 1)): s
        for idx, s in enumerate(baseline_segments)
    }
    total = len(current_segments)
    edited_count = 0
    similarity_scores: List[float] = []
    strategy_counter: Dict[str, int] = {}
    rows_html: List[str] = []

    for idx, curr in enumerate(current_segments, start=1):
        b_seg = base_map.get(idx, curr)
        mt_tgt = normalize_multiline_text(b_seg.get("target", ""))
        pe_tgt = normalize_multiline_text(curr.get("target", ""))
        src_txt = normalize_multiline_text(curr.get("source", ""))

        sim = difflib.SequenceMatcher(None, mt_tgt, pe_tgt).ratio()
        similarity_scores.append(sim)
        is_edited = mt_tgt != pe_tgt
        if is_edited:
            edited_count += 1

        applied_strats = detect_pe_strategies(src_txt, mt_tgt, pe_tgt)
        for st in applied_strats:
            if "No Edit Needed" not in st:
                strategy_counter[st] = strategy_counter.get(st, 0) + 1

        badge = (
            '<span style="background:#ca8a04;color:#fff;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;">✏️ Post-Edited / 已修改</span>'
            if is_edited
            else '<span style="background:#334155;color:#cbd5e1;padding:2px 8px;border-radius:999px;font-size:11px;">✓ Unchanged / 未變更</span>'
        )
        strat_badges = " ".join(
            f'<span style="display:inline-block;background:#1e3a8a;color:#93c5fd;padding:2px 7px;border-radius:6px;font-size:11px;margin:2px 2px 0 0;">{html.escape(st)}</span>'
            for st in applied_strats
        )

        rows_html.append(
            f"""
            <tr style="border-bottom:1px solid #1e293b;background:{'rgba(234,179,8,0.07)' if is_edited else 'transparent'};">
                <td style="padding:8px;font-weight:700;color:#38bdf8;">#{idx}<br><span style="font-size:11px;color:#64748b;">{curr.get('start',0):.2f}s-{curr.get('end',0):.2f}s</span></td>
                <td style="padding:8px;color:#e2e8f0;font-size:13px;white-space:pre-line;">{html.escape(src_txt)}</td>
                <td style="padding:8px;color:#94a3b8;font-size:13px;white-space:pre-line;">{html.escape(mt_tgt)}</td>
                <td style="padding:8px;color:#fde047;font-size:13px;font-weight:600;white-space:pre-line;">{html.escape(pe_tgt)}</td>
                <td style="padding:8px;">{badge}<div style="margin-top:4px;">{strat_badges}</div></td>
            </tr>
            """
        )

    avg_edit_dist = round(
        (1.0 - (sum(similarity_scores) / max(total, 1))) * 100, 1
    )
    edit_rate = round((edited_count / max(total, 1)) * 100, 1)

    strat_summary_html = (
        "".join(
            f'<span style="background:#0284c7;color:#fff;padding:4px 10px;border-radius:999px;font-size:12px;margin-right:6px;display:inline-block;margin-bottom:4px;">{html.escape(k)}: <b>{v}</b></span>'
            for k, v in sorted(strategy_counter.items(), key=lambda x: -x[1])
        )
        or '<span style="color:#94a3b8;font-size:12px;">Edit, add, delete, or split any subtitle in the Workbench above and click "💾 Save & Apply Edits" to update your C-E / E-C strategy analytics and PDF report.</span>'
    )

    return f"""
    <div style="background:#0f172a;color:#f8fafc;padding:16px;border-radius:12px;border:1px solid #334155;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:12px;">
            <h3 style="margin:0;color:#38bdf8;font-size:16px;">📊 Post-Editing (MTPE) Summary & C-E / E-C Strategy Report / 譯後編輯與中英策略分析報告</h3>
            <div style="display:flex;gap:12px;font-size:13px;">
                <span style="background:#1e293b;padding:5px 12px;border-radius:8px;border:1px solid #475569;">
                    Segments Edited: <b style="color:#fde047;">{edited_count} / {total} ({edit_rate}%)</b>
                </span>
                <span style="background:#1e293b;padding:5px 12px;border-radius:8px;border:1px solid #475569;">
                    Avg. Edit Distance (HTER proxy): <b style="color:#38bdf8;">{avg_edit_dist}%</b>
                </span>
            </div>
        </div>
        <div style="margin-bottom:12px;padding:10px;background:#1e293b;border-radius:8px;">
            <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;font-weight:600;">🧭 Detected C-E / E-C Post-Editing Strategies Applied:</div>
            {strat_summary_html}
        </div>
        <div style="max-height:320px;overflow-y:auto;">
            <table style="width:100%;border-collapse:collapse;text-align:left;font-size:13px;">
                <thead>
                    <tr style="border-bottom:2px solid #334155;color:#94a3b8;font-size:12px;">
                        <th style="padding:8px;">#</th>
                        <th style="padding:8px;">Source (原文)</th>
                        <th style="padding:8px;">Raw MT Baseline (原始機譯)</th>
                        <th style="padding:8px;">Post-Edited Final (譯後定稿)</th>
                        <th style="padding:8px;">Status & C-E / E-C Strategy</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
    </div>
    """


def get_pdf_cjk_font() -> str:
    global PDF_CJK_FONT_NAME
    if PDF_CJK_FONT_NAME is not None:
        return PDF_CJK_FONT_NAME

    ensure_cjk_font_for_subtitles()
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        os.path.join(FONTS_CACHE_DIR, "NotoSansTC-Regular.otf"),
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                if path.endswith(".ttc"):
                    pdfmetrics.registerFont(
                        TTFont("NotoSansCJK", path, subfontIndex=0)
                    )
                else:
                    pdfmetrics.registerFont(TTFont("NotoSansCJK", path))
                PDF_CJK_FONT_NAME = "NotoSansCJK"
                return PDF_CJK_FONT_NAME
            except Exception:
                pass

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))
        PDF_CJK_FONT_NAME = "MSung-Light"
        return PDF_CJK_FONT_NAME
    except Exception:
        PDF_CJK_FONT_NAME = "Helvetica"
        return PDF_CJK_FONT_NAME


def generate_pe_report_pdf(
    baseline_segments: List[Dict[str, Any]],
    current_segments: List[Dict[str, Any]],
    pdf_path: str,
) -> str:
    font_name = get_pdf_cjk_font()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PETitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
    )
    meta_style = ParagraphStyle(
        "PEMeta",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    cell_style = ParagraphStyle(
        "PECell",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        wordWrap="CJK",
    )
    header_style = ParagraphStyle(
        "PEHeader",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9.5,
        leading=13,
        textColor=colors.white,
        wordWrap="CJK",
    )

    base_map = {
        int(s.get("id", idx + 1)): s
        for idx, s in enumerate(baseline_segments)
    }
    total = len(current_segments)
    edited_count = 0
    similarity_scores: List[float] = []
    strategy_counter: Dict[str, int] = {}

    table_data: List[List[Any]] = [
        [
            Paragraph("<b># / Time</b>", header_style),
            Paragraph("<b>Source Speech (原文)</b>", header_style),
            Paragraph("<b>Raw MT Baseline (原始機譯)</b>", header_style),
            Paragraph("<b>Post-Edited Final (譯後定稿)</b>", header_style),
            Paragraph(
                "<b>Status & Applied Strategies (譯後編輯策略)</b>", header_style
            ),
        ]
    ]

    def _fmt_multiline_pdf(txt: str) -> str:
        return html.escape(normalize_multiline_text(txt)).replace(
            "\n", "<br/>"
        )

    for idx, curr in enumerate(current_segments, start=1):
        b_seg = base_map.get(idx, curr)
        mt_tgt = normalize_multiline_text(b_seg.get("target", ""))
        pe_tgt = normalize_multiline_text(curr.get("target", ""))
        src_txt = normalize_multiline_text(curr.get("source", ""))

        sim = difflib.SequenceMatcher(None, mt_tgt, pe_tgt).ratio()
        similarity_scores.append(sim)
        is_edited = mt_tgt != pe_tgt
        if is_edited:
            edited_count += 1

        applied_strats = detect_pe_strategies(src_txt, mt_tgt, pe_tgt)
        for st in applied_strats:
            if "No Edit Needed" not in st:
                strategy_counter[st] = strategy_counter.get(st, 0) + 1

        status_label = "Edited / 已修改" if is_edited else "Unchanged / 未變更"
        strat_str = "<br/>".join(
            f"• {html.escape(st)}" for st in applied_strats
        )
        time_str = f"<b>#{idx}</b><br/>{curr.get('start', 0.0):.2f}s - {curr.get('end', 0.0):.2f}s"

        table_data.append(
            [
                Paragraph(time_str, cell_style),
                Paragraph(_fmt_multiline_pdf(src_txt), cell_style),
                Paragraph(_fmt_multiline_pdf(mt_tgt), cell_style),
                Paragraph(_fmt_multiline_pdf(pe_tgt), cell_style),
                Paragraph(f"<b>[{status_label}]</b><br/>{strat_str}", cell_style),
            ]
        )

    avg_edit_dist = round(
        (1.0 - (sum(similarity_scores) / max(total, 1))) * 100, 1
    )
    edit_rate = round((edited_count / max(total, 1)) * 100, 1)
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

    story: List[Any] = []
    story.append(
        Paragraph(
            "SubCraft-PE: Audiovisual Post-Editing (MTPE) Summary & C-E / E-C Strategy Report",
            title_style,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            f"<b>Generated At:</b> {timestamp_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Segments Edited:</b> {edited_count} / {total} ({edit_rate}%) &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Avg. Edit Distance (HTER Proxy):</b> {avg_edit_dist}%",
            meta_style,
        )
    )

    if strategy_counter:
        strat_summary = " &nbsp;•&nbsp; ".join(
            f"{html.escape(k)}: <b>{v}</b>"
            for k, v in sorted(strategy_counter.items(), key=lambda x: -x[1])
        )
    else:
        strat_summary = (
            "No post-editing modifications recorded yet (Raw MT accepted)."
        )

    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            f"<b>Detected C-E / E-C Post-Editing Strategies:</b> {strat_summary}",
            meta_style,
        )
    )
    story.append(Spacer(1, 4 * mm))

    col_widths = [24 * mm, 62 * mm, 62 * mm, 62 * mm, 63 * mm]
    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8fafc")],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(tbl)
    doc.build(story)
    return pdf_path


# ==============================================================================
# C-E / E-C Audiovisual Post-Editing Skills Curriculum & Reference Guide (Tab 2)
# ==============================================================================

CE_EC_STRATEGY_GUIDE_HTML = """
<div style="background:#0f172a;color:#f8fafc;padding:22px;border-radius:14px;border:1px solid #334155;font-family:system-ui,-apple-system,sans-serif;line-height:1.6;">
    <div style="border-bottom:1px solid #334155;padding-bottom:14px;margin-bottom:18px;">
        <h2 style="margin:0;color:#38bdf8;font-size:22px;">📚 SubCraft-PE: C-E &amp; E-C Audiovisual Post-Editing (MTPE) Skills &amp; Anti-Drift ASR Guide / 中英影視字幕譯後編輯核心技巧指南</h2>
        <p style="margin:6px 0 0 0;color:#94a3b8;font-size:14px;">
            Complete classroom handbook for English-to-Chinese (E-C) and Chinese-to-English (C-E) subtitle post-editing, reading-speed (CPS) control, and whole-context translation.
        </p>
    </div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:12px;margin-bottom:24px;">
        <div style="background:#1e293b;padding:14px;border-radius:10px;border-left:4px solid #34d399;">
            <div style="font-size:12px;color:#94a3b8;font-weight:700;">⚡ Default SOTA Distilled English</div>
            <div style="font-size:16px;font-weight:700;color:#f8fafc;margin-top:4px;">faster-distil-whisper-large-v3.5</div>
            <div style="font-size:12px;color:#cbd5e1;margin-top:4px;">Configured with <b>chunk_length=15</b> sequential windowing, peak audio normalization &amp; silence hallucination guard so accuracy stays high from start to end.</div>
        </div>
        <div style="background:#1e293b;padding:14px;border-radius:10px;border-left:4px solid #38bdf8;">
            <div style="font-size:12px;color:#94a3b8;font-weight:700;">🧠 Whole-Passage Context MT</div>
            <div style="font-size:16px;font-weight:700;color:#f8fafc;margin-top:4px;">Full Text → Split Back to Cues</div>
            <div style="font-size:12px;color:#cbd5e1;margin-top:4px;">Combines split subtitle cues into complete sentences &amp; discourse passages before translating, then proportionally maps translations back to each cue.</div>
        </div>
        <div style="background:#1e293b;padding:14px;border-radius:10px;border-left:4px solid #fde047;">
            <div style="font-size:12px;color:#94a3b8;font-weight:700;">⏱️ AVT Reading Speed Rules</div>
            <div style="font-size:16px;font-weight:700;color:#f8fafc;margin-top:4px;">CPS &amp; Line-Length Limits</div>
            <div style="font-size:12px;color:#cbd5e1;margin-top:4px;"><b>Chinese:</b> ≤ 14–16 chars/line (4–5 CPS)<br><b>English:</b> ≤ 42 chars/line (15–17 CPS)<br>Max <b>2 lines</b> per subtitle cue.</div>
        </div>
        <div style="background:#1e293b;padding:14px;border-radius:10px;border-left:4px solid #a855f7;">
            <div style="font-size:12px;color:#94a3b8;font-weight:700;">🔥 Guaranteed CJK ASS Hard-Burn</div>
            <div style="font-size:16px;font-weight:700;color:#f8fafc;margin-top:4px;">Bilingual &amp; Monolingual MP4</div>
            <div style="font-size:12px;color:#cbd5e1;margin-top:4px;">Hard-burned MP4 videos embed Noto Sans CJK / WenQuanYi fonts via <b>fontsdir</b> and honor your exact <b>Mode</b> and <b>Font Size</b>.</div>
        </div>
    </div>

    <h3 style="color:#fde047;font-size:18px;margin:0 0 12px 0;border-left:4px solid #fde047;padding-left:10px;">
        📘 Part I: English → Chinese (E-C) Subtitle Post-Editing Skills / 英譯中影視字幕譯後編輯六大技巧
    </h3>
    <div style="overflow-x:auto;margin-bottom:24px;">
        <table style="width:100%;border-collapse:collapse;font-size:13px;background:#1e293b;border-radius:10px;overflow:hidden;">
            <thead>
                <tr style="background:#0284c7;color:#ffffff;text-align:left;">
                    <th style="padding:10px 12px;width:22%;">E-C Skill / 譯後編輯技巧</th>
                    <th style="padding:10px 12px;width:24%;">English Source / 英文原句</th>
                    <th style="padding:10px 12px;width:27%;">❌ Raw MT (Translationese / 機譯腔)</th>
                    <th style="padding:10px 12px;width:27%;">✅ Post-Edited Subtitle (自然影視字幕)</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px 12px;font-weight:700;color:#38bdf8;">1. Nominal → Verbal Shift<br><span style="font-size:11px;color:#94a3b8;">抽象名詞轉動詞（化靜為動）</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">The implementation of this policy led to a significant improvement in safety.</td>
                    <td style="padding:10px 12px;color:#f87171;">這項政策的實施導致了安全性的顯著改善。</td>
                    <td style="padding:10px 12px;color:#4ade80;font-weight:600;">這項政策落實後，安全大幅提升。</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;background:rgba(15,23,42,0.4);">
                    <td style="padding:10px 12px;font-weight:700;color:#38bdf8;">2. Passive → Active Voice<br><span style="font-size:11px;color:#94a3b8;">被動轉主動／隱去施事者</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">The final decision will be announced by the committee tomorrow.</td>
                    <td style="padding:10px 12px;color:#f87171;">最終決定將於明天被委員會宣布。</td>
                    <td style="padding:10px 12px;color:#4ade80;font-weight:600;">委員會明天將公布最終決定。</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px 12px;font-weight:700;color:#38bdf8;">3. Pronoun &amp; Determiner Pruning<br><span style="font-size:11px;color:#94a3b8;">省譯冗餘代名詞與冠詞（去「的／一個」）</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">He put his hand into his pocket and took out a small notebook.</td>
                    <td style="padding:10px 12px;color:#f87171;">他把他的手放進他的口袋裡，拿出了一個小筆記本。</td>
                    <td style="padding:10px 12px;color:#4ade80;font-weight:600;">他把手伸進口袋，掏出小筆記本。</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;background:rgba(15,23,42,0.4);">
                    <td style="padding:10px 12px;font-weight:700;color:#38bdf8;">4. Parataxis Bamboo Clauses<br><span style="font-size:11px;color:#94a3b8;">英文複合長句拆解為中文流水句</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">Wait until the rain stops before you go out, or you'll catch a cold.</td>
                    <td style="padding:10px 12px;color:#f87171;">在你出去之前等到雨停，否則你將會感冒。</td>
                    <td style="padding:10px 12px;color:#4ade80;font-weight:600;">等雨停了再出門，免得著涼。</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px 12px;font-weight:700;color:#38bdf8;">5. Spoken Register &amp; Modal Particles<br><span style="font-size:11px;color:#94a3b8;">影視口語語域與語氣助詞潤飾</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">You can't be serious! We've already tried that three times.</td>
                    <td style="padding:10px 12px;color:#f87171;">你不能是認真的！我們已經嘗試過那個三次了。</td>
                    <td style="padding:10px 12px;color:#4ade80;font-weight:600;">別開玩笑了！我們都試過三次了。</td>
                </tr>
                <tr>
                    <td style="padding:10px 12px;font-weight:700;color:#38bdf8;">6. CPS Reading-Speed Condensation<br><span style="font-size:11px;color:#94a3b8;">字幕閱讀速率精簡（刪減套話與贅字）</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">As a matter of fact, what I'm trying to say is that we need more time.</td>
                    <td style="padding:10px 12px;color:#f87171;">事實上，我試圖要說的事情是我們需要更多的時間。</td>
                    <td style="padding:10px 12px;color:#4ade80;font-weight:600;">其實我們還需要更多時間。</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h3 style="color:#34d399;font-size:18px;margin:0 0 12px 0;border-left:4px solid #34d399;padding-left:10px;">
        📗 Part II: Chinese → English (C-E) Subtitle Post-Editing Skills / 中譯英影視字幕譯後編輯六大技巧
    </h3>
    <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:13px;background:#1e293b;border-radius:10px;overflow:hidden;">
            <thead>
                <tr style="background:#059669;color:#ffffff;text-align:left;">
                    <th style="padding:10px 12px;width:22%;">C-E Skill / 譯後編輯技巧</th>
                    <th style="padding:10px 12px;width:24%;">Chinese Source / 中文原句</th>
                    <th style="padding:10px 12px;width:27%;">❌ Raw MT (Broken / Run-on / 機譯直譯)</th>
                    <th style="padding:10px 12px;width:27%;">✅ Post-Edited Subtitle (道地英文字幕)</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px 12px;font-weight:700;color:#34d399;">1. Topic-Comment → SVO Structure<br><span style="font-size:11px;color:#94a3b8;">話題鏈轉英文主謂賓結構</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">這場專案的預算，昨天開會已經全部批下來了。</td>
                    <td style="padding:10px 12px;color:#f87171;">This project's budget, yesterday meeting already approved all.</td>
                    <td style="padding:10px 12px;color:#fde047;font-weight:600;">The full project budget was approved at yesterday's meeting.</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;background:rgba(15,23,42,0.4);">
                    <td style="padding:10px 12px;font-weight:700;color:#34d399;">2. Zero-Pronoun Subject Restoration<br><span style="font-size:11px;color:#94a3b8;">補全中文省略的主詞與受詞</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">聽說下週要出差，還沒訂機票呢。</td>
                    <td style="padding:10px 12px;color:#f87171;">Heard going on a business trip next week, haven't booked tickets yet.</td>
                    <td style="padding:10px 12px;color:#fde047;font-weight:600;">I heard you're traveling next week, but haven't booked flights yet.</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px 12px;font-weight:700;color:#34d399;">3. Hypotaxis &amp; Logical Connectives<br><span style="font-size:11px;color:#94a3b8;">意合轉形合：顯化隱含因果／轉折連接詞</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">現在才出發，過去早就關門了。</td>
                    <td style="padding:10px 12px;color:#f87171;">Leave right now, go there already closed.</td>
                    <td style="padding:10px 12px;color:#fde047;font-weight:600;">If we leave now, it'll be closed by the time we get there.</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;background:rgba(15,23,42,0.4);">
                    <td style="padding:10px 12px;font-weight:700;color:#34d399;">4. Serial Verbs → Main Verb + Prep<br><span style="font-size:11px;color:#94a3b8;">中文連謂結構化為單一核心動詞＋介詞片語</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">我們打電話去問問看能不能改期。</td>
                    <td style="padding:10px 12px;color:#f87171;">Let's make a phone call to go ask and see if can reschedule.</td>
                    <td style="padding:10px 12px;color:#fde047;font-weight:600;">Let's call to see if we can reschedule.</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:10px 12px;font-weight:700;color:#34d399;">5. Four-Character Idiom Transcreation<br><span style="font-size:11px;color:#94a3b8;">四字成語與文化俗諺道地創譯</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">別再臨時抱佛腳了，趕快開始準備吧！</td>
                    <td style="padding:10px 12px;color:#f87171;">Stop hugging Buddha's feet at the last minute, start preparing!</td>
                    <td style="padding:10px 12px;color:#fde047;font-weight:600;">Stop cramming at the last minute—start preparing now!</td>
                </tr>
                <tr>
                    <td style="padding:10px 12px;font-weight:700;color:#34d399;">6. Syntactic Line-Break Integrity<br><span style="font-size:11px;color:#94a3b8;">影視雙行斷行不拆散名詞組或介系詞片語</span></td>
                    <td style="padding:10px 12px;color:#e2e8f0;">我們必須在截止日期前提交這份完整報告。</td>
                    <td style="padding:10px 12px;color:#f87171;">We must submit this complete<br>report before the final deadline.</td>
                    <td style="padding:10px 12px;color:#fde047;font-weight:600;">We must submit this complete report<br>before the final deadline.</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
"""


# ==============================================================================
# Core Pipeline Actions (Live Streaming + Anti-Drift ASR + Whole-Text Context MT)
# ==============================================================================


def generate_all_export_files(
    current_segments: List[Dict[str, Any]],
    baseline_segments: List[Dict[str, Any]],
    presentation_mode: str,
    font_size: int,
) -> Tuple[str, str, str, str, str]:
    out_dir = tempfile.mkdtemp(prefix="bilingual_exports_")
    orig_srt_path = write_srt_file(
        current_segments,
        os.path.join(out_dir, "original_source.srt"),
        mode="source_only",
    )
    bilingual_srt_path = write_srt_file(
        current_segments,
        os.path.join(out_dir, "post_edited_subtitles.srt"),
        mode=presentation_mode,
    )
    bilingual_vtt_path = write_vtt_file(
        current_segments,
        os.path.join(out_dir, "bilingual_presentation.vtt"),
        mode=presentation_mode,
    )
    pdf_report_path = generate_pe_report_pdf(
        baseline_segments,
        current_segments,
        os.path.join(out_dir, "post_editing_summary_report.pdf"),
    )
    json_backup_path = save_project_checkpoint_json(
        current_segments,
        baseline_segments,
        presentation_mode,
        font_size,
        out_dir,
    )
    return (
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
    )


def run_step1_pipeline(
    video_input: Optional[str],
    start_sec: float,
    end_sec: float,
    whisper_size: str,
    mt_engine_choice: str,
    translation_mode_choice: str,
    src_lang_choice: str,
    tgt_lang_choice: str,
    presentation_mode: str,
    font_size: int,
) -> Generator[Tuple[Any, ...], None, None]:
    global FORCE_CPU, WHISPER_MODEL, WHISPER_BATCHED, WHISPER_KEY
    t0 = time.time()
    logs: List[str] = []

    def log_line(msg: str) -> None:
        elapsed = time.time() - t0
        line = f"[{elapsed:6.1f}s] {msg}"
        logs.append(line)
        print(line, flush=True)

    if not video_input or not os.path.exists(video_input):
        raise gr.Error(
            "Please upload a video or load a Classroom Sample Clip first! / 請先上傳影片或載入課堂範例影片！"
        )

    effective_asr_model = whisper_size
    if (
        whisper_size in ENGLISH_ONLY_MODELS or whisper_size.endswith(".en")
    ) and src_lang_choice not in ("auto", "en-US"):
        log_line(
            f"Note: '{whisper_size}' is an English-specialized distilled model; switching to 'large-v3' for {src_lang_choice}."
        )
        effective_asr_model = "large-v3"

    log_line(
        f"🔄 Starting full pipeline run with ASR='{effective_asr_model}', MT='{mt_engine_choice}' ({translation_mode_choice}), Lang={src_lang_choice}→{tgt_lang_choice}"
    )

    rebuilding_wb_html = f"""
    <div style="padding:26px;border:2px solid #0284c7;border-radius:14px;background:#0f172a;color:#f8fafc;text-align:center;font-family:system-ui,-apple-system,sans-serif;">
        <h3 style="margin:0 0 8px 0;color:#38bdf8;">🔄 Running Anti-Drift ASR + Whole-Passage Context Translation...</h3>
        <p style="margin:0;color:#cbd5e1;font-size:14px;">
            Active ASR Model: <b>{html.escape(effective_asr_model)}</b> &nbsp;|&nbsp;
            Active MT Engine: <b>{html.escape(mt_engine_choice)}</b> ({html.escape(translation_mode_choice)}) &nbsp;|&nbsp;
            Languages: <b>{html.escape(src_lang_choice)} → {html.escape(tgt_lang_choice)}</b>
        </p>
    </div>
    """
    init_banner = build_dynamic_status_banner(
        "running",
        f"Running Pipeline ({effective_asr_model} + {mt_engine_choice})...",
        "Clearing previous results and starting fresh video clip preparation, Anti-Drift ASR & Whole-Passage Translation",
        pct=4,
        elapsed_sec=0.1,
    )
    yield (
        None,
        "",
        "",
        rebuilding_wb_html,
        "",
        None,
        None,
        None,
        None,
        None,
        "\n".join(logs),
        init_banner,
        "",
    )

    def ui_progress(title: str, subtitle: str, pct: int) -> Tuple[Any, ...]:
        elapsed = time.time() - t0
        banner = build_dynamic_status_banner(
            "running", title, subtitle, pct=pct, elapsed_sec=elapsed
        )
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "\n".join(logs),
            banner,
            gr.skip(),
        )

    full_dur = get_video_duration(video_input)
    s_sec = max(0.0, float(start_sec or 0.0))
    e_sec = float(end_sec or 0.0)
    if e_sec <= s_sec and full_dur > 0:
        e_sec = full_dur
    clip_dur = max(1.0, e_sec - s_sec)

    log_line(
        f"Preparing H.264 video clip ({s_sec:.2f}s – {e_sec:.2f}s of {full_dur:.2f}s) / 準備影片片段"
    )
    yield ui_progress(
        "Step 0/2: Preparing Video Clip...",
        f"Checking stream copy / GPU encoder for {clip_dur:.1f}s video",
        pct=10,
    )

    prepared_video = prepare_h264_clip(
        video_input, s_sec, e_sec, log_fn=log_line
    )
    log_line("Prepared clip ready / 影片片段已就緒")
    yield ui_progress(
        f"Step 1/2: Loading SOTA ASR '{effective_asr_model}' on T4 GPU...",
        "Initializing CTranslate2 float16 Anti-Drift Sequential Pipeline",
        pct=18,
    )

    model, _, device = load_whisper_model(effective_asr_model, log_line)
    yield ui_progress(
        "Step 1/2: Extracting & Peak-Normalizing 16kHz Audio...",
        "Applying bandpass filter + loudness normalization so late-video audio stays crystal clear",
        pct=26,
    )

    log_line(
        "Extracting & peak-normalizing 16kHz PCM audio with FFmpeg / 提取並正規化 16kHz 音訊"
    )
    audio_np = decode_audio_ffmpeg_np(prepared_video, sample_rate=16000)

    opt_chunk_len = (
        15 if effective_asr_model in DISTIL_SHORT_WINDOW_MODELS else 25
    )

    yield ui_progress(
        f"Step 1/2: Transcribing with '{effective_asr_model}' ({device.upper()}, {opt_chunk_len}s Window)...",
        "Anti-drift sequential decoding + temperature fallback + hallucination guard active",
        pct=34,
    )

    iso_lang = WHISPER_ISO_MAP.get(src_lang_choice, None)
    if (
        effective_asr_model in ENGLISH_ONLY_MODELS
        or effective_asr_model.endswith(".en")
    ):
        iso_lang = "en"

    vad_params = dict(
        threshold=0.45,
        min_speech_duration_ms=150,
        max_speech_duration_s=float(opt_chunk_len),
        min_silence_duration_ms=250,
        speech_pad_ms=150,
    )

    segments: List[Dict[str, Any]] = []
    detected_iso = "en"

    try:
        segments_gen, info = model.transcribe(
            audio_np,
            language=iso_lang,
            beam_size=5,
            best_of=5,
            patience=1.2,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=0.5,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            word_timestamps=True,
            chunk_length=opt_chunk_len,
            hallucination_silence_threshold=1.5,
            vad_filter=True,
            vad_parameters=vad_params,
        )
        detected_iso = info.language

        for s in segments_gen:
            if getattr(s, "no_speech_prob", 0.0) > 0.80:
                continue
            refined_chunks = split_segment_by_words(
                s,
                source_lang_code=src_lang_choice,
                max_pause_sec=0.50,
                max_duration_sec=4.5,
                max_chars_latin=65,
                max_chars_cjk=28,
            )
            for ch in refined_chunks:
                ch["id"] = len(segments) + 1
                segments.append(ch)
            log_line(
                f"ASR [{effective_asr_model}] ({opt_chunk_len}s Anti-Drift Window): {s.end:.1f}s / {clip_dur:.1f}s ({len(segments)} cues)"
            )
            sub_pct = 35 + int(min(1.0, s.end / clip_dur) * 30)
            yield ui_progress(
                f"Step 1/2: '{effective_asr_model}' Transcribing ({s.end:.1f}s / {clip_dur:.1f}s)...",
                f"Created {len(segments)} sentence-aligned subtitle cues so far",
                pct=sub_pct,
            )
    except Exception as e:
        log_line(f"ASR fallback to CPU standard transcribe: {e}")
        FORCE_CPU = True
        release_whisper_vram()
        model_cpu, _, _ = load_whisper_model(effective_asr_model, log_line)
        segments_gen, info = model_cpu.transcribe(
            audio_np,
            language=iso_lang,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            chunk_length=opt_chunk_len,
            vad_filter=True,
            vad_parameters=vad_params,
        )
        detected_iso = info.language
        for s in segments_gen:
            if getattr(s, "no_speech_prob", 0.0) > 0.80:
                continue
            refined_chunks = split_segment_by_words(
                s,
                source_lang_code=src_lang_choice,
                max_pause_sec=0.50,
                max_duration_sec=4.5,
                max_chars_latin=65,
                max_chars_cjk=28,
            )
            for ch in refined_chunks:
                ch["id"] = len(segments) + 1
                segments.append(ch)
            log_line(
                f"ASR [{effective_asr_model}] (Smart AVT Split): {s.end:.1f}s / {clip_dur:.1f}s ({len(segments)} cues)"
            )
            yield ui_progress(
                f"Step 1/2: '{effective_asr_model}' Transcribing ({s.end:.1f}s / {clip_dur:.1f}s)...",
                f"Created {len(segments)} sentence-aligned subtitle cues",
                pct=55,
            )

    resolved_src = (
        DETECTED_TO_LANG_CODE.get(detected_iso, "en-US")
        if src_lang_choice == "auto"
        else src_lang_choice
    )

    total_segs = len(segments)
    if total_segs > 0:
        log_line(
            f"Step 2/2: Translating {total_segs} cues via '{mt_engine_choice}' ({translation_mode_choice}) [{resolved_src} → {tgt_lang_choice}]"
        )
        yield ui_progress(
            f"Step 2/2: Whole-Passage Context Translation ({total_segs} Cues via {mt_engine_choice})...",
            f"Translating full sentences & discourse passages first, then splitting back to each cue [{resolved_src} → {tgt_lang_choice}]",
            pct=74,
        )

        segments = translate_segments_full_context(
            segments=segments,
            src_lang=resolved_src,
            tgt_lang=tgt_lang_choice,
            mt_engine=mt_engine_choice,
            translation_mode=translation_mode_choice,
            log_fn=log_line,
        )
        log_line(
            f"✅ Context-aware translation complete & aligned back to all {total_segs} cues!"
        )
    else:
        log_line("No spoken words detected in clip / 影片中未偵測到人聲")

    (
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
    ) = generate_all_export_files(
        segments, segments, presentation_mode, int(font_size or 20)
    )

    segments_json = json.dumps(segments, ensure_ascii=False)
    wb_html = build_bilingual_workbench_html(
        prepared_video,
        segments,
        presentation_mode,
        font_size=int(font_size or 20),
        model_badge=f"ASR: {effective_asr_model} • MT: {mt_engine_choice} ({translation_mode_choice})",
    )
    report_html = generate_pe_comparison_report(segments, segments)

    total_elapsed = time.time() - t0
    log_line(
        f"Ready! {len(segments)} cues transcribed ({effective_asr_model}) & context-translated ({mt_engine_choice}) in {total_elapsed:.1f}s!"
    )
    done_banner = build_dynamic_status_banner(
        "done",
        f"Ready! {len(segments)} Cues ({effective_asr_model} + Whole-Context {mt_engine_choice})",
        "Practice C-E / E-C post-editing in the Workbench below, then click Step 2 or Burn MP4!",
        pct=100,
        elapsed_sec=total_elapsed,
    )

    yield (
        prepared_video,
        segments_json,
        segments_json,
        wb_html,
        report_html,
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
        "\n".join(logs),
        done_banner,
        "",
    )


def retranslate_existing_cues(
    wb_json_str: str,
    mt_engine_choice: str,
    translation_mode_choice: str,
    src_lang_choice: str,
    tgt_lang_choice: str,
    presentation_mode: str,
    font_size: int,
    current_json_str: str,
    prepared_video: Optional[str],
) -> Generator[Tuple[Any, ...], None, None]:
    t0 = time.time()
    segments, chosen_mode, chosen_font_size = parse_workbench_or_state_segments(
        wb_json_str, current_json_str, presentation_mode, font_size
    )
    if not segments:
        raise gr.Error(
            "No subtitle segments to re-translate! Please run Step 1 first."
        )

    resolved_src = "en-US" if src_lang_choice == "auto" else src_lang_choice
    total_segs = len(segments)

    yield (
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        f"[  0.1s] 🔄 Re-translating {total_segs} cues with '{mt_engine_choice}' ({translation_mode_choice})...",
        build_dynamic_status_banner(
            "running",
            f"Re-Translating {total_segs} Cues with {mt_engine_choice} ({translation_mode_choice})...",
            f"[{resolved_src} → {tgt_lang_choice}] (Whole-passage context translation active)",
            pct=50,
            elapsed_sec=0.1,
        ),
    )

    segments = translate_segments_full_context(
        segments=segments,
        src_lang=resolved_src,
        tgt_lang=tgt_lang_choice,
        mt_engine=mt_engine_choice,
        translation_mode=translation_mode_choice,
    )

    (
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
    ) = generate_all_export_files(
        segments, segments, chosen_mode, chosen_font_size
    )

    segments_json = json.dumps(segments, ensure_ascii=False)
    wb_html = build_bilingual_workbench_html(
        prepared_video,
        segments,
        chosen_mode,
        font_size=chosen_font_size,
        model_badge=f"MT Re-Translated: {mt_engine_choice} ({translation_mode_choice})",
    )
    report_html = generate_pe_comparison_report(segments, segments)
    elapsed = time.time() - t0
    msg = f"[{elapsed:6.1f}s] ✅ Re-translated all {total_segs} cues using '{mt_engine_choice}' ({translation_mode_choice})!"
    done_banner = build_dynamic_status_banner(
        "done",
        f"Re-Translated {total_segs} Cues via {mt_engine_choice} in {elapsed:.1f}s!",
        "Whole-passage context translation and Workbench target subtitles have been updated.",
        pct=100,
        elapsed_sec=elapsed,
    )

    yield (
        segments_json,
        segments_json,
        wb_html,
        report_html,
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
        msg,
        done_banner,
    )


def parse_workbench_or_state_segments(
    wb_json_str: str,
    current_json_str: str,
    presentation_mode: str,
    font_size: int,
) -> Tuple[List[Dict[str, Any]], str, int]:
    """
    Collects segments from the HTML Workbench if present, and merges with `current_json_str`
    so that if any target/source text was left empty or unsynced, it falls back safely.
    """
    state_segments: List[Dict[str, Any]] = []
    if current_json_str:
        try:
            state_segments = json.loads(current_json_str)
        except Exception:
            state_segments = []

    segments: List[Dict[str, Any]] = []
    chosen_mode = presentation_mode or "bilingual_target_top"
    chosen_font_size = int(font_size or 20)

    if wb_json_str and wb_json_str.strip():
        try:
            parsed = json.loads(wb_json_str)
            if isinstance(parsed, dict) and "segments" in parsed:
                raw_segs = parsed.get("segments", [])
                if parsed.get("presentation_mode"):
                    chosen_mode = parsed["presentation_mode"]
                if parsed.get("font_size"):
                    chosen_font_size = int(parsed["font_size"])
                for idx, item in enumerate(raw_segs):
                    src_t = normalize_multiline_text(item.get("source", ""))
                    tgt_t = normalize_multiline_text(item.get("target", ""))
                    # If target is unexpectedly empty in DOM, fallback to state_segments target if available
                    if not tgt_t and idx < len(state_segments):
                        tgt_t = normalize_multiline_text(
                            state_segments[idx].get("target", "")
                        )
                    if not src_t and idx < len(state_segments):
                        src_t = normalize_multiline_text(
                            state_segments[idx].get("source", "")
                        )
                    if not src_t and not tgt_t:
                        continue
                    s_val = round(
                        parse_flexible_time(item.get("start", 0.0)), 2
                    )
                    e_val = round(
                        max(
                            s_val + 0.15,
                            parse_flexible_time(item.get("end", s_val + 0.8)),
                        ),
                        2,
                    )
                    segments.append(
                        {
                            "id": len(segments) + 1,
                            "start": s_val,
                            "end": e_val,
                            "source": src_t,
                            "target": tgt_t,
                        }
                    )
        except Exception:
            segments = []

    if not segments:
        segments = state_segments

    return segments, chosen_mode, chosen_font_size


def apply_html_workbench_edits(
    wb_json_str: str,
    presentation_mode: str,
    font_size: int,
    current_json_str: str,
    baseline_json_str: str,
    prepared_video: Optional[str],
):
    segments, chosen_mode, chosen_font_size = parse_workbench_or_state_segments(
        wb_json_str, current_json_str, presentation_mode, font_size
    )

    baseline_segments: List[Dict[str, Any]] = []
    if baseline_json_str:
        try:
            baseline_segments = json.loads(baseline_json_str)
        except Exception:
            baseline_segments = copy.deepcopy(segments)
    else:
        baseline_segments = copy.deepcopy(segments)

    (
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
    ) = generate_all_export_files(
        segments, baseline_segments, chosen_mode, chosen_font_size
    )

    updated_current_json = json.dumps(segments, ensure_ascii=False)
    wb_html = build_bilingual_workbench_html(
        prepared_video,
        segments,
        chosen_mode,
        font_size=chosen_font_size,
        model_badge="Post-Edited Session Saved",
    )
    report_html = generate_pe_comparison_report(baseline_segments, segments)
    stamp = time.strftime("%H:%M:%S")
    mode_human = PRESENTATION_MODE_LABELS.get(chosen_mode, chosen_mode)
    status_msg = (
        f"[{stamp}] ✅ Saved & Applied Workbench edits ({len(segments)} segments, "
        f"Mode='{mode_human}', Font={chosen_font_size}px, SRT/VTT/PDF/JSON updated)!"
    )
    done_banner = build_dynamic_status_banner(
        "done",
        f"Saved & Applied Edits ({len(segments)} segments • {mode_human})",
        "All SRT, VTT, C-E / E-C PDF Summary Report & JSON Checkpoint files are ready for download below!",
        pct=100,
        elapsed_sec=0.1,
    )

    return (
        updated_current_json,
        chosen_mode,
        chosen_font_size,
        wb_html,
        report_html,
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
        status_msg,
        done_banner,
    )


def restore_from_uploaded_or_server_backup(
    uploaded_backup_file: Optional[str],
    prepared_video: Optional[str],
    video_input: Optional[str],
):
    target_path = None
    if uploaded_backup_file and os.path.exists(uploaded_backup_file):
        target_path = uploaded_backup_file
    elif os.path.exists(AUTO_BACKUP_PATH):
        target_path = AUTO_BACKUP_PATH

    if not target_path:
        raise gr.Error(
            "No backup file found! Upload a saved `.json` project backup file or run Step 1 first."
        )

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise gr.Error(f"Failed to read backup JSON file: {e}")

    segments = data.get("segments", [])
    baseline_segments = data.get(
        "baseline_segments", copy.deepcopy(segments)
    )
    chosen_mode = data.get("presentation_mode", "bilingual_target_top")
    chosen_font_size = int(data.get("font_size", 20))
    saved_at = data.get("saved_at", "previous session")

    active_video = prepared_video
    if (not active_video or not os.path.exists(active_video)) and video_input:
        if os.path.exists(video_input):
            active_video = prepare_h264_clip(video_input, 0.0, 0.0)

    (
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
    ) = generate_all_export_files(
        segments, baseline_segments, chosen_mode, chosen_font_size
    )

    current_json = json.dumps(segments, ensure_ascii=False)
    baseline_json = json.dumps(baseline_segments, ensure_ascii=False)

    wb_html = build_bilingual_workbench_html(
        active_video,
        segments,
        chosen_mode,
        font_size=chosen_font_size,
        model_badge=f"Restored Backup ({saved_at})",
    )
    report_html = generate_pe_comparison_report(baseline_segments, segments)
    status_msg = f"✅ Restored {len(segments)} subtitle segments from backup ({saved_at})!"
    done_banner = build_dynamic_status_banner(
        "done",
        f"Restored Project Backup ({len(segments)} segments from {saved_at})",
        "Your post-edited subtitles, baseline MT, and PDF report have been fully restored.",
        pct=100,
        elapsed_sec=0.1,
    )

    return (
        active_video,
        current_json,
        baseline_json,
        chosen_mode,
        chosen_font_size,
        wb_html,
        report_html,
        orig_srt_path,
        bilingual_srt_path,
        bilingual_vtt_path,
        pdf_report_path,
        json_backup_path,
        status_msg,
        done_banner,
    )


def escape_ffmpeg_filter_path(p: str) -> str:
    """Escapes file/directory paths for FFmpeg `subtitles` / `ass` filter arguments."""
    return (
        p.replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def burn_subtitles_to_mp4(
    wb_json_str: str,
    presentation_mode: str,
    font_size: int,
    current_json_str: str,
    baseline_json_str: str,
    prepared_video: Optional[str],
) -> Generator[Tuple[Any, ...], None, None]:
    """
    Hard-burns subtitles into the MP4 video with 100% CJK + Multilingual visibility:
      1. Resolves/installs/downloads a true CJK font (`Noto Sans CJK TC` or `WenQuanYi Zen Hei`)
         via `ensure_cjk_font_for_subtitles()` and passes `fontsdir` to FFmpeg `subtitles` filter.
      2. Writes an `.ass` file (`write_ass_file`) using explicit inline override tags (`\fn`, `\fs`,
         `\1c`, `\b`) and `BorderStyle=1` thick outline + shadow so `libass` renders both Translated
         and Original lines with zero missing glyphs and zero box-clipping.
      3. Syncs any Workbench edits back to `current_json_state` at the same time.
    """
    if not prepared_video or not os.path.exists(prepared_video):
        raise gr.Error(
            "Please run Step 1 first so a video clip is ready! / 請先執行步驟 1！"
        )

    t0 = time.time()
    segments, chosen_mode, chosen_font_size = parse_workbench_or_state_segments(
        wb_json_str, current_json_str, presentation_mode, font_size
    )
    if not segments:
        raise gr.Error(
            "No subtitle segments found to burn! Please run Step 1 first."
        )

    font_family, fonts_dir = ensure_cjk_font_for_subtitles()
    mode_human = PRESENTATION_MODE_LABELS.get(chosen_mode, chosen_mode)

    non_empty_targets = sum(
        1 for s in segments if normalize_multiline_text(s.get("target", ""))
    )
    non_empty_sources = sum(
        1 for s in segments if normalize_multiline_text(s.get("source", ""))
    )

    out_dir = tempfile.mkdtemp(prefix="burned_bilingual_")
    ass_path = os.path.join(out_dir, "burn_aligned.ass")
    write_ass_file(
        segments,
        ass_path,
        mode=chosen_mode,
        font_size=int(chosen_font_size),
        font_family=font_family,
    )

    # Also write a mode-aligned SRT file as secondary fallback
    srt_fallback_path = os.path.join(out_dir, "burn_aligned.srt")
    write_srt_file(segments, srt_fallback_path, mode=chosen_mode)

    mode_slug = chosen_mode.replace("_", "-")
    burned_mp4 = os.path.join(
        out_dir, f"subcraft_burned_{mode_slug}_{int(chosen_font_size)}px.mp4"
    )

    escaped_ass = escape_ffmpeg_filter_path(ass_path)
    escaped_fonts_dir = escape_ffmpeg_filter_path(fonts_dir)
    vf_ass_arg = f"subtitles=filename='{escaped_ass}':fontsdir='{escaped_fonts_dir}'"

    use_nvenc = has_nvenc_support()
    enc_tag = "GPU h264_nvenc" if use_nvenc else "CPU libx264 ultrafast"

    updated_current_json = json.dumps(segments, ensure_ascii=False)

    yield (
        gr.skip(),
        gr.skip(),
        updated_current_json,
        chosen_mode,
        chosen_font_size,
        (
            f"[  0.1s] 🔥 Starting FFmpeg Hard-Burn | Mode: {mode_human} | "
            f"Font: {font_family} ({chosen_font_size}px) | "
            f"Cues: {non_empty_targets} translated / {non_empty_sources} source | Encoder: {enc_tag}"
        ),
        build_dynamic_status_banner(
            "running",
            f"Burning Subtitles into MP4 ({mode_human})...",
            f"Font: {font_family} ({chosen_font_size}px) | {non_empty_targets} Translated & {non_empty_sources} Source Cues | {enc_tag}",
            pct=40,
            elapsed_sec=0.1,
        ),
    )

    v_codec = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
        if use_nvenc
        else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        prepared_video,
        "-vf",
        vf_ass_arg,
        *v_codec,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        burned_mp4,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Fallback 1: If NVENC failed, retry with CPU libx264 + ASS + fontsdir
    if res.returncode != 0 and use_nvenc:
        cmd_cpu = [
            "ffmpeg",
            "-y",
            "-i",
            prepared_video,
            "-vf",
            vf_ass_arg,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            burned_mp4,
        ]
        res = subprocess.run(
            cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    # Fallback 2: If older FFmpeg rejects `subtitles=filename=...:fontsdir=...`, use `ass=...` or `subtitles=...` with force_style
    if res.returncode != 0 or not os.path.exists(burned_mp4):
        escaped_srt = escape_ffmpeg_filter_path(srt_fallback_path)
        force_fs = max(16, int(round(int(chosen_font_size or 20) * 1.15)))
        vf_srt_arg = (
            f"subtitles='{escaped_srt}':fontsdir='{escaped_fonts_dir}':"
            f"force_style='FontName={font_family},FontSize={force_fs},PrimaryColour=&H0047E0FD,OutlineColour=&H00170602,BorderStyle=1,Outline=2.5,Shadow=1.2,Alignment=2,MarginV=28'"
        )
        cmd_srt = [
            "ffmpeg",
            "-y",
            "-i",
            prepared_video,
            "-vf",
            vf_srt_arg,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            burned_mp4,
        ]
        subprocess.run(
            cmd_srt, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    elapsed = time.time() - t0
    msg = (
        f"[{elapsed:6.1f}s] 🔥 Burned MP4 complete! Mode: {mode_human} | "
        f"Font: {font_family} ({chosen_font_size}px) | "
        f"Rendered {non_empty_targets} translated & {non_empty_sources} source cues | Encoder: {enc_tag}"
    )
    done_banner = build_dynamic_status_banner(
        "done",
        f"Burned-in MP4 Ready ({mode_human})!",
        f"Completed in {elapsed:.1f}s ({font_family} @ {chosen_font_size}px • {non_empty_targets} Translated Cues Rendered)",
        pct=100,
        elapsed_sec=elapsed,
    )
    yield (
        burned_mp4,
        burned_mp4,
        updated_current_json,
        chosen_mode,
        chosen_font_size,
        msg,
        done_banner,
    )


def on_video_upload(
    video_path: Optional[str],
) -> Tuple[float, float, str, str]:
    if not video_path or not os.path.exists(video_path):
        return (
            0.0,
            0.0,
            "Waiting for video upload or sample selection...",
            build_dynamic_status_banner(
                "idle",
                "Waiting for Video Upload or Sample Clip",
                "Upload an MP4/MOV/MKV video or load a Classroom Sample Clip above.",
                pct=0,
            ),
        )
    dur = get_video_duration(video_path)
    msg = f"✅ Video loaded ({os.path.basename(video_path)})! Full duration: {dur:.2f}s. Clip range set to 0.00s – {dur:.2f}s."
    banner = build_dynamic_status_banner(
        "idle",
        f"Video Loaded ({dur:.2f}s Full Duration Auto-Selected)",
        "Click '🚀 Step 1: Transcribe & Translate Video' to start live T4 GPU processing.",
        pct=100,
    )
    return 0.0, dur, msg, banner


def load_selected_sample_clip(
    sample_path: Optional[str],
) -> Tuple[Optional[str], float, float, str, str, str, str]:
    """Loads a pre-packaged classroom sample clip directly into the video player and sets optimal language pair."""
    if not sample_path or not os.path.exists(sample_path):
        raise gr.Error(
            "Sample clip not found! Please check the `samples/` folder."
        )
    dur = get_video_duration(sample_path)
    fname = os.path.basename(sample_path)

    if "sample_02_CE" in fname or "_CE_" in fname:
        src_lang = "zh-TW"
        tgt_lang = "en-US"
        pair_note = "C-E (Traditional Chinese → English)"
    else:
        src_lang = "en-US"
        tgt_lang = "zh-TW"
        pair_note = "E-C (English → Traditional Chinese)"

    msg = (
        f"🎓 Loaded Classroom Sample Clip '{fname}' ({dur:.2f}s) | "
        f"Auto-selected language pair: {pair_note}. Click '🚀 Step 1' to begin!"
    )
    banner = build_dynamic_status_banner(
        "idle",
        f"Classroom Sample Loaded: {fname} ({dur:.2f}s)",
        f"Language pair set to {pair_note}. Click '🚀 Step 1: Transcribe & Translate Video' now!",
        pct=100,
    )
    return sample_path, 0.0, dur, src_lang, tgt_lang, msg, banner


# ==============================================================================
# Gradio UI Layout (SubCraft-PE • Default: ⚡ faster-distil-whisper-large-v3.5)
# ==============================================================================


def build_app() -> gr.Blocks:
    sample_choices = ensure_builtin_sample_clips()
    default_sample_val = sample_choices[0][1] if sample_choices else None
    ensure_cjk_font_for_subtitles()

    with gr.Blocks(
        title="SubCraft-PE | AI Bilingual Video Subtitle & C-E / E-C Post-Editing Workbench"
    ) as demo:
        gr.Markdown(
            """
            # 🎬 SubCraft-PE: AI Video Bilingual Subtitle & C-E / E-C Post-Editing Workbench
            **⚡ `deepdml/faster-distil-whisper-large-v3.5` Default (Anti-Drift 15s Window) • 🧠 Whole-Passage Context Translation • 🎓 Classroom Sample Clips • 🔥 Guaranteed CJK & Mode-Aligned ASS Hard-Burn MP4**
            """
        )

        prepared_video_state = gr.State(value=None)
        current_json_state = gr.State(value="")
        baseline_json_state = gr.State(value="")
        wb_json_bridge = gr.Textbox(value="", visible=False)

        with gr.Tabs():
            with gr.Tab(
                "🛠️ 1. Bilingual Subtitle & Post-Editing Workbench / 雙語字幕與譯後編輯工作台"
            ):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="1. Upload Video or Load Classroom Sample Below / 上傳影片或由下方載入課堂練習片段",
                            sources=["upload"],
                        )
                        with gr.Group():
                            sample_dropdown = gr.Dropdown(
                                label="🎓 In-Class Practice Sample Clips (`samples/` folder) / 課堂實作範例影片選單",
                                choices=sample_choices,
                                value=default_sample_val,
                            )
                            load_sample_btn = gr.Button(
                                "📥 Load Selected Sample Clip for In-Class Practice / 一鍵載入課堂範例影片",
                                variant="secondary",
                            )
                        with gr.Row():
                            start_sec_input = gr.Number(
                                label="Clip Start (sec) / 開始秒數",
                                value=0.0,
                                precision=2,
                            )
                            end_sec_input = gr.Number(
                                label="Clip End (sec, auto-set to video end) / 結束秒數",
                                value=0.0,
                                precision=2,
                            )

                    with gr.Column(scale=1):
                        whisper_size_input = gr.Dropdown(
                            label="🎙️ SOTA T4 GPU Speech Recognition Model / 高精準語音辨識模型 (Default: ⚡ faster-distil-whisper-large-v3.5)",
                            choices=ASR_MODEL_CHOICES,
                            value=DEFAULT_ASR_MODEL,
                        )
                        with gr.Row():
                            mt_engine_input = gr.Dropdown(
                                label="🌐 Machine Translation Engine (Free, No API Key) / 免費機器翻譯引擎",
                                choices=MT_ENGINE_CHOICES,
                                value="google_free",
                            )
                            translation_mode_input = gr.Dropdown(
                                label="🧠 Context Translation Strategy / 翻譯語境策略",
                                choices=TRANSLATION_MODE_CHOICES,
                                value="full_context_passage",
                            )
                        with gr.Row():
                            src_lang_input = gr.Dropdown(
                                label="Original Video Language / 影片原始語言",
                                choices=SOURCE_LANG_CHOICES,
                                value="auto",
                            )
                            tgt_lang_input = gr.Dropdown(
                                label="Target Subtitle Language / 目標翻譯語言",
                                choices=TARGET_LANG_CHOICES,
                                value="zh-TW",
                            )
                        with gr.Row():
                            presentation_mode_input = gr.Dropdown(
                                label="📺 Subtitle Presentation Mode (Live, SRT/VTT & Hard-Burn MP4) / 雙語或單語字幕呈現模式",
                                choices=PRESENTATION_MODE_CHOICES,
                                value="bilingual_target_top",
                            )
                            font_size_slider = gr.Slider(
                                minimum=12,
                                maximum=38,
                                step=1,
                                value=20,
                                label="🔠 Subtitle Font Size (Live & Hard-Burn MP4) / 字幕字體大小",
                            )
                        with gr.Row():
                            run_step1_btn = gr.Button(
                                "🚀 Step 1: Transcribe & Translate Video (Anti-Drift ASR + Whole-Passage Context MT) / 執行完整辨識與全文語境翻譯",
                                variant="primary",
                                scale=2,
                            )
                            retranslate_btn = gr.Button(
                                "🔄 Only Re-Translate Text (Keep Current ASR) / 僅重新翻譯文字",
                                variant="secondary",
                                scale=1,
                            )

                with gr.Accordion(
                    "🛡️ Student Data-Loss Protection & Session Recovery (防止斷線或誤刪：上傳備份檔或一鍵還原) / 學生進度救援區",
                    open=False,
                ):
                    gr.Markdown(
                        """
                        - **Automatic Browser Backup**: Every character you type, or subtitle you **➕ Add**, **🗑️ Delete**, or **✂️ Split**, is immediately saved to your browser's `localStorage`. If your page reloads, click **`⏪ Restore Browser Draft`** inside the Workbench header!
                        - **Server & File Checkpoint Recovery**: You can also download the **`💾 Project Backup (.json)`** file below at any time, or upload it here to restore your exact session.
                        """
                    )
                    with gr.Row():
                        upload_backup_input = gr.File(
                            label="📂 Upload Saved Project Backup (.json) (Optional — leave empty to restore latest server auto-backup)",
                            file_types=[".json"],
                        )
                        restore_backup_btn = gr.Button(
                            "⏪ Restore Session from Uploaded .JSON or Latest Server Auto-Backup / 還原譯後編輯進度",
                            variant="secondary",
                        )

                dynamic_status_banner = gr.HTML(
                    value=build_dynamic_status_banner(
                        "idle",
                        "SubCraft-PE Ready (Default: ⚡ faster-distil-whisper-large-v3.5 + Whole-Passage Context MT)",
                        "Click '📥 Load Selected Sample Clip' (or upload a video) and press Step 1 to begin!",
                        pct=0,
                    )
                )

                status_log = gr.Textbox(
                    label="📡 Live Streaming Pipeline & T4 GPU Execution Log / 即時執行狀態紀錄",
                    lines=5,
                    max_lines=10,
                    autoscroll=True,
                    interactive=False,
                )

                workbench_html = gr.HTML(
                    value=build_bilingual_workbench_html(None, [])
                )

                with gr.Row():
                    apply_wb_btn = gr.Button(
                        "💾 Step 2: Save & Apply HTML Workbench Edits (Updates Added/Deleted/Split Cues, SRT, VTT, C-E/E-C PDF Report & JSON Backup) / 儲存並更新所有匯出檔與策略報告",
                        variant="primary",
                    )

                pe_report_html = gr.HTML(
                    label="Post-Editing Comparison & C-E / E-C Strategy Report"
                )

                with gr.Accordion(
                    "📥 Step 3: Download Subtitles (.SRT/.VTT), Post-Editing Report (.PDF), Project Backup (.JSON) & Mode-Aligned Burned MP4",
                    open=True,
                ):
                    with gr.Row():
                        orig_srt_file = gr.File(
                            label="📄 Original Source .SRT / 原文字幕檔"
                        )
                        trans_srt_file = gr.File(
                            label="📄 Post-Edited .SRT (Matches Selected Mode: Bilingual or Monolingual) / 依所選模式匯出之 .SRT"
                        )
                        bilingual_vtt_file = gr.File(
                            label="📄 Web Presentation .VTT (Matches Selected Mode) / 網頁字幕 .VTT"
                        )
                    with gr.Row():
                        pe_report_pdf_file = gr.File(
                            label="📊 Post-Editing Summary & C-E / E-C Strategy Report (.PDF) / 譯後編輯策略分析報告 PDF"
                        )
                        project_backup_json_file = gr.File(
                            label="💾 Student Project Backup Checkpoint (.JSON) / 學生譯後編輯進度備份檔"
                        )

                    burn_mp4_btn = gr.Button(
                        "🔥 Burn Subtitles into MP4 (100% Aligned with Selected Bilingual/Monolingual Mode, CJK Font & Font Size) / 依所選模式與字級燒錄硬字幕 MP4",
                        variant="primary",
                    )

                    with gr.Row():
                        burned_video_player = gr.Video(
                            label="🎬 Burned-in MP4 Video Preview (Aligned with Your Mode & Font Size) / 硬字幕完成影片預覽"
                        )
                        burned_video_file = gr.File(
                            label="⬇️ Download Burned MP4 / 下載燒錄後 MP4"
                        )

            with gr.Tab(
                "📚 2. SOTA T4 Models & C-E / E-C Post-Editing Skills Guide / T4 語音模型與中英譯後編輯六大技巧指南"
            ):
                gr.HTML(value=CE_EC_STRATEGY_GUIDE_HTML)

        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[
                start_sec_input,
                end_sec_input,
                status_log,
                dynamic_status_banner,
            ],
        )

        load_sample_btn.click(
            fn=load_selected_sample_clip,
            inputs=[sample_dropdown],
            outputs=[
                video_input,
                start_sec_input,
                end_sec_input,
                src_lang_input,
                tgt_lang_input,
                status_log,
                dynamic_status_banner,
            ],
        )

        run_step1_btn.click(
            fn=run_step1_pipeline,
            inputs=[
                video_input,
                start_sec_input,
                end_sec_input,
                whisper_size_input,
                mt_engine_input,
                translation_mode_input,
                src_lang_input,
                tgt_lang_input,
                presentation_mode_input,
                font_size_slider,
            ],
            outputs=[
                prepared_video_state,
                current_json_state,
                baseline_json_state,
                workbench_html,
                pe_report_html,
                orig_srt_file,
                trans_srt_file,
                bilingual_vtt_file,
                pe_report_pdf_file,
                project_backup_json_file,
                status_log,
                dynamic_status_banner,
                wb_json_bridge,
            ],
        )

        retranslate_btn.click(
            fn=retranslate_existing_cues,
            inputs=[
                wb_json_bridge,
                mt_engine_input,
                translation_mode_input,
                src_lang_input,
                tgt_lang_input,
                presentation_mode_input,
                font_size_slider,
                current_json_state,
                prepared_video_state,
            ],
            outputs=[
                current_json_state,
                baseline_json_state,
                workbench_html,
                pe_report_html,
                orig_srt_file,
                trans_srt_file,
                bilingual_vtt_file,
                pe_report_pdf_file,
                project_backup_json_file,
                status_log,
                dynamic_status_banner,
            ],
            js=COLLECT_WORKBENCH_JS,
        )

        apply_wb_btn.click(
            fn=apply_html_workbench_edits,
            inputs=[
                wb_json_bridge,
                presentation_mode_input,
                font_size_slider,
                current_json_state,
                baseline_json_state,
                prepared_video_state,
            ],
            outputs=[
                current_json_state,
                presentation_mode_input,
                font_size_slider,
                workbench_html,
                pe_report_html,
                orig_srt_file,
                trans_srt_file,
                bilingual_vtt_file,
                pe_report_pdf_file,
                project_backup_json_file,
                status_log,
                dynamic_status_banner,
            ],
            js=COLLECT_WORKBENCH_JS,
        )

        presentation_mode_input.change(
            fn=apply_html_workbench_edits,
            inputs=[
                wb_json_bridge,
                presentation_mode_input,
                font_size_slider,
                current_json_state,
                baseline_json_state,
                prepared_video_state,
            ],
            outputs=[
                current_json_state,
                presentation_mode_input,
                font_size_slider,
                workbench_html,
                pe_report_html,
                orig_srt_file,
                trans_srt_file,
                bilingual_vtt_file,
                pe_report_pdf_file,
                project_backup_json_file,
                status_log,
                dynamic_status_banner,
            ],
            js=COLLECT_WORKBENCH_JS,
        )

        font_size_slider.release(
            fn=apply_html_workbench_edits,
            inputs=[
                wb_json_bridge,
                presentation_mode_input,
                font_size_slider,
                current_json_state,
                baseline_json_state,
                prepared_video_state,
            ],
            outputs=[
                current_json_state,
                presentation_mode_input,
                font_size_slider,
                workbench_html,
                pe_report_html,
                orig_srt_file,
                trans_srt_file,
                bilingual_vtt_file,
                pe_report_pdf_file,
                project_backup_json_file,
                status_log,
                dynamic_status_banner,
            ],
            js=COLLECT_WORKBENCH_JS,
        )

        restore_backup_btn.click(
            fn=restore_from_uploaded_or_server_backup,
            inputs=[
                upload_backup_input,
                prepared_video_state,
                video_input,
            ],
            outputs=[
                prepared_video_state,
                current_json_state,
                baseline_json_state,
                presentation_mode_input,
                font_size_slider,
                workbench_html,
                pe_report_html,
                orig_srt_file,
                trans_srt_file,
                bilingual_vtt_file,
                pe_report_pdf_file,
                project_backup_json_file,
                status_log,
                dynamic_status_banner,
            ],
        )

        burn_mp4_btn.click(
            fn=burn_subtitles_to_mp4,
            inputs=[
                wb_json_bridge,
                presentation_mode_input,
                font_size_slider,
                current_json_state,
                baseline_json_state,
                prepared_video_state,
            ],
            outputs=[
                burned_video_player,
                burned_video_file,
                current_json_state,
                presentation_mode_input,
                font_size_slider,
                status_log,
                dynamic_status_banner,
            ],
            js=COLLECT_WORKBENCH_JS,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(
        share=True,
        debug=True,
        allowed_paths=[SHARED_MEDIA_DIR, SAMPLES_DIR, tempfile.gettempdir()],
    )

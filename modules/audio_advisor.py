"""
Audio Advisory & Voice Synthesis Module for Smart Farm Assistant.
Generates audio explanations using gTTS for farmers.
"""

import os
import io
from typing import Optional

def synthesize_speech_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    """
    Synthesizes speech audio in MP3 format from text.
    Returns bytes suitable for playback via st.audio.
    """
    try:
        from gtts import gTTS
        clean_text = text.replace("*", "").replace("#", "").replace("`", "")[:500]
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        print(f"Speech synthesis error: {e}")
        return None

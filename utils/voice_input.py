"""
Voice Input Utilities for SegMed-Pro

This module provides speech-to-text functionality for voice input
"""

import logging
import numpy as np
import whisper
from typing import Optional, Tuple
import os

# Set up logger for this module
logger = logging.getLogger(__name__)

class VoiceInputHandler:
    """Handles voice input and speech-to-text conversion"""
    
    def __init__(self, model_name: str = "medium"):
        """
        Initialize the voice input handler with improved model for better accuracy
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
                       - small is recommended for better English speech accuracy
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model lazily"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name} for English speech recognition")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            self.model = None
    
    def transcribe_audio(self, audio_data) -> Tuple[str, bool]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data from Gradio Audio component (tuple of sample_rate, audio_array) or file path
        
        Returns:
            Tuple of (transcribed_text, success_flag)
        """
        try:
            if self.model is None:
                self._load_model()
                if self.model is None:
                    return "Error: Whisper model not loaded", False
            
            if audio_data is None:
                return "No audio data received", False
            
            # Handle different audio data formats from Gradio
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                # Standard format: (sample_rate, audio_array)
                sample_rate, audio_array = audio_data
            elif isinstance(audio_data, str):
                # File path format - Whisper can handle this directly
                logger.info(f"Transcribing audio file: {audio_data}")
                result = self.model.transcribe(
                    audio_data,
                    language="en",  # Force English language
                    task="transcribe",
                    fp16=False
                )
                transcribed_text = result["text"].strip()
                logger.info(f"English transcription successful: '{transcribed_text}'")
                return transcribed_text, True
            else:
                return "Unsupported audio format", False
            
            if audio_array is None or len(audio_array) == 0:
                return "Empty audio recording", False
            
            # Convert to float32 and normalize if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                # Normalize if the audio is in int16 range
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / 32768.0
            
            # Ensure the audio is mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Audio preprocessing for better transcription
            # Remove silence at beginning and end
            audio_array = self._trim_silence(audio_array)
            
            # Check if audio is too short or too quiet
            if len(audio_array) < sample_rate * 0.5:  # Less than 0.5 seconds
                return "Audio too short (minimum 0.5 seconds)", False
            
            # Check audio volume
            max_amplitude = np.max(np.abs(audio_array))
            if max_amplitude < 0.01:  # Very quiet audio
                return "Audio too quiet - please speak louder", False
            
            logger.info(f"Transcribing audio: sample_rate={sample_rate}, duration={len(audio_array)/sample_rate:.2f}s, max_amplitude={max_amplitude:.3f}")
            
            # Try transcription with optimized parameters first
            try:
                result = self.model.transcribe(
                    audio_array, 
                    language="en",  # Force English language for better accuracy
                    task="transcribe",  # Explicit transcription task
                    fp16=False,  # Use fp32 for better stability on CPU
                    temperature=0.0,  # Use greedy decoding for more consistent results
                    beam_size=5,  # Use beam search for better accuracy
                    best_of=5,  # Consider multiple candidates
                    no_speech_threshold=0.6,  # Lower threshold for speech detection
                    logprob_threshold=-1.0,  # Accept more confident predictions
                    compression_ratio_threshold=2.4,  # Standard compression ratio
                )
            except Exception as e:
                logger.warning(f"Optimized transcription failed: {e}. Trying basic transcription...")
                # Fallback to basic transcription if optimized parameters fail
                result = self.model.transcribe(
                    audio_array, 
                    language="en",  # Force English language for better accuracy
                    task="transcribe",  # Explicit transcription task
                    fp16=False  # Use fp32 for better stability on CPU
                )
            
            transcribed_text = result["text"].strip()
            
            # Additional quality checks
            if not transcribed_text:
                return "No speech detected - please try speaking more clearly", False
            
            # Check for confidence using segments
            if "segments" in result and result["segments"]:
                avg_confidence = np.mean([seg.get("no_speech_prob", 0.5) for seg in result["segments"]])
                if avg_confidence > 0.8:  # High no_speech probability means low confidence
                    logger.warning(f"Low confidence transcription: {transcribed_text}")
            
            logger.info(f"English transcription successful: '{transcribed_text}'")
            return transcribed_text, True
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            return f"Transcription error: {str(e)}", False
    
    def _trim_silence(self, audio_array, threshold=0.01):
        """
        Trim silence from the beginning and end of audio
        
        Args:
            audio_array: Audio data as numpy array
            threshold: Amplitude threshold below which audio is considered silence
        
        Returns:
            Trimmed audio array
        """
        try:
            # Find the start and end of non-silent audio
            abs_audio = np.abs(audio_array)
            non_silent = abs_audio > threshold
            
            if not np.any(non_silent):
                # If all audio is below threshold, return original
                return audio_array
            
            # Find first and last non-silent sample
            non_silent_indices = np.where(non_silent)[0]
            start_idx = max(0, non_silent_indices[0] - 1000)  # Keep small buffer
            end_idx = min(len(audio_array), non_silent_indices[-1] + 1000)
            
            return audio_array[start_idx:end_idx]
        except:
            # If trimming fails, return original audio
            return audio_array
    
    def is_model_loaded(self) -> bool:
        """Check if the Whisper model is loaded"""
        return self.model is not None

# Global instance for reuse
_voice_handler = None

def get_voice_handler() -> VoiceInputHandler:
    """Get or create the global voice input handler"""
    global _voice_handler
    if _voice_handler is None:
        _voice_handler = VoiceInputHandler("large")  # Use medium model for better accuracy
    return _voice_handler

def transcribe_voice_input(audio_data) -> Tuple[str, bool]:
    """
    Convenience function to transcribe voice input
    
    Args:
        audio_data: Audio data from Gradio Audio component
    
    Returns:
        Tuple of (transcribed_text, success_flag)
    """
    handler = get_voice_handler()
    return handler.transcribe_audio(audio_data)

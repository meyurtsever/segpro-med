"""
Voice Input Utilities for SegMed-Pro

This module provides speech-to-text functionality using Whisper Small with exact HuggingFace implementation
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

class VoiceInputHandler:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load Whisper Small model using the exact HuggingFace example"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            logger.info("Loading Whisper Small model...")
            
            # Load model and processor exactly as shown in HF examples
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            
            # Set forced decoder ids to None for automatic language detection
            # Or set to English for better performance
            self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            logger.info(f"Whisper Small model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper Small model: {e}")
            self.processor = None
            self.model = None
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """
        Transcribe audio using Whisper Small with exact HuggingFace implementation
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
        
        Returns:
            str: Transcribed text
        """
        if self.processor is None or self.model is None:
            return "Error: Whisper model not loaded"
        
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample to 16kHz if needed (Whisper always expects 16kHz)
            if sample_rate != 16000:
                try:
                    import librosa
                    logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                except ImportError:
                    logger.warning("librosa not available for resampling. Install with: pip install librosa")
                    # Simple downsampling fallback
                    if sample_rate > 16000:
                        step = sample_rate // 16000
                        audio_data = audio_data[::step]
                        logger.info(f"Applied simple downsampling by factor {step}")
            
            # Process audio using the processor - exactly as in HF examples
            input_features = self.processor(
                audio_data, 
                sampling_rate=16000,  # Always use 16000 since we resampled
                return_tensors="pt"
            ).input_features
            
            # Move input to device
            input_features = input_features.to(self.device)
            
            # Generate transcription - exactly as in HF examples
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode token ids to text - exactly as in HF examples
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # Return the first (and only) transcription
            result = transcription[0].strip()
            logger.info(f"Transcription successful: '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"Error: {str(e)}"

# Global instance
_voice_handler = None

def get_voice_handler():
    """Get or create the global voice input handler"""
    global _voice_handler
    if _voice_handler is None:
        _voice_handler = VoiceInputHandler()
    return _voice_handler

def transcribe_voice_input(audio_data, sample_rate=16000):
    """
    Transcribe voice input using Whisper Small
    
    Args:
        audio_data: Audio data as numpy array (from Gradio Audio component)
        sample_rate: Sample rate of the audio
    
    Returns:
        str: Transcribed text
    """
    try:
        if audio_data is None:
            return "No audio data provided"
        
        # Handle Gradio audio format (sample_rate, audio_array)
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            audio_data = audio_array
        
        # Ensure we have valid audio data
        if audio_data is None or len(audio_data) == 0:
            return "No audio data to transcribe"
        
        # Get the voice handler and transcribe
        handler = get_voice_handler()
        return handler.transcribe_audio(audio_data, sample_rate)
        
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}")
        return f"Transcription error: {str(e)}"

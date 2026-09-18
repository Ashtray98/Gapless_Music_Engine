import logging
import math
from dataclasses import dataclass
from typing import Tuple

import librosa
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio ingestion and processing failures."""
    pass

@dataclass
class EngineConfig:
    """Configuration parameters for the DSP engine."""
    sample_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 512
    threshold_db: float = -60.0
    pre_roll_ms: int = 300
    svd_rank_k: int = 20


class GaplessAudioEngine:
    def __init__(self, config: EngineConfig = EngineConfig()):
        self.config = config

    def load_audio(self, file_path: str) -> npt.NDArray[np.float32]:
        """Ingests high-fidelity audio into a stereo numpy array."""
        try:
            audio, _ = librosa.load(file_path, sr=self.config.sample_rate, mono=False)
            return audio
        except Exception as e:
            logger.error(f"Failed to ingest audio file {file_path}: {e}")
            raise AudioProcessingError(f"Ingestion failed: {e}")

    def denoise_svd(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Applies Truncated SVD to isolate structural signal from stochastic noise."""
        mono = librosa.to_mono(audio) if audio.ndim > 1 else audio
        
        stft_matrix = librosa.stft(
            mono, 
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length
        )
        
        U, Sigma, Vt = np.linalg.svd(stft_matrix, full_matrices=False)
        
        Sigma_filtered = np.zeros_like(Sigma)
        Sigma_filtered[:self.config.svd_rank_k] = Sigma[:self.config.svd_rank_k]
        
        stft_clean = U @ np.diag(Sigma_filtered) @ Vt
        
        return librosa.istft(stft_clean, hop_length=self.config.hop_length)

    def detect_boundaries(self, audio: npt.NDArray[np.float32]) -> Tuple[int, int]:
        """Calculates gapless start and end sample indices using RMS thresholding."""
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=self.config.n_fft, 
            hop_length=self.config.hop_length
        )[0]
        
        db_energy = librosa.amplitude_to_db(rms, ref=np.max)
        active_frames = np.where(db_energy > self.config.threshold_db)[0]
        
        if active_frames.size == 0:
            raise AudioProcessingError("Signal is entirely below noise threshold.")
            
        start_sample = librosa.frames_to_samples(
            active_frames[0], 
            hop_length=self.config.hop_length
        )
        end_sample = librosa.frames_to_samples(
            active_frames[-1], 
            hop_length=self.config.hop_length
        )
        
        pre_roll_samples = math.floor((self.config.pre_roll_ms / 1000.0) * self.config.sample_rate)
        final_start = max(0, start_sample - pre_roll_samples)
        
        return final_start, end_sample

    def process_track(self, file_path: str) -> Tuple[int, int]:
        """Executes the full DSP pipeline on a target audio file."""
        logger.info(f"Initiating DSP pipeline for: {file_path}")
        
        raw_audio = self.load_audio(file_path)
        clean_audio = self.denoise_svd(raw_audio)
        start_idx, end_idx = self.detect_boundaries(clean_audio)
        
        logger.info(f"Calculated Boundaries - Start: {start_idx}, End: {end_idx}")
        return start_idx, end_idx
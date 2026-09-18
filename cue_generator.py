import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

class CueGenerationError(Exception):
    """Custom exception for CUE formatting failures."""
    pass

@dataclass
class TrackContext:
    """Metadata and boundary state for a single track."""
    track_num: int
    title: str
    performer: str
    file_path: str
    start_sample: int

class CueSheetGenerator:
    def __init__(self, album_title: str, album_performer: str, sample_rate: int = 44100):
        self.album_title = album_title
        self.album_performer = album_performer
        self.sample_rate = sample_rate
        self.tracks: List[TrackContext] = []

    def generate_single_file_cue(self) -> str:
        """Compiles a Red Book CUE string for a single merged master file."""
        if not self.tracks:
            raise CueGenerationError("No tracks provided.")
            
        # The master file name is stored in the first track's context
        master_file = self.tracks[0].file_path
            
        lines = [
            f'PERFORMER "{self.album_performer}"',
            f'TITLE "{self.album_title}"',
            f'FILE "{master_file}" WAVE'
        ]
        
        for track in sorted(self.tracks, key=lambda t: t.track_num):
            timecode = self._samples_to_timecode(track.start_sample)
            lines.extend([
                f'  TRACK {track.track_num:02d} AUDIO',
                f'    TITLE "{track.title}"',
                f'    PERFORMER "{track.performer}"',
                f'    INDEX 01 {timecode}'
            ])
            
        return "\n".join(lines) + "\n"

    def _samples_to_timecode(self, samples: int) -> str:
        """
        Converts absolute sample index to CD-DA mm:ss:ff format.
        Strict integer arithmetic prevents floating-point drift.
        """
        samples_per_frame = self.sample_rate // 75
        total_frames = samples // samples_per_frame
        
        frames = total_frames % 75
        total_seconds = total_frames // 75
        seconds = total_seconds % 60
        minutes = total_seconds // 60
        
        return f"{minutes:02d}:{seconds:02d}:{frames:02d}"

    def add_track(self, track: TrackContext) -> None:
        """Registers a processed track into the album sequence."""
        self.tracks.append(track)
        
    def generate(self) -> str:
        """Compiles the tracked data into a compliant Red Book CUE string."""
        if not self.tracks:
            raise CueGenerationError("Cannot generate CUE sheet: No tracks provided.")
            
        # Album-level metadata
        lines = [
            f'PERFORMER "{self.album_performer}"',
            f'TITLE "{self.album_title}"'
        ]
        
        # Track-level indices
        for track in sorted(self.tracks, key=lambda t: t.track_num):
            timecode = self._samples_to_timecode(track.start_sample)
            lines.extend([
                f'FILE "{track.file_path}" WAVE',
                f'  TRACK {track.track_num:02d} AUDIO',
                f'    TITLE "{track.title}"',
                f'    PERFORMER "{track.performer}"',
                f'    INDEX 01 {timecode}'
            ])
            
        return "\n".join(lines) + "\n"
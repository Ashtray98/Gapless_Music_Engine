import logging
from pathlib import Path
import numpy as np
import soundfile as sf  # Required for high-fidelity lossless export

from engine import GaplessAudioEngine, EngineConfig
from cue_generator import CueSheetGenerator, TrackContext

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def main():
    album_dir = Path("./my_flac_folder")
    flac_files = sorted(album_dir.glob("*.flac"))
    
    if not flac_files:
        logging.error("No FLAC files found.")
        return

    album_title = album_dir.name.replace("_", " ").title()
    config = EngineConfig(n_fft=1024, threshold_db=-60.0, pre_roll_ms=300, svd_rank_k=20)
    engine = GaplessAudioEngine(config)
    
    # We tell the generator the name of our final master file
    master_file_name = f"{album_dir.name}_master.flac"
    cue_gen = CueSheetGenerator(album_title=album_title, album_performer="Various Artists")
    
    master_audio_buffer = []
    cumulative_sample_index = 0
    
    for track_num, file_path in enumerate(flac_files, start=1):
        try:
            # 1. Load the pristine stereo matrix (librosa loads as 2 x N)
            raw_audio = engine.load_audio(str(file_path.resolve()))
            
            # 2. Process control plane to get precise trim points
            clean_audio = engine.denoise_svd(raw_audio)
            start_idx, end_idx = engine.detect_boundaries(clean_audio)
            
            # 3. Slice the pristine data plane (stereo requires slicing column-wise)
            trimmed_slice = raw_audio[:, start_idx:end_idx]
            master_audio_buffer.append(trimmed_slice)
            
            # 4. Map the CUE sheet using the CUMULATIVE timeline, not the local start
            clean_title = file_path.stem.replace("_", " ").title()
            ctx = TrackContext(
                track_num=track_num,
                title=clean_title,
                performer="Various Artists",
                file_path=master_file_name, # Point to the master file
                start_sample=cumulative_sample_index
            )
            cue_gen.add_track(ctx)
            
            # 5. Advance the global album clock by the EXACT length of this trimmed slice
            slice_duration_samples = end_idx - start_idx
            cumulative_sample_index += slice_duration_samples
            
        except Exception as e:
            logging.error(f"Pipeline failed on {file_path.name}: {e}")
            continue

    # 6. Matrix Concatenation
    # Stitch all stereo slices together along the time axis (axis=1)
    master_audio_matrix = np.concatenate(master_audio_buffer, axis=1)
    
    # 7. Write to Disk (Forced to CD-DA 16-bit PCM spec)
    master_out_path = album_dir / master_file_name
    sf.write(
        str(master_out_path), 
        master_audio_matrix.T, 
        config.sample_rate, 
        subtype='PCM_16'
    )
    logging.info(f"Successfully rendered master audio: {master_out_path}")
    
    # 8. Render the Single-File CUE Sheet
    final_cue = cue_gen.generate_single_file_cue()
    cue_out_path = album_dir / f"{album_dir.name}_gapless.cue"
    
    with open(cue_out_path, "w") as f:
        f.write(final_cue)
    logging.info(f"Successfully generated Master CUE: {cue_out_path}")

if __name__ == "__main__":
    main()
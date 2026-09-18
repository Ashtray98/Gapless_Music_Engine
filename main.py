import logging
from pathlib import Path
from engine import GaplessAudioEngine, EngineConfig
from cue_generator import CueSheetGenerator, TrackContext

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def main():
    # 1. Point this directly to the folder containing your FLAC files
    # (Change this path to match your actual directory)
    album_dir = Path("./my_flac_folder")
    
    # 2. Dynamically grab all .flac files in the folder and sort them
    flac_files = sorted(album_dir.glob("*.flac"))
    
    if not flac_files:
        logging.error(f"No FLAC files found in {album_dir.resolve()}")
        return

    # Use the folder name as a fallback album title
    album_title = album_dir.name.replace("_", " ").title()
    
    config = EngineConfig(n_fft=1024, threshold_db=-60.0, pre_roll_ms=300, svd_rank_k=20)
    engine = GaplessAudioEngine(config)
    
    cue_gen = CueSheetGenerator(
        album_title=album_title,
        album_performer="Various Artists" # You can hardcode this or use a tagging library later
    )
    
    # 3. Process the folder dynamically
    for track_num, file_path in enumerate(flac_files, start=1):
        try:
            # We pass the absolute path to the engine
            start_sample, end_sample = engine.process_track(str(file_path.resolve()))
            
            # Use the filename (minus the .flac extension) as the track title
            clean_title = file_path.stem.replace("_", " ").title()
            
            ctx = TrackContext(
                track_num=track_num,
                title=clean_title,
                performer="Various Artists",
                file_path=file_path.name, # The CUE sheet only needs the raw filename
                start_sample=start_sample
            )
            
            cue_gen.add_track(ctx)
            
        except Exception as e:
            logging.error(f"Pipeline failed on {file_path.name}: {e}")
            continue
            
    # 4. Generate the CUE sheet directly inside that same folder
    try:
        final_cue_sheet = cue_gen.generate()
        output_file = album_dir / f"{album_dir.name}_gapless.cue"
        
        with open(output_file, "w") as f:
            f.write(final_cue_sheet)
            
        logging.info(f"Successfully generated CUE sheet at {output_file.resolve()}")
        
    except Exception as e:
        logging.error(f"Failed to compile CUE sheet: {e}")

if __name__ == "__main__":
    main()
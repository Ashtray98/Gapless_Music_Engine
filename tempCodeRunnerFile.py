 # 8. Render the Single-File CUE Sheet
    final_cue = cue_gen.generate_single_file_cue()
    cue_out_path = album_dir / f"{album_dir.name}_gapless.cue"
    
    with open(cue_out_path, "w") as f:
        f.write(final_cue)
    logging.info(f"Successfully generated Master CUE: {cue_out_path}")
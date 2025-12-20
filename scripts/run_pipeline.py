import sys, os
import numpy as np

# SMART PATH: Find the project root automatically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import yaml
from modules.loader import load_keypoints, load_gloss_map
from modules.translate import build_vocabulary
from modules.builder import build_sequence_from_text
from modules.renderer import render_layered

AUTO = {
 "GLOSS_0":"HELLO","GLOSS_1":"WORLD","GLOSS_2":"GOOD","GLOSS_3":"MORNING",
 "GLOSS_4":"NIGHT","GLOSS_5":"THANK","GLOSS_6":"YOU","GLOSS_7":"NAME",
 "GLOSS_8":"WHAT","GLOSS_9":"YES","GLOSS_10":"NO","GLOSS_11":"PLEASE",
 "GLOSS_12":"SORRY","GLOSS_13":"HELP","GLOSS_14":"STOP","GLOSS_15":"GO",
 "GLOSS_16":"COME","GLOSS_17":"EAT","GLOSS_18":"DRINK","GLOSS_19":"HAPPY",
}

def main(cfg):
    # Smart Path Construction
    pkl_path = os.path.join(PROJECT_ROOT, cfg["pkl_path"])
    gloss_csv = os.path.join(PROJECT_ROOT, cfg["gloss_csv"])
    output_dir = os.path.join(PROJECT_ROOT, cfg["output_dir"])

    kp_data = load_keypoints(pkl_path)
    gloss_to_video, video_to_gloss = load_gloss_map(gloss_csv)
    vocab = build_vocabulary(gloss_to_video, AUTO)
    os.makedirs(output_dir, exist_ok=True)

    test_sentences = [
        "HELLO WORLD",
        "GOOD MORNING",
        "THANK YOU",
        "WHAT IS YOUR NAME",
        "STOP PLEASE",
        "HAPPY MORNING"
    ]

    for text in test_sentences:
        sequences = build_sequence_from_text(text, kp_data, vocab)
        
        filename = text.replace(" ", "_").lower()
        outfile = os.path.join(output_dir, filename + ".mp4")
        
        # Save .npy for comparison
        if len(sequences) > 0:
            full_data = np.concatenate(sequences, axis=0)
            npy_outfile = outfile.replace(".mp4", ".npy")
            np.save(npy_outfile, full_data)
            print(f"Saved data: {npy_outfile}")

        render_layered(
            sequences, outfile,
            width=cfg["render_width"], height=cfg["render_height"],
            fps=cfg["render_fps"], transition=cfg["transition_frames"]
        )
        print(f"Saved video: {outfile}")

if __name__ == "__main__":
    # Load config from PROJECT_ROOT/configs/default.yaml
    config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    cfg = yaml.safe_load(open(config_path))
    main(cfg)

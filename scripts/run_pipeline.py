import sys, os
# Add project root so modules/ can be imported when using python scripts/run_pipeline.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import yaml
from modules.loader import load_keypoints, load_gloss_map
from modules.translate import build_vocabulary
from modules.builder import build_sequence_from_text
from modules.renderer import render_layered

# 20-word English → Gloss mapping
AUTO = {
 "GLOSS_0":"HELLO","GLOSS_1":"WORLD","GLOSS_2":"GOOD","GLOSS_3":"MORNING",
 "GLOSS_4":"NIGHT","GLOSS_5":"THANK","GLOSS_6":"YOU","GLOSS_7":"NAME",
 "GLOSS_8":"WHAT","GLOSS_9":"YES","GLOSS_10":"NO","GLOSS_11":"PLEASE",
 "GLOSS_12":"SORRY","GLOSS_13":"HELP","GLOSS_14":"STOP","GLOSS_15":"GO",
 "GLOSS_16":"COME","GLOSS_17":"EAT","GLOSS_18":"DRINK","GLOSS_19":"HAPPY",
}

def main(cfg):
    # Load keypoint data
    kp_data = load_keypoints(cfg["pkl_path"])
    gloss_to_video, video_to_gloss = load_gloss_map(cfg["gloss_csv"])

    # Build vocabulary from gloss map
    vocab = build_vocabulary(gloss_to_video, AUTO)

    # Ensure output directory exists
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Test sentences
    test_sentences = [
        "HELLO WORLD",
        "GOOD MORNING",
        "THANK YOU",
        "WHAT IS YOUR NAME",
        "STOP PLEASE",
        "HAPPY MORNING"
    ]

    # Render each output
    for text in test_sentences:
        sequences = build_sequence_from_text(text, kp_data, vocab)
        outfile = os.path.join(
            cfg["output_dir"],
            text.replace(" ", "_").lower() + ".mp4"
        )

        render_layered(
            sequences,
            outfile,
            width=cfg["render_width"],
            height=cfg["render_height"],
            fps=cfg["render_fps"],
            transition=cfg["transition_frames"]
        )

        print("Saved:", outfile)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    main(cfg)

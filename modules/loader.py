import pickle
import csv

def load_keypoints(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_gloss_map(csv_path):
    gloss_to_video = {}
    video_to_gloss = {}

    with open(csv_path, "r") as f:
        r = csv.reader(f)
        header = next(r)
        video_idx = header.index("video")
        gloss_idx = header.index("gloss")

        for row in r:
            vid = row[video_idx].strip()
            gloss = row[gloss_idx].strip()
            gloss_to_video[gloss] = vid
            video_to_gloss[vid] = gloss

    return gloss_to_video, video_to_gloss

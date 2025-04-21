import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))



UVQA_ROOT_DIR = "dataset/"
VIDEO_CHATGPT_ANNOTATION_DIR = "dataset/video_chatgpt/"
UVQA_VIDEO_DIR = [MOMA-LRG VIDEO FRAME DIRECTORY HERE] # e.g., "~/MOMA-LRG/frames"
VIDEO_CHATGPT_ROOT_DIR = [Video-ChatGPT DIRECTORY HERE] # e.g., "~/Video-ChatGPT"
VIDEO_CHATGPT_VIDEO_DIR = [Video-ChatGPT VIDEO FRAME DIRECTORY HERE] # e.g., "~/Video-ChatGPT/frames"
DIDEMO_ROOT_DIR = [DiDeMo DIRECTORY HERE] # e.g., "~/DiDeMo"
DIDEMO_VIDEO_DIR = [DiDeMo VIDEO FRAME DIRECTORY HERE] # e.g., "~/DiDeMo/frames"

UVQA_TRAIN_REL_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/relation", "train_relation.json")
UVQA_EVAL_REL_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/relation", "eval_relation.json")

UVQA_TRAIN_NEG_REL_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/relation", "train_negative_relation.json")

UVQA_TRAIN_OBJ_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/object", "train_object.json")
UVQA_EVAL_OBJ_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/object", "eval_object.json")

UVQA_TRAIN_NEG_OBJ_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/object", "train_negative_object.json")

UVQA_TRAIN_ATT_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/attribute", "train_attribute.json")
UVQA_EVAL_ATT_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/attribute", "eval_attribute.json")

UVQA_TRAIN_NEG_ATT_PATH = os.path.join(UVQA_ROOT_DIR, "uvqa/attribute", "train_negative_attribute.json")
# UVQA_EVAL_NEG_ATT_PATH = None

VIDEO_CHATGPT1_TRAIN_PATH = os.path.join(VIDEO_CHATGPT_ANNOTATION_DIR, "train", "train_uvqa_format_chunk1.json")
VIDEO_CHATGPT2_TRAIN_PATH = os.path.join(VIDEO_CHATGPT_ANNOTATION_DIR, "train", "train_uvqa_format_chunk2.json")
VIDEO_CHATGPT3_TRAIN_PATH = os.path.join(VIDEO_CHATGPT_ANNOTATION_DIR, "train", "train_uvqa_format_chunk3.json")
VIDEO_CHATGPT4_TRAIN_PATH = os.path.join(VIDEO_CHATGPT_ANNOTATION_DIR, "train", "train_uvqa_format_chunk4.json")
VIDEO_CHATGPT5_TRAIN_PATH = os.path.join(VIDEO_CHATGPT_ANNOTATION_DIR, "train", "train_uvqa_format_chunk5.json")
VIDEO_CHATGPT_EVAL_PATH = os.path.join(VIDEO_CHATGPT_ANNOTATION_DIR, "eval", "eval_uvqa_format.json")

VIDEO_CHATGPT1_TRAIN_NEG_PATH = os.path.join(VIDEO_CHATGPT_ROOT_DIR, "train_nagative", "train_neg_chunk1.json")
VIDEO_CHATGPT2_TRAIN_NEG_PATH = os.path.join(VIDEO_CHATGPT_ROOT_DIR, "train_nagative", "train_neg_chunk2.json")
VIDEO_CHATGPT3_TRAIN_NEG_PATH = os.path.join(VIDEO_CHATGPT_ROOT_DIR, "train_nagative", "train_neg_chunk3.json")
VIDEO_CHATGPT4_TRAIN_NEG_PATH = os.path.join(VIDEO_CHATGPT_ROOT_DIR, "train_nagative", "train_neg_chunk4.json")
VIDEO_CHATGPT5_TRAIN_NEG_PATH = os.path.join(VIDEO_CHATGPT_ROOT_DIR, "train_nagative", "train_neg_chunk5.json")


data_match = {
    "uvqa_relation": [UVQA_ROOT_DIR, UVQA_VIDEO_DIR, UVQA_TRAIN_REL_PATH, UVQA_EVAL_REL_PATH],
    "uvqa_object": [UVQA_ROOT_DIR, UVQA_VIDEO_DIR, UVQA_TRAIN_OBJ_PATH, UVQA_EVAL_OBJ_PATH],
    "uvqa_attribute": [DIDEMO_ROOT_DIR, DIDEMO_VIDEO_DIR, UVQA_TRAIN_ATT_PATH, UVQA_EVAL_ATT_PATH],
    "video_chatgpt1": [VIDEO_CHATGPT_ROOT_DIR, VIDEO_CHATGPT_VIDEO_DIR, VIDEO_CHATGPT1_TRAIN_PATH, VIDEO_CHATGPT_EVAL_PATH],
    "video_chatgpt2": [VIDEO_CHATGPT_ROOT_DIR, VIDEO_CHATGPT_VIDEO_DIR, VIDEO_CHATGPT2_TRAIN_PATH, VIDEO_CHATGPT_EVAL_PATH],
    "video_chatgpt3": [VIDEO_CHATGPT_ROOT_DIR, VIDEO_CHATGPT_VIDEO_DIR, VIDEO_CHATGPT3_TRAIN_PATH, VIDEO_CHATGPT_EVAL_PATH],
    "video_chatgpt4": [VIDEO_CHATGPT_ROOT_DIR, VIDEO_CHATGPT_VIDEO_DIR, VIDEO_CHATGPT4_TRAIN_PATH, VIDEO_CHATGPT_EVAL_PATH],
    "video_chatgpt5": [VIDEO_CHATGPT_ROOT_DIR, VIDEO_CHATGPT_VIDEO_DIR, VIDEO_CHATGPT5_TRAIN_PATH, VIDEO_CHATGPT_EVAL_PATH],
    "uvqa_relation_negative": UVQA_TRAIN_NEG_REL_PATH,
    "uvqa_object_negative": UVQA_TRAIN_NEG_OBJ_PATH,
    "uvqa_attribute_negative": UVQA_TRAIN_NEG_ATT_PATH,
    "video_chatgpt1_negative": VIDEO_CHATGPT1_TRAIN_NEG_PATH,
    "video_chatgpt2_negative": VIDEO_CHATGPT2_TRAIN_NEG_PATH,
    "video_chatgpt3_negative": VIDEO_CHATGPT3_TRAIN_NEG_PATH,
    "video_chatgpt4_negative": VIDEO_CHATGPT4_TRAIN_NEG_PATH,
    "video_chatgpt5_negative": VIDEO_CHATGPT5_TRAIN_NEG_PATH,
}
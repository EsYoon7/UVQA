import os
import json

file = VIDEO_CHATGPT_JSON_PATH

with open(file, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    new_item = {}
    new_item['vid'] = item['video_id']
    new_item['new_question'] = item['q']
    new_item['new_answer'] = item['a']
    new_data.append(new_item)

save_file = SAVE_VIDEO_CHATGPT_JSON_PATH

with open(save_file, 'w') as f:
    json.dump(new_data, f, indent=4)
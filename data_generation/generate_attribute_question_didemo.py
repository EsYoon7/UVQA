import json
import ipdb
import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
import openai
from openai import OpenAI
import concurrent.futures
import json
import time
import nltk
import webcolors

from utils import *


def get_arguments():
    parser = argparse.ArgumentParser(description='generate unanswerable question on relation category')

    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--data_root_dir", type=str, default="/data2/esyoon_hdd")
    parser.add_argument("--root_dir", type=str, default="/data/kakao/workspace/answerability_alignment")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_file_name", type=str, default="debug")
    parser.add_argument("--debugging", action='store_true')

    parser.add_argument(
    "--api_key",
    type=str,
    help="OpenAI API key"
    )

    parser.add_argument(
    "--gpt_version",
    choices=["gpt3.5", "gpt4o", "gpt4-turbo"],
    default="gpt4-turbo"
    )

    parser.add_argument(
    "--multi_process",
    action="store_true"
    )

    parser.add_argument(
        "--num_generations",
        type=int,
        default=12000
    )

    args = parser.parse_args()

    return args

PATTERN_DICT = {
    "pattern_1": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n<|Start Of Instruction|>\n{}\n<|End Of Instruction|>\n",
}

INSTRUCTION ={"instruction_1": "Given the relation provided in the video, generate a new object-focused question that could be asked based on the observed objects or subjects. However, the specific relation mentioned does not appear in the video, making the question unanswerable. The answer should indicate that the question is unanswerable due to the absence of the mentioned relation in the video. The question should focus on different aspects of the identified object or subject. The possible question types could include, but are not limited to:\n \
* Asking about the color of the object * Inquiring about the shape of the object * Questioning the state or condition of the object (e.g., solid, liquid, old, new) \
* Asking about the size or dimensions of the object * Inquiring about the material or substance the object is made from * Asking about the function or use of the object \
* Inquiring about the quantity or number of objects * Asking about the location or place where the object can be found * Questioning the ownership or association of the object \
* Asking about the different parts or components of the object\nEnsure the relation should be mentioned in the generated question. Aim to create a mix of question types to cover different aspects of the object, promoting a deeper understanding and engagement. Here is the unrelated relation: \n<|Start of Unrelated Relation|>\n{}\n<|End of Unrelated Relation|>. If the given relation doesn't make sense, please output 'None' for the both question and answer based on the origianl correct relation. \n<|Start of Related Relation|>\n{}\n<|End of Related Relation|> Otherwise, output your generated question and answer pair by strictly following this format: '<|Start of Question|>\n[question]\n<|End of Question|>\n<|Start of Answer|>\n[answer]\n<|End of Answer|>'",

"instruction_2": "Given the relation provided in the video, generate a new question that could be asked based on the observed content. However, the specific relation mentioned does not appear in the video, making the question unanswerable because the object, or subject being asked about is not present in the video. The answer should indicate that the question is unanswerable due to the absence of the mentioned object, or subject in the video. The question can focus on various aspects, including but not limited to:\n \
* The characteristics of an object (e.g., color, shape, size) * The actions or behaviors of a subject * The state or condition of something (e.g., solid, liquid, old, new) \
* The number or quantity of objects or events * The location or place where something is found * The relationship or association between objects, subjects, or events \
* The timing or sequence of events * The relationship or association between objects, subjects, or events * The parts or components of an object or subject \nEnsure the relation should be mentioned in the generated question. Aim to create a variety of question types to cover different aspects, promoting a deeper understanding and engagement.\n\
Based on given relation in the video, the unrelated relation to the video is generated. Here is two relations:\n<|Start of Related Relation|>\n{}\n<|End of Related Relation|>\n<|Start of Unrelated Relation|>\n{}\n<|End of Unrelated Relation|>. Focus that in the given relations the obejct mentioned in the video is '{}' and the obejct not appear is '{}'. If the given relation doesn't make sense, please generate 'Not applicable' for both the question and answer. Otherwise, output your generated question and answer pair by strictly following this format: '<|Start of Question|>\n[question]\n<|End of Question|>\n<|Start of Answer|>\n[answer]\n<|End of Answer|>'",

"instruction_3": "Given the query provided in the video, generate a new question that could be asked based on the observed content. However, the query mentioned does not appear in the video, making the question unanswerable because the attribute of object, or subject being asked about is not present in the video. The answer should indicate that the question is unanswerable due to the absence of the mentioned attribute in the video. The question can focus on various aspects, including but not limited to:\n \
* The actions or behaviors of a subject * The state or condition of something (e.g., solid, liquid, old, new) \
* The number or quantity of objects or events * The location or place where something is found * The relationship or association between objects, subjects, or events \
* The timing or sequence of events * The relationship or association between objects, subjects, or events * The parts or components of an object or subject \nEnsure the relation should be mentioned in the generated question. Aim to create a variety of question types to cover different aspects, promoting a deeper understanding and engagement.\n\
To help guide this, you'll be provided with two types of queries: query included the video content are:\n<|Start of Related Query|>\n{}\n<|End of Related Query|>\n and unrelated query is:\n<|Start of Unrelated Query|>\n{}\n<|End of Unrelated Query|>\n Focus that in the given query, attributes of the obejct mentioned in the video are '{}' and the attributes not appear are '{}'.\
Step:\n1. If the unrelated query is grammatically incorrect or similar to the given related query, output 'Not applicable' without format.\n2. If the unrelated query is valid, generate an unanswerable question and answer stating that the question is unanswerable and explaining why it's unanswerable, following this strict format:  '<|Start of Question|>\n[question]\n<|End of Question|>\n<|Start of Answer|>\n[answer]\n<|End of Answer|>'"
}

def get_client_response(args, origianl_query, unanswerable_query, orig_attributes, new_attributes):
    openai.api_key = args.api_key
    client = OpenAI(api_key=args.api_key)

    end_signal = "<|End Of Answer|>"
    instruction = INSTRUCTION['instruction_3'].format(origianl_query, unanswerable_query, orig_attributes, new_attributes)
    prompt = PATTERN_DICT["pattern_1"].format(instruction)
    import ipdb; ipdb.set_trace()
    if args.gpt_version == "gpt3.5":
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=1.0,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_tokens=2048,
            stop=[end_signal]
        )
        return prompt, response.choices[0].text.strip()

    elif args.gpt_version == "gpt4-turbo":
        # response = "this is dummy"
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and harmless assistant"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            stop=[end_signal]
            )
        return prompt, response.choices[0].message.content.strip()
    elif args.gpt_version == "gpt4o":
        # response = "this is dummy"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful and harmless assistant"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            stop=[end_signal]
            )
        return prompt, response.choices[0].message.content.strip()

def query_tagging(query):
    tagged_query = nltk.pos_tag(nltk.word_tokenize(query))
    return tagged_query

def extract_qeury_list(data):
    query_list = []
    for v in data:
        query_list.append(v["description"])
    return query_list

def get_candidate_attribute_list(tokenized_query, attribute_lists):
    output_attribute_dict = {}

    for k, v in attribute_lists.items():
        for token in tokenized_query:
            if token in v:
                output_attribute_dict[token] = v
    return output_attribute_dict
    

def main():
    args = get_arguments()
    if args.run_name is None:
        args = set_runname(args)
    args = set_device(args)
    if not args.debugging:
        args =  set_output_setting(args, question_type="attribute")

    didemo_annotation_file = os.path.join(args.data_root_dir, "DiDeMo", "data", "train_data.json")
    # didemo_annotation_file = os.path.join(args.data_root_dir, "DiDeMo", "data", "test_data.json")
    didemo_data = load_json(didemo_annotation_file)
    # data_list, data_dict = process_a2dre_data(didemo_data)
    query_list = extract_qeury_list(didemo_data)
    tagged_query_list = [query_tagging(query) for query in query_list]
    attribute_list = []
    for query in tagged_query_list:
        for item in query:
            if item[1] == "JJ":
                attribute_list.append(item[0])
    attribute_list = set_unique_list(attribute_list)
    attribute_list.remove("uniform")
    attribute_list.remove("next")

    candidate_color_list = webcolors.names()
    color_list_in_attribute = [item for item in attribute_list if item in candidate_color_list]
    color_list_in_attribute += ["greenish", "reddish", "brownish", "yellowish", "blueish", "pinkish", "purpleish", "orangeish", "blackish", "whiteish", "grayish", "red-colored", "blue-green", "red-ish", "reddish-brown", 'brown/red']
    position_attribute = ["left", "right", "top", "bottom", "center", "middle", "back"]
    pattern_attribute = ["striped", "dotted", "solid", "patterned", "plain", "checkered", "floral", "geometric", "textured", "netted", "printed"]
    material_attribute = ["metal", "wood", "plastic", "glass", "paper", "cloth", "fabric", "leather", "fur"]
    size_attribute = ["big", "small", "large", "tiny", "short", "long", "thick", "thin", "huge", "shoulder-length"]
    status_attribute = ["old", "new", "clean", "dirty", "broken", "damaged", "worn", "torn", "used", "unused", "naked", "pale", "molten", "beautiful", "decorated", "zoomed-in", "zoomed", "grown", "useless", 
                        "distinct", "hard", "closed-up", "close-up", "prolonged", "decorative", "chopped", "wonderful", "animated", "clear"]
    shape_attribute = ["round", "square", "triangle", "rectangular", "oval", "circular", "cylindrical", "conical", "spherical", "elliptical", "hexagonal", "octagonal", "pentagonal", "pyramidal", "cubical", "roundish", "rounded"]
    human_status = ["bored", "passionate", "happy", "sad", "angry", "excited", "tired", "sleepy", "hungry", "full", "thirsty", "drunk", "sober", "sick", "healthy", "injured", "wounded", "dead", "alive", "naked", "clothed", "upset", "disabled", "hungry",
                    "dressed", "undressed", "sweaty", "shiny", "dull", "pale", "tanned", "burnt", "bruised", "scared", "frightened", "calm", "relaxed", "stressed", "anxious", "worried", "confused", "focused", "distracted", "alert", "drowsy", "sleepy", "awake", "asleep", "horrible"]


    attribute_lists = {
        "color": color_list_in_attribute,
        "position": position_attribute,
        "pattern": pattern_attribute,
        "material": material_attribute,
        "size": size_attribute,
        "status": status_attribute,
        "shape": shape_attribute,
        "human_status": human_status
    }
    video_iterator = iter(didemo_data)
    idx = 0
    generated_unrelated_data = []
    with tqdm(total=args.num_generations) as pbar:
        while True:
            try:
                data_item = next(video_iterator)
            except:
                video_iterator = iter(didemo_data)
                data_item = next(video_iterator)

    # for video_id in tqdm(data_dict.keys()):
            # for idx, query in enumerate(data):
            query = data_item["description"]
            splited_query = nltk.word_tokenize(query)
            candidate_attributes_dict = get_candidate_attribute_list(splited_query, attribute_lists)
            candidate_selected_attribute_dict = {}
            for k, v in candidate_attributes_dict.items():
                if len(v) > 0:
                    while True:
                        candidate_selected_attribute_dict[k] = random.choice(v)
                        if k not in candidate_selected_attribute_dict[k] and candidate_selected_attribute_dict[k] not in k and candidate_selected_attribute_dict[k] != k:
                            break
        
            for k, v in candidate_selected_attribute_dict.items():
                generated_query = query.replace(k, v)

                item = {"vid": data_item['video'],
                    "orig_relation": query,
                    "generated_relation": generated_query,
                    "orig_attribute": k,
                    "generated_attributes": v
                }
                prompt, response = get_client_response(args, query, generated_query, list(candidate_selected_attribute_dict.keys()), list(candidate_selected_attribute_dict.values()))

                item['prompt'] = prompt
                item['response'] = response

                generated_unrelated_data.append(item) 

                pbar.update(1)
                idx += 1


            # save the generated data
            if not args.debugging:
                with open(args.output_fname, 'w') as f:
                    json.dump(generated_unrelated_data, f, indent=4)

            if len(generated_unrelated_data) > args.num_generations:
                break

    if not args.debugging:
        with open(args.output_fname, 'w') as f:
            json.dump(generated_unrelated_data, f, indent=4)



        
if __name__ == '__main__':
    main()
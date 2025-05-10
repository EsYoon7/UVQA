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

from utils import *


def get_arguments():
    parser = argparse.ArgumentParser(description='generate unanswerable question on relation category')

    parser.add_argument("--seed", type=int, default=4096)
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

INSTRUCTION ={"instruction_1": "You will be provided with a relation that does not appear in a video. Based on the objects or subjects in the video, your task is to generate a new, object-focused question that cannot be answered because the provided relation is unrelated to what is shown. The possible question types could include, but are not limited to:\n \
* Asking about the color of the object * Inquiring about the shape of the object * Questioning the state or condition of the object (e.g., solid, liquid, old, new) \
* Asking about the size or dimensions of the object * Inquiring about the material or substance the object is made from * Asking about the function or use of the object \
* Inquiring about the quantity or number of objects * Asking about the location or place where the object can be found * Questioning the ownership or association of the object \
* Asking about the different parts or components of the object\nYou should include the generated unrelated relation in the question to make the question unanswerable. Additionally, ensure that your question covers different aspects of the video to encourage a deeper understanding. The relations are observed in the videos are:\n<|Start of Related Relation|>\n{}\n<|End of Related Relation|>\n\
Here is the unrelated relation you should include: \n<|Start of Unrelated Relation|>\n{}\n<|End of Unrelated Relation|>\n Step:\n1. If the unrelated relation is grammatically incorrect or matches one of the related relations, output 'Not applicable' without format.\n2. If the unrelated relation is valid, generate an unanswerable question and answer stating that the question is unanswerable and explaining why it's unanswerable, following this strict format:  '<|Start of Question|>\n[question]\n<|End of Question|>\n<|Start of Answer|>\n[answer]\n<|End of Answer|>'"
}
def get_same_char_relation(relation_in_vid, all_relations):

        target_rel = all_relations[relation_in_vid]

        same_rel_type_relation_list = []
        same_char_relation_list = []
        same_node_type_list = []
        for k, v in all_relations.items():
            if v['type'] == target_rel['type']:
                same_rel_type_relation_list.append(k)
            if v['characteristic'] == target_rel['characteristic']:
                same_char_relation_list.append(k)
            if v['nodes'] == target_rel['nodes']:
                same_node_type_list.append(k)
        
        return [same_rel_type_relation_list, same_char_relation_list, same_node_type_list]


def get_client_response(args, unanswerable_relation, origianl_correct_relation):
    openai.api_key = args.api_key
    client = OpenAI(api_key=args.api_key)

    end_signal = "<|End Of Answer|>"
    instruction = INSTRUCTION['instruction_1'].format(origianl_correct_relation, unanswerable_relation)
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
            max_tokens=2048,
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
def main():
    args = get_arguments()
    set_seed(args)

    if args.run_name is None:
        args = set_runname(args)
    
    args = set_device(args)
    # if not args.debugging:
    #     args = set_output_setting(args, question_type="relation")

    moma_annotation_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "anns.json")
    actor_sact_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "act_sact.json")
    actor_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "actor.json")
    attribute_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "attribute.json")
    intransitive_action_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "intransitive_action.json")
    lvis_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "lvis.json")
    object_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "object.json")
    relationship_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "relationship.json")
    transitive_action_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "taxonomy",  "transitive_action.json")

    data_split_file = os.path.join(args.data_root_dir, "MOMA-LRG", "anns", "splits",  "standard.json")


    moma_annotation = load_json(moma_annotation_file)
    actor_sact = load_json(actor_sact_file)
    actor = load_json(actor_file)
    attribute = load_json(attribute_file)
    intransitive_action = load_json(intransitive_action_file)
    lvis = load_json(lvis_file)
    object_ = load_json(object_file)
    relationship = load_json(relationship_file)
    transitive_action = load_json(transitive_action_file)

    data_split = load_json(data_split_file)
    test_vids = data_split['test']
    # test_vids = data_split['train']
    annotation_types = ['attributes', 'relationships', 'intransitive_actions', 'transitive_actions']
    annotation_data_list = [attribute, relationship, intransitive_action, transitive_action]

    all_relations = {}
    all_relations_list = []
    for annotation_type, annotation_data in zip(annotation_types, annotation_data_list):
        for k, v in annotation_data.items():
            for v_item in v:
                all_relations_list.append(v_item[0])
                all_relations[v_item[0]] = {"nodes": v_item[1:], "type": annotation_type, "characteristic": k}


    processed_data = {}
    merged_processed_data = {}
    split_processed_data = {}
    rel_processed_data = {}
    specific_actor_candiates = {}
    specific_object_candiates = {}
    



    for item in tqdm(moma_annotation):
        video_id = item['file_name'][:-4]
        if video_id in test_vids:
            rel_dict = {type_ : [] for type_ in annotation_types}
            object_list = []
            actor_list = []
            raw_object_list = [] # without numbering with multiuple object
            raw_actor_list = [] # without numbering with multiuple actor
            sub_activity_list = []
            rel_list = []
            rel_only_list = []
            rel_dict_key_with_action = {}

            for sub_act in item['activity']['sub_activities']:
                sub_activity_list.append(sub_act['class_name'])
                interactions = sub_act['higher_order_interactions']
                for interaction in interactions:
                    for rel_type in annotation_types:
                        if interaction[rel_type] == []:
                            pass
                        else:
                            for rel_item in interaction[rel_type]:
                                actor_list += [i['class_name']+'_' + i['id'] for i in interaction['actors']]
                                object_list += [i['class_name']+'_' + i['id'] for i in interaction['objects']]
                                raw_actor_list += [i['class_name']for i in interaction['actors']]
                                raw_object_list += [i['class_name']for i in interaction['objects']]
                                
                                actor_list = set_unique_list(actor_list)
                                object_list = set_unique_list(object_list)
                                raw_actor_list = set_unique_list(raw_actor_list)
                                raw_object_list = set_unique_list(raw_object_list)

                                rel = ''
                                src_obj = None
                                trg_obj = None

                                if "source_id" in list(rel_item.keys()):
                                    if rel_item['source_id'].isalpha(): # actors 
                                        src_obj = [i['class_name'] for i in interaction['actors'] if i['id'] == rel_item['source_id']]
                                    elif rel_item['source_id'].isdigit(): # objects
                                        src_obj = [i['class_name'] for i in interaction['objects'] if i['id'] == rel_item['source_id']]
                                    else:
                                        import ipdb; ipdb.set_trace() # check what is the source_id 

                                    src_obj = src_obj[0]
                                    rel = rel_item['class_name'].replace('[src]',  src_obj)
                                
                                if "target_id" in list(rel_item.keys()):
                                    if rel_item['target_id'].isalpha(): # actors 
                                        trg_obj = [i['class_name'] for i in interaction['actors'] if i['id'] == rel_item['target_id']]
                                    elif rel_item['target_id'].isdigit(): # objects
                                        trg_obj = [i['class_name'] for i in interaction['objects'] if i['id'] == rel_item['target_id']]
                                    else:
                                        import ipdb; ipdb.set_trace() # check what is the target_id 

                                    trg_obj = trg_obj[0]
                                    rel = rel.replace('[trg]',  trg_obj)

                                rel_dict[rel_type].append(rel)
                                rel_list.append(rel)
                                rel_only_list.append(rel_item['class_name'])

                                if rel_item['class_name'] in rel_dict_key_with_action.keys():
                                    if rel in rel_dict_key_with_action[rel_item['class_name']]['relation']:
                                        pass
                                    else:
                                        rel_dict_key_with_action[rel_item['class_name']]['relation'].append(rel)
                                        rel_dict_key_with_action[rel_item['class_name']]['object'].append((src_obj, trg_obj))
                                else:
                                    rel_dict_key_with_action[rel_item['class_name']] = {'relation': [rel], "object": [(src_obj, trg_obj)]}

                                
                rel_dict = set_unique_dict(rel_dict)
                rel_list = set_unique_list(rel_list)
                rel_only_list = set_unique_list(rel_only_list)
                

                split_processed_data[video_id+'_'+sub_act['id']] = [{
                        "actors":actor_list,
                        "objects": object_list,
                        "rel_dict" : rel_dict,
                        "rel_list": rel_list,
                        "rel_only_list": rel_only_list,
                    }]
                
                if video_id in processed_data.keys():
                    processed_data[video_id].append({
                        "actors":actor_list,
                        "objects": object_list,
                        "rel_dict" : rel_dict,
                        'rel_list': rel_list,
                        "rel_only_list": rel_only_list,
                    })

                    for key, dict_item_ in zip(['actors', 'objects', 'rel_dict', 'rel_list', 'rel_only_list'], [raw_actor_list, raw_object_list, rel_dict, rel_list, rel_only_list]):
                        if key != "rel_dict":
                            merged_processed_data[video_id][key] += dict_item_
                            merged_processed_data[video_id][key] = list(set(merged_processed_data[video_id][key]))
                        elif key == "rel_dict":
                            for rel_key in dict_item_.keys():
                                merged_processed_data[video_id][key][rel_key] += dict_item_[rel_key]
                                merged_processed_data[video_id][key][rel_key] = list(set(merged_processed_data[video_id][key][rel_key]))
                        else: # check the data format
                            import ipdb; ipdb.set_trace()
                        

                else:   
                    processed_data[video_id] = [{
                        "actors":actor_list,
                        "objects": object_list,
                        "rel_dict" : rel_dict,
                        'rel_list': rel_list,
                        "rel_only_list": rel_only_list,
                        
                    }]

                    merged_processed_data[video_id] = {}
                    for key, dict_item_ in zip(['actors', 'objects', 'rel_dict', 'rel_list', "rel_only_list"], [raw_actor_list, raw_object_list, rel_dict, rel_list, rel_only_list]):
                        if key != "rel_dict":
                            merged_processed_data[video_id][key] = dict_item_
                        elif key == "rel_dict":
                            merged_processed_data[video_id][key] = {}
                            for rel_key in dict_item_.keys():
                                merged_processed_data[video_id][key][rel_key] = dict_item_[rel_key]
                        else:
                            import ipdb; ipdb.set_trace() # check dataset form

            rel_processed_data[video_id] = rel_dict_key_with_action
                

            specific_actor_candiates[video_id] = {key: get_list_from_dict_item(actor, key) for key in raw_actor_list}
            specific_object_candiates[video_id] = {key: get_list_from_dict_item(object_, key) for key in raw_object_list}
    
    large_actor_candidates = get_list_from_dict(actor)
    # unanswerable question 


    # generate unrelated pharase
    generated_unrelated_data = []
    vids_iterator = iter(test_vids)
    idx = 0
    with tqdm(total=args.num_generations) as pbar:

        # for idx, video_id in enumerate(tqdm(test_vids)):
        while True:
            try:
                video_id = next(vids_iterator)
            except:
                vids_iterator = iter(test_vids)
                video_id = next(vids_iterator)
        
            relation_list = merged_processed_data[video_id]['rel_only_list']
            try:
                random_chosen_rel = random.choice(relation_list)
            except:
                continue

            relation_in_vid_dict = rel_processed_data[video_id][random_chosen_rel]
            relation_in_vid = relation_in_vid_dict['relation']
            try:
                selected_related_for_gen = random.choice(relation_in_vid)
            except:
                continue

            actor_candidate = specific_actor_candiates[video_id]
            object_candidate = specific_object_candiates[video_id]

            rel_types = get_same_char_relation(random_chosen_rel, all_relations)
            rel_types = [item for sublist in rel_types for item in sublist]
            rel_types = set_unique_list(rel_types)
            rel_types.remove(random_chosen_rel)

            try:
                # unrelated_rel = random.choice(rel_types[random.choice(range(len(rel_types)))])
                unrelated_rel = random.choice(rel_types)
            except:
                continue
            pruned_unrelated_rel = unrelated_rel.replace("[src]", "").strip()
            pruned_unrelated_rel = pruned_unrelated_rel.replace("[trg]", "").strip()

            pruned_related_rel = random_chosen_rel.replace("[src]", "").strip()
            pruned_related_rel = pruned_related_rel.replace("[trg]", "").strip()

            relation_sposition = selected_related_for_gen.find(pruned_related_rel)
            rel_length = len(pruned_related_rel)


            generated_rel = selected_related_for_gen[:relation_sposition] + pruned_unrelated_rel + selected_related_for_gen[relation_sposition+rel_length: ]
            
            item = {"vid": video_id,
                    "orig_relation": relation_in_vid,
                    "generated_relation": generated_rel
            }

            prompt, response = get_client_response(args, generated_rel, relation_in_vid)
            # get_client_response(args, generated_rel, relation_in_vid)

            item['prompt'] = prompt
            item['response'] = response
            generated_unrelated_data.append(item)   

            # save the generated data
            if not args.debugging:
                with open(args.output_fname, 'w') as f:
                    json.dump(generated_unrelated_data, f, indent=4)

            pbar.update(1)
            idx += 1

            if len(generated_unrelated_data) > args.num_generations:
                break

    # save the generated data
    if not args.debugging:
        with open(args.output_fname, 'w') as f:
            json.dump(generated_unrelated_data, f, indent=4)



if __name__ == "__main__":
    if not torch.cuda.is_available(): #
        print("Need available GPU(s) to run this model...") #
        quit() #

    main()

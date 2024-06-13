# %%
import sys
import os

# %%
import os
import math
import random
from time import sleep
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import decord
decord.bridge.set_bridge('torch')

# %%
import openai
import tiktoken
with open('api_mine.key') as f:
    client = openai.OpenAI(api_key=f.read().strip())

# %%
EGO4D_ANN_PATH = '/vision/group/ego4d/v2/annotations'
assert '2' in EGO4D_ANN_PATH
EGO4D_SPLIT = 'train' # 'train', 'val'
print(f"Using EGO4D split: {EGO4D_SPLIT}")

# %%
CODEGEN_MODEL = "gpt-4"
CODEGEN_MODEL_CONTEXT_LENGTH = {
    "gpt-4": 8192,
}
codegen_temperature = 0.0                            # Temperature for Codex. (Almost) deterministic if 0
codegen_max_tokens = 256                             # Maximum number of tokens to generate for Codex
# codegen_best_of = 1                                 # Number of tries to choose from. Use when temperature > 0
MAX_TRIES = 2

# %%
SYSTEM_PROMPT = """You're helping me generate a new dataset for an online assistant that receives a first person view of the world (via a head mounted camera, eg augmented reality glasses).
In short, will receive a question and must convert it into a query in which a human asks an assistant to identify the point at which the event starts.
Eg. {
    "query": "Let me know when it's safe to cross the street.",
    "alert": "You may cross the street now."
}

You won't have access to the full video, instead I'll provide narrations of the events in the video, and high level summaries.
Use them to enrich the query with context (eg. say "talk to the cashier" or "interact with person X" instead of "converse/interact with someone"), but remember to keep the query grounded in the video (*do not* invent details).
Importantly, the narrations might contain mistakes, especially with the language.

The videos are in first person and `#C` ALWAYS refers to the camera wearer, ie. the person that is using the assistant. #O refers to others in the video.

If the event B has already occurred before, make sure that the query is unambiguous. For example, by referring to another event A that happened before.

You should only return a JSON file with a single query formatted as indicated (with the keys in the same order):
{
    "question": string,
    "event": string,
    "event_is_grounded_in_narrations": true|false,
    "event_has_occurred_before": true|false,
    "request": string,
    "query": string,
    "alert": string,
    "query_is_specific_only_to_this_instance_of_event": true|false
}
"""

# %%
examples = [
    {
        "question": "what did I pick from the fridge?",
        "event": "pick from fridge",
        "event_is_grounded_in_narrations": False,
        "event_has_occurred_before": False,
        "request": "reminder to check expriration date of milk.",
        "query": None,
        "alert": "Remember to check the expiration date of the milk.",
        "query_is_specific_only_to_this_instance_of_event": True
    },
]
example_queries = [
    "Next time I pick something from the fridge, remind me to check the expiration date of the milk.",
    "Ask me to check the expiration date of the milk when I pick something from the fridge.",
    "When I pick something from the fridge, remind me to check the expiration date of the milk.",
    "As I start picking something from the fridge, remind me to check the expiration date of the milk.",
]
example_disambiguated_queries = [
    "Next time I'm picking something from the fridge after arranging the groceries, remind me to check the expiration date of the milk.",
    "Ask me to check the expiration date of the milk when I'm picking something from the fridge after arranging the groceries.",
    "When I pick something from the fridge after arranging the groceries, remind me to check the expiration date of the milk.",
    "As I start picking something from the fridge after arranging the groceries, remind me to check the expiration date of the milk.",
]


# %%
word_blacklist = {
    'before',   # can't answer in streaming mode
    'talk', 'chat', 'discuss', 'converse', 'interact'   # narrations tipically don't mention conversations
}

template_whitelist = {
    'Objects: What did I put in X?',
    'Place: Where did I put X?',                              # narrations might not have enough context for this (fixed w/ event_is_grounded_in_narrations)
    'Objects: Where is object X before / after event Y?',     # before/after is too hard to generate queries for (after is much easier, before should blacklisted)
    # 'People: When did I talk to or interact with person with role X?',    # narrations tipically don't mention conversations (OR ROLES)
    # 'Objects: How many X’s? (quantity question)',             # quantity questions are hard to generate queries for
    'Objects: State of an object',                            # narrations might not have enough context for this
    # 'Objects: Where is object X?',                            # narrations might not have enough context for this (objects that are not interacted with might not be mentioned) (fixed w/ event_is_grounded_in_narrations)
    # 'Objects: In what location did I see object X ?',
    'Objects: What X did I Y?',
    'Objects: What X is Y?',
    # 'Objects: Where is my object X?',
    # None,                                                     # no template
    # 'People: Who did I interact with when I did activity X?',   # risky, but event_is_grounded_in_narrations should help
    # 'People: Who did I talk to in location X?',               # narrations tipically don't mention conversations, and locations are not always mentioned
}


# %% [markdown]
# ## Utils

# %%
tokenizer = tiktoken.encoding_for_model(CODEGEN_MODEL)
prompt_n_tokens = len(tokenizer.encode(SYSTEM_PROMPT))
def count_total_tokens(text):
    return len(tokenizer.encode(text)) + prompt_n_tokens + 11 # 11 is estimate for the roles (ie. system, user)


# %%
def load_video(path, num_frames=32):
    vr = decord.VideoReader(path)
    batch_frame_idxs = np.linspace(0, len(vr)-1, num_frames, endpoint=True, dtype=int)
    video_frames = vr.get_batch(batch_frame_idxs)           # (T, H, W, C)
    video_frames = video_frames.permute(0, 3, 1, 2)       # to (T, C, H, W)
    return video_frames

# %%
def get_video_metadata(path):
    vr = decord.VideoReader(path)

def get_video_fps(path):
    vr = decord.VideoReader(path)
    return vr.get_avg_fps()

# %%
def show_video(video_frames, max_frames=8):
    # subsample frames
    # video_frames = video_frames[::len(video_frames)//max_frames]
    # using np.linspace(0, len(video_frames)-1, max_frames, endpoint=True, dtype=int)
    video_frames = video_frames[np.linspace(0, len(video_frames)-1, max_frames, endpoint=True, dtype=int)]

    n_rows = math.ceil(len(video_frames) / 8)
    n_cols = min(len(video_frames), 8)

    # subplots for each frame
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2*n_rows))
    for i, frame in enumerate(video_frames):
        if n_rows == 1:
            axes[i].imshow(frame.permute(1, 2, 0).numpy())
        else:
            axes[i//n_cols, i%n_cols].imshow(frame.permute(1, 2, 0).numpy())

    [ax.axis('off') for ax in axes.ravel()]
    plt.tight_layout()
    plt.show()

def get_video_frame(path, time: float = None, frame_idx: int = None):
    # time is in seconds, e.g. 1.5
    vr = decord.VideoReader(path)
    if time is not None:
        frame_idx = int(time * vr.get_avg_fps())
    frame = vr.get_batch([frame_idx]).permute(0, 3, 1, 2).squeeze(0)      # to (C, H, W)
    return frame

def show_frame(frame):
    plt.imshow(frame.permute(1, 2, 0).numpy())   # to (H, W, C)
    plt.axis('off')
    plt.show()

# %%
def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int((seconds % 3600) % 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'  # Use :02d to ensure two digits with leading zeros

# %% [markdown]
# ### Load narration data
# Each video has a `status` and two narration passes (`narration_pass_1` and `narration_pass_2`), each with `summaries` and event `narrations`.

# %% load narrations
def load_narrations():
    with open(os.path.join(EGO4D_ANN_PATH, 'narration.json'), 'r') as f:
        annotations = json.load(f)
    # keep only the annotations that have status 'complete' (others are 'redacted', 'redacted_partial')
    annotations = {vid_id: desc for vid_id, desc in annotations.items() if desc['status'] == 'complete'}
    return annotations

# %% [markdown]
# ### Load nlqs data
# - 948 videos in train set, but 885 have narration data
# - X videos in val set, but Y have narration data

# %%
def load_nlqs(narrations):
    with open(os.path.join(EGO4D_ANN_PATH, f'nlq_{EGO4D_SPLIT}.json'), 'r') as f:
        nlqs = json.load(f)

    for vid_idx, vid_nlqs in enumerate(nlqs['videos']):
        video_uid = vid_nlqs['video_uid']
        vid_clips_nlqs = vid_nlqs['clips']
        for clip_idx, clip in enumerate(vid_clips_nlqs):
            clip_uid = clip['clip_uid']
            for annotator_idx, full_ann in enumerate(clip['annotations']):
                annotation_uid = full_ann['annotation_uid']
                for ann_idx, ann in enumerate(full_ann['language_queries']):
                    if 'query' in ann:
                        ann['question'] = ann['query']
                        del ann['query']   # remove the query to regenerate it


    # filter nlqs that don't have narrations
    print(f"Number of videos with nlqs data: {len(nlqs['videos'])}")
    nlqs['videos'] = [vid_nlqs for vid_nlqs in nlqs['videos'] if vid_nlqs['video_uid'] in narrations]
    print(f"Number of videos with both nlq and narration data: {len(nlqs['videos'])}")
    return nlqs

# %% [markdown]
# ### Generate Queries

# %% generate script
def generate_script(ann_question, ann_start_time, ann_end_time, video_narrations):
    script = ""

    # specify the event we want to generate a query for
    script += f"Event you should generate a \"start\" query for: \"{ann_question}\" ({convert_seconds(ann_start_time)} - {convert_seconds(ann_end_time)})\n"

    script += f"\nVIDEO NARRATIONS:\n"
    # add high level description
    script += f"The high level descriptions of the video are:\n"
    for summary in video_narrations['summaries']:
        summary_text = summary['summary_text']
        start_sec = summary['start_sec']
        end_sec = summary['end_sec']

        if start_sec > ann_end_time:
            break
        script += f'- ({convert_seconds(start_sec)} - {convert_seconds(end_sec)}) {summary_text}\n'

    # add event narrations
    script += f"\n\nThe low level events in the video are:\n"
    # (sort by start time just in case)
    for event in sorted(video_narrations['narrations'], key=lambda x: x['timestamp_sec']):
        event_text = event['narration_text']
        event_text = event_text.replace('# #unsure', '(unsure)')
        event_sec = event['timestamp_sec']

        if (event_sec > ann_start_time) and not '>>>> EVENT STARTS HERE' in script:
            script += f'>>>> EVENT STARTS HERE ({convert_seconds(ann_start_time)}) <<<<\n'

        if event_sec > ann_end_time:
            break
        script += f'- ({convert_seconds(event["timestamp_sec"])}) {event_text}\n'
    script += f'>>>> EVENT ENDS HERE ({convert_seconds(ann_end_time)}) <<<<\n'

    # reminder to format query in first person
    script += f"\n\nRemember to format the query in first person, as if the user is asking the assistant. (like the examples in the prompt).\n"
    script += "Eg. {}\n"

    # specify the event we want to generate a query for (again)
    script += f"\nAgain, the event you should generate the query for is \"{ann_question}\" (occurs between {convert_seconds(ann_start_time)} - {convert_seconds(ann_end_time)})\n"
    script += f"Remember that query should be a reminder to do something when the EVENT STARTS (event starts to occur).\n"
    script += f"Remember to make the query unambiguous if the event has already occurred before (if `event_has_occurred_before: true`). If the query is not specific *only* to the given instance of the event, then set `query_is_specific_only_to_this_instance_of_event: false`.\n"
    return script

# %% llm generate query
def llm_generate_query(script, example, example_disambiguated, prev_duplicate_label=False, temperature=codegen_temperature, max_tokens=codegen_max_tokens, top_p=1.0):
    cur_example = example
    if prev_duplicate_label:
        # try to use the disambiguated example (if it fits in the context)
        full_script = script.format(json.dumps(example_disambiguated, indent=4))
        if count_total_tokens(full_script) < (CODEGEN_MODEL_CONTEXT_LENGTH[CODEGEN_MODEL] - max_tokens):
            cur_example = example_disambiguated

    num_tries = 0
    while num_tries < MAX_TRIES:
        if num_tries > 0:
            print(f"Trying again (try {num_tries + 1} of {MAX_TRIES}).")

        # add the example to the script
        full_script = script.format(json.dumps(cur_example, indent=4))

        # try generating the query, if it fails for a known reason, try again. If it fails for an unknown reason, raise the error.
        try:
            # print(script)
            response = client.chat.completions.create(
                model=CODEGEN_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_script},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,  # top_p=1.0
                frequency_penalty=0,
                presence_penalty=0,
                # best_of=codegen_best_of,
                stop=["\n\n"],
            )

            # make sure output is json parseable (if not, try again)
            llm_out = json.loads(response.choices[0].message.content)

            # make sure all the keys are present
            for k in example.keys():
                assert k in llm_out, f"Missing key: {k}"

            # if the label has been seen before, the model should have detected it from the narrations.
            if prev_duplicate_label:
                if not llm_out['event_has_occurred_before']:
                    llm_out['event_has_occurred_before'] = True                 # fix value.
                    llm_out['query_is_specific_only_to_this_instance_of_event'] = False     # it might not be specific to this event if the llm didn't detect the previous occurrence
                    raise AssertionError("Label has occurred before, but model did not detect it.")

            # model should not generate ambiguous queries for events that have occurred before (unavoidable in some cases, but worth it to at least try again?)
            if llm_out['event_has_occurred_before']:
                if not llm_out['query_is_specific_only_to_this_instance_of_event']:
                    prev_duplicate_label = True     # set this to True for the next iteration such that the model is forced to disambiguate next time (even if the next iteration doesn't detect the previous occurrence)
                    raise AssertionError("Ambiguous query.")

            # fix the query_is_specific_only_to_this_instance_of_event key when event_has_occurred_before is False
            # TODO: this is a hack, fix the prompt instead
            if not llm_out['event_has_occurred_before']:
                llm_out['query_is_specific_only_to_this_instance_of_event'] = True

            return llm_out
        except json.JSONDecodeError as e:
            # formatting error, try again
            print(e, end=" ... ")
            num_tries += 1
            # import pdb; pdb.set_trace()
        except AssertionError as e:
            print(str(e))
            num_tries += 1

            # if one of the keys was missing, try again
            # if "Missing key" in str(e):

            # if the label has occurred before, but the model didn't detect it, modify the script (if possible) and try again
            # if ("Label has occurred before" in str(e)):
            #     modded_script = script + "\nA SIMILAR EVENT HAS OCCURRED BEFORE. IF POSSIBLE, DISAMBIGUATE BY MAKING THE QUERY SPECIFIC TO THIS INSTANCE OF THE EVENT. ELSE, SET `query_is_specific_only_to_this_instance_of_event: false`.\n"
            #     full_modded_script = modded_script.format(json.dumps(cur_example, indent=4))
            #     if count_total_tokens(full_modded_script) < (CODEGEN_MODEL_CONTEXT_LENGTH[CODEGEN_MODEL] - max_tokens):
            #         script  = modded_script
            # if the query is ambiguous, modify the script (if possible) and try again
            if ("Ambiguous query." in str(e)):
                # # try to modify the script (if it fits in the context)
                # modded_script = script + "\nA SIMILAR EVENT HAS OCCURRED BEFORE. IF POSSIBLE, DISAMBIGUATE BY MAKING THE QUERY SPECIFIC TO THIS INSTANCE OF THE EVENT. ELSE, SET `query_is_specific_only_to_this_instance_of_event: false`.\n"
                # full_modded_script = modded_script.format(json.dumps(cur_example, indent=4))
                # if count_total_tokens(full_modded_script) < (CODEGEN_MODEL_CONTEXT_LENGTH[CODEGEN_MODEL] - max_tokens):
                #     script  = modded_script
                # try to use the disambiguated example (if it fits in the context)
                full_script = script.format(json.dumps(example_disambiguated, indent=4))
                if count_total_tokens(full_script) < (CODEGEN_MODEL_CONTEXT_LENGTH[CODEGEN_MODEL] - max_tokens):
                    cur_example = example_disambiguated
        except openai.BadRequestError as e:
            # if "This model's maximum context length is " in str(e):
            #     # shouldn't happen, since we're checking the length before generating the query, and the max_tokens is set accordingly
            #     # is_long_script[str(vid_idx)][str(clip_idx)][str(annotator_idx)][str(ann_idx)] = True
            #     break
            raise e # no idea what else could cause this, so raise to be safe
        except openai.RateLimitError as e:
            # ran out of tokens. halt and catch fire
            # eg. openai.RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details.[...]', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}
            print(e, end=" ... ")
            raise e
        except openai.APITimeoutError as e:
            # api error, wait and try again
            print(e, end=" ... ")
            print("Waiting 15 seconds before retrying...")
            sleep(15)
            num_tries += 1
        except openai.InternalServerError as e:
            # api error, wait and try again
            print(e, end=" ... ")
            print("Waiting 15 seconds before retrying...")
            sleep(15)
            num_tries += 1
        # except Exception as e:
        #     # some other error, raise it
        #     raise e
    print(f"Failed to generate query after {MAX_TRIES} tries. Returning last output.")
    return llm_out  # return the last output after MAX_TRIES, even if it failed

# %% main
# def main():
total_anns = 0
# load narrations
narrations = load_narrations()

# load nlqs
nlqs = load_nlqs(narrations)

# load previously generated queries
vids_with_gens = set()
if os.path.exists(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq.json')):
    with open(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq.json'), 'r') as f:
        existing_gens = json.load(f)
        vids_with_gens = {vid['video_uid'] for vid in existing_gens['videos']}
        print(f"Loaded {len(existing_gens['videos'])} videos with generated queries.")

# main loop, generate queries for each video, clip, annotator, annotation
new_nlqs = deepcopy(nlqs) # make a copy of nlqs to store the generated queries
is_duplicate  = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))) # to keep track of duplicate annotations for each video_idx, clip_idx, annotator_idx, annotation_idx
is_long_script = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))) # to keep track of long scripts for each video_idx, clip_idx, annotator_idx, annotation_idx
new_nlqs['videos'] = [] # remove the videos data, will add it back after adding the generated queries
os.makedirs(os.path.join('generated_queries', CODEGEN_MODEL), exist_ok=True)
for vid_idx, vid_nlqs in enumerate(tqdm(nlqs['videos'])):
    video_uid = vid_nlqs['video_uid']

    # skip videos that already have generated queries (from a previous run)
    if vid_nlqs['video_uid'] in vids_with_gens:
        assert existing_gens['videos'][vid_idx]['video_uid'] == video_uid   # make sure the video uids match
        new_nlqs['videos'].append(existing_gens['videos'][vid_idx])      # add the existing generated queries (so that we don't overwrite them)
        continue

    # make a deep copy of the nlqs data to store the generated queries
    vid_nlqs_copy = deepcopy(vid_nlqs)

    # store all the annotations we have already generated queries for in this video
    # to avoid generating queries for the same annotation twice (eg if two annotators annotate the same event)
    # (could have used vid_nlqs_copy directly, but this is more readable)
    this_vids_anns = list()

    # load narration data for the video
    video_narrations = narrations[video_uid]['narration_pass_1'] # TODO: choose one of the two narrations (or combine)

    # all annotations in the video, sorted by start time
    all_anns = list()
    vid_clips_nlqs = vid_nlqs['clips']
    for clip_idx, clip in enumerate(vid_clips_nlqs):
        clip_uid = clip['clip_uid']
        for annotator_idx, full_ann in enumerate(clip['annotations']):
            annotation_uid = full_ann['annotation_uid']
            for ann_idx, ann in enumerate(full_ann['language_queries']):
                if 'question' in ann:
                    ann_question = ann['question']
                    ann_start_time = ann['video_start_sec']
                    ann_end_time = ann['video_end_sec']
                    all_anns.append((ann_question, ann_start_time, ann_end_time, clip_idx, annotator_idx, ann_idx))
    all_anns = sorted(all_anns, key=lambda x: x[1]) # sort by start time

    # loop through all the annotations in the video, generating queries
    for ann_question, ann_start_time, ann_end_time, clip_idx, annotator_idx, ann_idx in all_anns:
        ann = vid_clips_nlqs[clip_idx]['annotations'][annotator_idx]['language_queries'][ann_idx]

        # ###############################################################
        # don't generate queries for blacklisted templates/words ########
        # ###############################################################
        if ann['template'] not in template_whitelist:
            # print(f"Skipping annotation with template: {ann['template']} ({convert_seconds(ann['video_start_sec'])} - {convert_seconds(ann['video_end_sec'])})")
            continue
        if any(word in ann['question'] for word in word_blacklist):
            # print(f"Skipping annotation with blacklisted word: {ann['question']} ({convert_seconds(ann['video_start_sec'])} - {convert_seconds(ann['video_end_sec'])})")
            continue

        # # ###############################################################
        # # don't generate queries for the same annotation twice ##########
        # # ###############################################################
        # check if the we already have queries for another annotation with significant temporal overlap
        for prev_ann in this_vids_anns:
            prev_ann_start = prev_ann['video_start_sec']
            prev_ann_end = prev_ann['video_end_sec']
            # calculate temporal overlap (intersection over union) - drop annotations with iou > 0.5
            intersection = max(0, min(prev_ann_end, ann_end_time) - max(prev_ann_start, ann_start_time))
            union = max(prev_ann_end, ann_end_time) - min(prev_ann_start, ann_start_time)
            iou = intersection / union if union > 0 else 0
            if iou > 0.5:   # significant temporal overlap
                # repeat annotation, skip generating query
                # TODO: refine temporal bounds of the query
                prev_ann_query = deepcopy(prev_ann['query'])
                prev_ann_query['duplicate'] = True
                vid_nlqs_copy['clips'][clip_idx]['annotations'][annotator_idx]['language_queries'][ann_idx]['query'] = prev_ann_query
                this_vids_anns.append(vid_nlqs_copy['clips'][clip_idx]['annotations'][annotator_idx]['language_queries'][ann_idx])
                break
        # skip annotations that already have queries (eg. those generated in a previous run or duplicates)
        if 'query' in vid_nlqs_copy['clips'][clip_idx]['annotations'][annotator_idx]['language_queries'][ann_idx]:
            # print(f"Skipping annotation that already has a query: {ann['question']} ({convert_seconds(ann['video_start_sec'])} - {convert_seconds(ann['video_end_sec'])})")
            is_duplicate[str(vid_idx)][str(clip_idx)][str(annotator_idx)][str(ann_idx)] = True
            continue

        # ###############################################################
        # generate script ###############################################
        # ###############################################################
        # choose example to generate query, in a replicable way
        sum_idxs = hash(video_uid + clip_uid + annotation_uid + ann_question)  # sum of the hash of the uids
        example_idx = sum_idxs % len(examples)
        example = examples[0]
        example_query = example_queries[example_idx]
        example['query'] = example_query
        example_disambiguated = deepcopy(example)
        example_disambiguated_query = example_disambiguated_queries[example_idx]
        example_disambiguated['event_has_occurred_before'] = True
        example_disambiguated['query'] = example_disambiguated_query
        example_disambiguated['query_is_specific_only_to_this_instance_of_event'] = False
        script = generate_script(ann_question, ann_start_time, ann_end_time, video_narrations)

        # ###############################################################
        # verify that script is not too long ############################
        # ###############################################################
        full_script = script.format(json.dumps(example, indent=4))
        if count_total_tokens(full_script) >= (CODEGEN_MODEL_CONTEXT_LENGTH[CODEGEN_MODEL] - codegen_max_tokens):    # budget for the generated query
            # print(f"Skipping annotation due to long script: video {video_uid}, clip {clip_uid}, annotator {annotation_uid}, annotation {ann_idx}")
            is_long_script[str(vid_idx)][str(clip_idx)][str(annotator_idx)][str(ann_idx)] = True
            continue

        # ###############################################################
        # run llm on script to generate query ###########################
        # ###############################################################
        # ann['question'] in this_vids_anns
        # print(f"Generating query for event: {ann_question} ({convert_seconds(ann_start_time)} - {convert_seconds(ann_end_time)})")
        # print(ann['template'])
        llm_out = llm_generate_query(script, example, example_disambiguated, prev_duplicate_label=False)

        # llm_out = {'event_has_occurred_before': True,
        # 'event_readable': '#c uses the phone.',
        # 'query': 'Next time I use my phone after arranging documents on my work table, please remind me to check new emails.',
        # 'ans': 'Remember to check new emails.'}

        if llm_out is None:
            print(f"Failed to generate query for event: {ann_question} ({convert_seconds(ann_start_time)} - {convert_seconds(ann_end_time)})")
            continue    # skip this annotation if we failed to generate a query, will retry next time the script is run
        else:
            # save query
            vid_nlqs_copy['clips'][clip_idx]['annotations'][annotator_idx]['language_queries'][ann_idx]['query'] = llm_out
            # store the annotation so we don't generate queries for its duplicates
            this_vids_anns.append(vid_nlqs_copy['clips'][clip_idx]['annotations'][annotator_idx]['language_queries'][ann_idx])

            total_anns += 1

    #     break
    # break

    # ###############################################################
    # store outputs after finishing all clips in the video ##########
    # ###############################################################
    # store generated queries after finishing all clips in the video
    new_nlqs['videos'].append(vid_nlqs_copy)
    with open(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq.json'), 'w') as f:
        json.dump(new_nlqs, f, indent=4)
    # store duplicates and long scripts (remember to update the files if they already exist)
    if os.path.exists(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq_duplicates.json')):
        # load existing duplicates and update the new ones
        with open(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq_duplicates.json'), 'r') as f:
            existing_duplicates = json.load(f)
            is_duplicate.update(existing_duplicates)
    with open(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq_duplicates.json'), 'w') as f:
        json.dump(is_duplicate, f, indent=4)
    if os.path.exists(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq_long_scripts.json')):
        # load existing long scripts and update the new ones
        with open(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq_long_scripts.json'), 'r') as f:
            existing_long_scripts = json.load(f)
            is_long_script.update(existing_long_scripts)
    with open(os.path.join('generated_queries', CODEGEN_MODEL, f'{EGO4D_SPLIT}_v3.4_nlq_long_scripts.json'), 'w') as f:
        json.dump(is_long_script, f, indent=4)

    # if vid_idx >= 10:    # n - 1 videos have been processed
    #     break

# # %%
# if __name__ == "__main__":
#     main()
print(f"Generated queries for {total_anns} annotations.")
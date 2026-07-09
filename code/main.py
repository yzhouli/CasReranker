import json
import os
import pickle
import time
from tqdm import tqdm

from DiffAgent.Decision_Agent import DecisionAgent
from DiffAgent.DynamicInterest_Agent import DynamicInterestAgent
from DiffAgent.SourcePerception_Agent import SourcePerceptionAgent
from DiffAgent.TopologyAttraction_Agent import TopologyAttractionAgent

MODEL_NAME = "DiffAgent"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


def save_json(path, content):
    with open(path, "w", encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False))

def main(save_path, dataset_path):
    save_dict = dict()
    if os.path.exists(save_path):
        temp_dict = load_json(save_path)
        for key, value in temp_dict.items():
            save_dict[str(key)] = value

    print(f"Loaded {len(save_dict)} existing records from {save_path}")

    test_db = load_pkl(f'{dataset_path}/test_aligned.pkl')

    print("\nInitializing DiffAgent System...")
    semantic_expert = SourcePerceptionAgent(news_pkl_path=f'{dataset_path}/news_all.pkl', mm_dir_path=f'{dataset_path}/mm/mm')
    profile_expert = DynamicInterestAgent(users_pkl_path=f'{dataset_path}/users_all.pkl')
    topology_expert = TopologyAttractionAgent(users_pkl_path=f'{dataset_path}/users_all.pkl')

    coordinator = DecisionAgent(
        semantic_expert=semantic_expert,
        profile_expert=profile_expert,
        topology_expert=topology_expert,
        max_steps=5
    )
    print("System Initialization Complete.\n")

    pdbr = tqdm(total=len(test_db))
    for index, item in enumerate(test_db[:501]):
        pdbr.desc = 'DiffAgent Eval'
        pdbr.update(1)
        if str(index) in save_dict.keys():
            continue

        news_id = item["news_id"]
        next_uid = item["next_user"]

        candidate_users = item["neg_users"][:99]
        candidate_users.append(next_uid)

        history_uids = item["history_users"]

        start_time = time.perf_counter()
        try:
            final_decision, memory, tokens = coordinator.predict_diffusion(
                topic_id=news_id,
                candidate_uids=candidate_users,
                cascade_uids=history_uids
            )
        except Exception as e:
            print(f"[ERROR] sample {index} failed: {type(e).__name__}: {e}")
            continue
        content_str = json.dumps(final_decision, ensure_ascii=False)
        time_span = time.perf_counter() - start_time

        save_dict[str(index)] = {
            'time_span': time_span,
            'content': content_str,
            'memory': memory,
            'reasoning': "Multi-Agent ReAct Loop Executed",
            'tokens': tokens
        }

        save_json(save_path, save_dict)


if __name__ == '__main__':
    save_path = f'saves/{MODEL_NAME}_MAS1(9B_CoT_100).json'
    dataset_path = f'../Casbench'
    main(save_path, dataset_path)
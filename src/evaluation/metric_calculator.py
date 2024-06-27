import os
import json
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser

from src.data import HFDepthQALoader

def compute_score(score_value):
    if isinstance(score_value, list):
        return sum(score_value) / len(score_value)
    elif isinstance(score_value, int):
        return int(score_value)
    else:
        return 0.

def depth_score(results):
    scores = {
        "depth_1": [],
        "depth_2": [],
        "depth_3": []
    }

    for inst in results.values():
        score = inst["score"]
        scores[f'depth_{inst["depth"]}'].append(
                        compute_score(score))
    
    output = {}
    for depth, score in scores.items():
        if len(score) == 0:
            output[f"Average Accuracy - Depth {depth}"] = 0
        else:
            output[f"Average Accuracy - Depth {depth}"] = np.mean(score)

    return output

def forward_disc(nodes, node_results): 
    depths = [(2,3), (1,2)]
    output = {}

    overall_gaps = 0
    overall_cnt = 0
    for (depth_a, depth_b) in depths:
        gaps = []
        cnt = 0
        aggregate = defaultdict(list)
        score_dict = defaultdict(float)
        for nodeid, data in node_results.items():
            if data["depth"] == depth_a:
                depth_a_score = compute_score(data["score"])
                successors = nodes[nodeid]["direct_successors"]
                if len(successors) > 0:
                    for depth_b_id in successors:
                        depth_b_d = node_results[depth_b_id]
                        depth_b_score = compute_score(depth_b_d["score"])
                        score_dict[depth_b_id] = depth_b_score
                        aggregate[depth_b_id].append(depth_a_score)
                        cnt +=1 

        gaps = []
        cnt = 0 
        for bid in score_dict.keys():
            cnt += (np.average(aggregate[bid]) >= 4)
            gap = (np.average(aggregate[bid]) - score_dict[bid])/4
            gaps.append(max(0, gap) * (np.average(aggregate[bid]) >= 4))
        overall_gaps += sum(gaps)
        overall_cnt += cnt
        output[f"Forward Discrepancy - Depth {depth_a} <=> Depth {depth_b}"] = sum(gaps) / cnt

    output["Forward Discrepancy - Overall"] = overall_gaps / overall_cnt

    return output

def backward_disc(nodes, node_results):
    depths = [(2,3), (1,2)]
    output = {}

    overall_gaps = 0
    overall_cnt = 0
    for (depth_a, depth_b) in depths:
        gaps = []
        cnt = 0
        aggregate = defaultdict(list)
        score_dict = defaultdict(float)
        id_map = defaultdict(list)
        for nodeid, data in node_results.items():
            if data["depth"] == depth_b:
                depth_b_score = compute_score(data["score"])
                predecessors = nodes[nodeid]["direct_predecessors"]
                if len(predecessors) > 0:
                    for depth_a_id in predecessors:
                        depth_a_d = node_results[depth_a_id]
                        depth_a_score = compute_score(depth_a_d["score"])
                        score_dict[depth_a_id] = depth_a_score
                        aggregate[depth_a_id].append(depth_b_score)
                        id_map[depth_a_id].append(id)
                        cnt +=1
        
        gaps = []
        cnt = 0 

        for aid in score_dict.keys():
            cnt += (aggregate[aid][0] >= 4)
            gap = (aggregate[aid][0] - score_dict[aid]) / 4
            gaps.append(max(0, gap) * (aggregate[aid][0] >= 4))
        overall_gaps += sum(gaps)
        overall_cnt += cnt
        output[f"Backward Discrepancy - Depth {depth_a} <=> Depth {depth_b}"] = sum(gaps) / cnt

    output["Backward Discrepancy - Overall"] = overall_gaps / overall_cnt   
    return output        



def main(args):

    # Load src data
    dataloader = HFDepthQALoader()
    questions, nodes, node_to_q = dataloader.load_data()

    q_to_node = defaultdict(list)
    for nodeid, qid in node_to_q.items():
        q_to_node[qid].append(nodeid)

    # Load evaluation file
    with open(args.input, "r") as fr:
        results = json.load(fr)

    metrics = {}
    depth_metric = depth_score(results)
    metrics.update(depth_metric)

    node_results = {}
    if list(results.keys())[0] in q_to_node:
        # Convert Q to Node
        for qid, nids in q_to_node.items():
            for nodeid in nids:
                node_results[nodeid] = results[qid]
    else:
        node_results = results

    fwd_metric = forward_disc(nodes, node_results)
    metrics.update(fwd_metric)
    bwd_metric = backward_disc(nodes, node_results)
    metrics.update(bwd_metric)
    # print(metrics)

    with open(args.output_file, "w") as fw:
        json.dump(metrics, fw)


if __name__ == "__main__":
    parser = ArgumentParser()
    # I/O arguments
    parser.add_argument(
        "--src_input",
        type=str,
        default="kaist-ai/DepthQA",
        help="Dataset name in Hugging Face (for zero-shot) or local zero-shot JSON output file (for prompt-*)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file name to calculate metric."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file name. Unless the parent directory is specified, will be saved under outputs/inference by default.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if output file exists.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()
    assert os.path.exists(args.input), "Cannot find input file."
    assert args.output_file.endswith(".json"), "Output file must be a JSON file."
    main(args)

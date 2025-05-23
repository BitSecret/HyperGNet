import random
from formalgeo.tools import load_json, safe_save_json
from gps.utils import load_pickle, get_config
from gps.model import make_model
from gps.data import GeoDataset, nodes_collate_fn, edges_collate_fn, graph_collate_fn, nodes_words
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
import Levenshtein
from tqdm import tqdm
import argparse

config = get_config()
random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])
torch.cuda.manual_seed_all(config["random_seed"])


def get_pretrain_mark(model_type, use_structural_encoding):
    if model_type == "nodes":
        return "nodes"
    else:
        return f"{model_type}_{str(use_structural_encoding)[0]}"


def get_train_mark(use_pretrain, use_structural_encoding, use_hypertree):
    return "".join([str(use_pretrain)[0], str(use_structural_encoding)[0], str(use_hypertree)[0]])


def pretrain(use_structural_encoding, model_type, device):
    """Pretrain node or edges model."""
    mark = get_pretrain_mark(model_type, use_structural_encoding)
    log_path = f"../../data/outputs/log_pretrain_{mark}.json"
    log = {
        "next_epoch": 1,
        "train": {},  # epoch: {"avg_loss": 1, "timing": 1}
        "eval": {},  # epoch: {"acc": 1, "timing": 1}
        "best_epoch": 0,
        "best_levenshtein": 0
    }
    bst_model_path = f"../../data/checkpoints/pretrain_model_bst_{mark}.pth"
    bk_model_path = f"../../data/checkpoints/pretrain_model_bk_{mark}.pth"
    bk_optimizer_path = f"../../data/checkpoints/pretrain_optimizer_bk_{mark}.pth"
    bst_model_results_path = f"../../data/outputs/results_pretrain_model_bst_{mark}.txt"
    training_data_path = f"../../data/training_data/{model_type}_pretrain_data.pkl"

    model = make_model(use_structural_encoding=use_structural_encoding, use_hypertree=True).to(device)
    train_sets, val_sets = load_pickle(training_data_path)
    if model_type == "nodes":
        model = model.nodes_emb
        data_loader_train = DataLoader(
            dataset=GeoDataset(train_sets), collate_fn=nodes_collate_fn,
            batch_size=config["encoder"]["training"]["batch_size"], shuffle=True)
        data_loader_val = DataLoader(
            dataset=GeoDataset(val_sets), collate_fn=nodes_collate_fn,
            batch_size=config["encoder"]["training"]["batch_size"] * 2, shuffle=False)
    else:
        model = model.edges_emb
        data_loader_train = DataLoader(
            dataset=GeoDataset(train_sets), collate_fn=edges_collate_fn,
            batch_size=config["encoder"]["training"]["batch_size"], shuffle=True)
        data_loader_val = DataLoader(
            dataset=GeoDataset(val_sets), collate_fn=edges_collate_fn,
            batch_size=config["encoder"]["training"]["batch_size"] * 2, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["encoder"]["training"]["lr"])
    loss_func = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    if os.path.exists(log_path):
        log = load_json(log_path)
        model.load_state_dict(torch.load(bk_model_path, map_location=torch.device(device), weights_only=True))
        optimizer.load_state_dict(torch.load(bk_optimizer_path, map_location=torch.device(device), weights_only=True))

    epochs = config["encoder"]["training"]["epochs"]
    for epoch in range(log["next_epoch"], epochs + 1):
        model.train()
        loss_list = []
        timing = time.time()

        time.sleep(0.5)  # 防止进度条与日志输出冲突
        loop = tqdm(data_loader_train, leave=False)  # training loop
        for batch_data in loop:
            inputs, outputs = batch_data[0].to(device), batch_data[1].to(device)
            inputs_structure = None
            if model_type == "edges":
                inputs_structure = batch_data[2].to(device)

            predictions = model(mode='train', x=inputs, x_structure=inputs_structure)  # outputs
            loss = loss_func(predictions.transpose(1, 2), outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            loop.set_description(f"Pretraining <{mark}> (epoch [{epoch}/{epochs}])")
            loop.set_postfix(loss=loss.item())
        loop.close()

        avg_loss = round(sum(loss_list) / len(loss_list), 4)
        timing = round(time.time() - timing, 4)
        print(f"Pretrain <{mark}> (Epoch [{epoch}/{epochs}]): avg_loss={avg_loss}, timing={timing}.")
        log["train"][str(epoch)] = {"avg_loss": avg_loss, "timing": timing}

        model.eval()
        timing = time.time()
        levenshtein_sums = 0  # edition distance
        num_counts = 0  # sequence length
        results = []  # decoding results
        time.sleep(0.5)  # 防止进度条与日志输出冲突
        loop = tqdm(data_loader_val, leave=False)  # evaluating loop
        with torch.no_grad():
            for batch_data in loop:
                inputs, outputs = batch_data[0].to(device), batch_data[1].to(device)
                inputs_structure = None
                if model_type == "edges":
                    inputs_structure = batch_data[2].to(device)

                seqs_encoding = model(mode='encode', x=inputs, x_structure=inputs_structure)  # get nodes encoding
                predictions = torch.zeros(size=outputs.size(), dtype=torch.long).to(device)
                start_vector = torch.ones(size=(outputs.size(0), 1), dtype=torch.long).to(device)

                for i in range(outputs.size(1)):  # iterative decoding
                    predictions = torch.cat([start_vector, predictions], dim=1)[:, :-1]
                    predictions = model(mode='test', x_structure=inputs_structure,
                                        x_encoding=seqs_encoding, y=predictions)
                    predictions = torch.argmax(predictions, dim=2).long()
                predictions = predictions.cpu()
                outputs = outputs.cpu()

                levenshtein_sum, num_count, result = evaluate_decoding(predictions, outputs, model_type)
                levenshtein_sums += levenshtein_sum
                num_counts += num_count
                results.extend(result)
                loop.set_description(f"Evaluating <{mark}> (epoch [{epoch}/{epochs}])")
                loop.set_postfix(acc=levenshtein_sum / num_count)
            loop.close()

        levenshtein = round(levenshtein_sums / num_counts, 4)
        timing = round(time.time() - timing, 4)
        log["eval"][str(epoch)] = {"levenshtein": levenshtein, "timing": timing}
        print(f"Evaluate <{mark}> (Epoch [{epoch}/{epochs}]): levenshtein={levenshtein}, timing={timing}.")

        if levenshtein > log["best_levenshtein"]:
            log["best_levenshtein"] = levenshtein
            log["best_epoch"] = epoch
            torch.save(model.state_dict(), bst_model_path)
            with open(bst_model_results_path, 'w', encoding='utf-8') as file:
                file.write("\n".join(results))
                file.write(f"\nlevenshtein: {levenshtein}, nums: {num_counts}")

        torch.save(model.state_dict(), bk_model_path)
        torch.save(optimizer.state_dict(), bk_optimizer_path)
        log["next_epoch"] += 1
        safe_save_json(log, log_path)


def train(use_pretrain, use_structural_encoding, use_hypertree, device, beam_size, training):
    """Train theorem predictor."""

    mark = get_pretrain_mark("nodes", use_structural_encoding)
    bst_pretrain_nodes_model_path = f"../../data/checkpoints/pretrain_model_bst_{mark}.pth"
    mark = get_pretrain_mark("edges", use_structural_encoding)
    bst_pretrain_edges_model_path = f"../../data/checkpoints/pretrain_model_bst_{mark}.pth"
    mark = get_train_mark(use_pretrain, use_structural_encoding, use_hypertree)
    log_path = f"../../data/outputs/log_train_{mark}.json"
    log = {
        "next_epoch": 1,
        "train": {},  # epoch: {"avg_loss": 1, "acc": 1, "timing": 1}
        "eval": {},  # epoch: {"avg_loss": 1, "acc": 1, "timing": 1}
        "best_epoch": 0,
        "best_eval_loss": 100000,
        "best_eval_acc": 0
    }
    bst_model_path = f"../../data/checkpoints/train_model_bst_{mark}.pth"
    bk_model_path = f"../../data/checkpoints/train_model_bk_{mark}.pth"
    bk_optimizer_path = f"../../data/checkpoints/train_optimizer_bk_{mark}.pth"
    bst_model_val_results_path = f"../../data/outputs/results_train_bst_model_val_{mark}.txt"
    bst_model_test_results_path = f"../../data/outputs/results_train_bst_model_test_{mark}_bs{beam_size}.txt"
    training_data_path = "../../data/training_data/train_data.pkl"

    model = make_model(use_structural_encoding, use_hypertree).to(device)
    train_sets, val_sets, test_sets = load_pickle(training_data_path)
    data_loader_train = DataLoader(
        dataset=GeoDataset(train_sets), collate_fn=graph_collate_fn,
        batch_size=config["predictor"]["training"]["batch_size"], shuffle=True)
    data_loader_val = DataLoader(
        dataset=GeoDataset(val_sets), collate_fn=graph_collate_fn,
        batch_size=config["predictor"]["training"]["batch_size"] * 2, shuffle=False)
    data_loader_test = DataLoader(
        dataset=GeoDataset(test_sets), collate_fn=graph_collate_fn,
        batch_size=config["predictor"]["training"]["batch_size"] * 2, shuffle=False)
    loss_func = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    if training:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["predictor"]["training"]["lr"])  # Adam optimizer
        if os.path.exists(log_path):
            log = load_json(log_path)
            model.load_state_dict(torch.load(bk_model_path, map_location=torch.device(device), weights_only=True))
            optimizer.load_state_dict(
                torch.load(bk_optimizer_path, map_location=torch.device(device), weights_only=True))
        elif use_pretrain:
            model.nodes_emb.load_state_dict(
                torch.load(bst_pretrain_nodes_model_path, map_location=torch.device(device), weights_only=True))
            model.edges_emb.load_state_dict(
                torch.load(bst_pretrain_edges_model_path, map_location=torch.device(device), weights_only=True))

        epochs = config["predictor"]["training"]["epochs"]
        for epoch in range(log["next_epoch"], epochs + 1):

            avg_loss, acc, timing, results = run_one_epoch(  # training
                model, data_loader_train, loss_func, optimizer,
                loop_description=f"Training <{mark}> (epoch [{epoch}/{epochs}])")
            print(f"Train <{mark}> (Epoch [{epoch}/{epochs}]): avg_loss={avg_loss}, acc={acc}, timing={timing}.")
            log["train"][str(epoch)] = {"avg_loss": avg_loss, "acc": acc, "timing": timing}

            with torch.no_grad():  # evaluating
                avg_loss, acc, timing, results = run_one_epoch(
                    model, data_loader_val, loss_func, None,
                    loop_description=f"Evaluating <{mark}> (epoch [{epoch}/{epochs}])")
            log["eval"][str(epoch)] = {"avg_loss": avg_loss, "acc": acc, "timing": timing}
            print(f"Evaluate <{mark}> (Epoch [{epoch}/{epochs}]): avg_loss={avg_loss}, acc={acc}, timing={timing}.")

            if avg_loss < log["best_eval_loss"]:
                log["best_epoch"] = epoch
                log["best_eval_loss"] = avg_loss
                log["best_eval_acc"] = acc
                torch.save(model.state_dict(), bst_model_path)
                with open(bst_model_val_results_path, 'w', encoding='utf-8') as file:
                    file.write("\n".join(results))
                    file.write(f"\navg_loss: {avg_loss}, acc: {acc}")
            torch.save(model.state_dict(), bk_model_path)
            torch.save(optimizer.state_dict(), bk_optimizer_path)
            log["next_epoch"] += 1
            safe_save_json(log, log_path)

    model.load_state_dict(torch.load(bst_model_path, map_location=torch.device(device), weights_only=True))  # testing
    with torch.no_grad():
        avg_loss, acc, timing, results = run_one_epoch(
            model, data_loader_test, loss_func, None,
            loop_description=f"Testing <{mark}>", beam_size=beam_size)
    print(f"Test <{mark}>: avg_loss={avg_loss}, acc={acc}, beam_size: {beam_size}, timing={timing}.")
    with open(bst_model_test_results_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(results))
        file.write(f"\navg_loss: {avg_loss}, acc: {acc}, beam_size: {beam_size}")


def run_one_epoch(model, data_loader, loss_func, optimizer, loop_description, beam_size=5):
    """evaluate theorem predictor."""
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    device = next(model.parameters()).device
    timing = time.time()
    loss_list = []  # for acv_loss calculation
    acc_count = 0  # for acc calculation
    num_count = 0  # for acc calculation
    results = []  # decoding results

    time.sleep(0.5)
    loop = tqdm(data_loader, leave=False)  # evaluating loop
    loop.set_description(loop_description)
    for nodes, edges, structures, goals, theorems in loop:
        nodes, edges, structures = nodes.to(device), edges.to(device), structures.to(device)
        goals, theorems = goals.to(device), theorems.to(device)
        predictions = model(nodes=nodes, edges=edges, structures=structures, goals=goals)

        loss = loss_func(predictions.float(), theorems.float())
        loss_list.append(loss.item())  # loss
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        num_count += len(theorems)
        for i in range(len(theorems)):
            theorem_str = [str(idx)
                           for idx, t in enumerate(theorems[i]) if t != 0]
            prediction_str = [str(idx)
                              for idx, _ in sorted(enumerate(predictions[i]), key=lambda x: x[1], reverse=True)]
            if len(set(prediction_str[:beam_size]) & set(theorem_str)) > 0:
                acc_count += 1
            results.append(f"GT: [{', '.join(theorem_str)}]\tPD: [{', '.join(prediction_str[:beam_size])}]")

        loop.set_postfix(loss=loss.item())

    loop.close()

    avg_loss = round(sum(loss_list) / len(loss_list), 4)
    acc = round(acc_count / num_count, 4)
    timing = round(time.time() - timing, 4)

    return avg_loss, acc, timing, results


def parse_seqs(seqs, model_type):
    """Called by func: <evaluate_decoding>."""
    seqs_unicode = []  # for compute Levenshtein ratio
    seqs_sematic = []  # for human check
    for i in range(len(seqs)):
        if seqs[i] == 2:  # <end>
            break
        seqs_unicode.append(chr(seqs[i].item()))
        if model_type == "nodes":
            seqs_sematic.append(nodes_words[seqs[i].item()])
        else:
            seqs_sematic.append(str(seqs[i].item()))

    return seqs_unicode, seqs_sematic


def evaluate_decoding(predictions, output_seqs, model_type):
    """
    Evaluate decoding results. Called by func: <pretrain>.
    :param predictions: Model predictions.
    :param output_seqs: Ground truth sequences.
    :param model_type: model type, 'nodes' or 'edges'.
    """
    levenshtein_sum = 0
    num_count = 0
    result = []
    for i in range(predictions.size(0)):
        seqs_unicode, seqs_sematic = parse_seqs(predictions[i], model_type)
        seqs_unicode_gt, seqs_sematic_gt = parse_seqs(output_seqs[i], model_type)

        len_seqs = len(seqs_unicode_gt)
        levenshtein_sum += Levenshtein.ratio("".join(seqs_unicode), "".join(seqs_unicode_gt)) * len_seqs
        num_count += len_seqs

        join_str = ","
        if model_type == "nodes":
            seqs_sematic.insert(1, "(")
            seqs_sematic.append(")")
            seqs_sematic_gt.insert(1, "(")
            seqs_sematic_gt.append(")")
            join_str = ""

        result.append(f"GT: {join_str.join(seqs_sematic_gt)}\tPD: {join_str.join(seqs_sematic)}")

    return levenshtein_sum, num_count, result


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")

    parser.add_argument("--func", type=str, required=True,
                        choices=["pretrain_nodes", "pretrain_edges", "train", "test"],
                        help="Function that you want to run.")

    parser.add_argument("--device", type=str, required=False, default="cuda:0", choices=["cpu", "cuda:0", "cuda:1"],
                        help="Device for pretraining.")

    parser.add_argument("--use_pretrain", type=lambda x: x == "True", default=True,
                        help="Use pretrain.")
    parser.add_argument("--use_structural_encoding", type=lambda x: x == "True", default=True,
                        help="Use structural encoding.")
    parser.add_argument("--use_hypertree", type=lambda x: x == "True", default=True,
                        help="Use hypertree.")

    parser.add_argument("--beam_size", type=int, required=False, default=5,
                        help="Beam size when calculate acc, only used in testing.")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    Run in background:
    nohup <python test.py> >/dev/null 2>&1 &
    nohup <python test.py> --device cuda:1 >/dev/null 2>&1 &
    
    Pretrain:
    python train.py --func pretrain_nodes
    python train.py --func pretrain_edges
    python train.py --func pretrain_edges --use_structural_encoding False
    
    Train:
    python train.py --func train
    python train.py --func train --use_pretrain False
    python train.py --func train --use_structural_encoding False
    python train.py --func train --use_hypertree False
    
    Test:
    python train.py --func test --beam_size 3
    python train.py --func test --use_pretrain False --beam_size 3
    python train.py --func test --use_structural_encoding False --beam_size 3
    python train.py --func test --use_hypertree False --beam_size 3
    
    python train.py --func test --beam_size 1
    python train.py --func test --use_pretrain False --beam_size 1
    python train.py --func test --use_structural_encoding False --beam_size 1
    python train.py --func test --use_hypertree False --beam_size 1
    """

    args = get_args()
    if args.func == "pretrain_nodes":
        pretrain(use_structural_encoding=args.use_structural_encoding,
                 model_type="nodes", device=args.device)
    elif args.func == "pretrain_edges":
        pretrain(use_structural_encoding=args.use_structural_encoding,
                 model_type="edges", device=args.device)
    elif args.func == "train":
        train(use_pretrain=args.use_pretrain,
              use_structural_encoding=args.use_structural_encoding, use_hypertree=args.use_hypertree,
              device=args.device, beam_size=args.beam_size, training=True)
    elif args.func == "test":
        train(use_pretrain=args.use_pretrain,
              use_structural_encoding=args.use_structural_encoding, use_hypertree=args.use_hypertree,
              device=args.device, beam_size=args.beam_size, training=False)

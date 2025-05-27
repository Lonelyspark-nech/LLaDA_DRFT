# coding:gbk
import os
import torch
import numpy as np
import pickle
import datasets
from llm2vec import LLM2Vec
from beir import util
from beir.datasets.data_loader import GenericDataLoader as BeirDataLoader
from contextlib import nullcontext
from torch.cuda.amp import autocast
from typing import Dict, List
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import freeze_support

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '16286'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ���������ָ��
def append_instruction(instruction, sentences):
    return [[instruction, s, 0] for s in sentences]

# ����������queries
def encode_queries(queries: List[str], batch_size: int, model, **kwargs):
    new_sentences = append_instruction(instruction, queries)
    kwargs["show_progress_bar"] = False
    return model.encode(new_sentences, batch_size=batch_size, **kwargs)

# ����������corpus
def encode_corpus(corpus: List[Dict[str, str]], batch_size: int, model, **kwargs):
    sentences = [
        (doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip()
        for doc in corpus
    ]
    new_sentences = append_instruction("", sentences)
    # ֱ��ʹ��ģ�ͱ��룬����ʹ���ڲ��Ķ����
    with torch.no_grad():
        return model.encode(new_sentences, batch_size=batch_size, **kwargs)

def main(rank, world_size):
    setup(rank, world_size)
    
    batch_size = 256  # ÿ��GPU�����δ�С
    dataset = "msmarco"
    instruction = "Given a claim, find documents that refute the claim: "

    if rank == 0:
        print("Loading dataset...")
    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    )
    download_path = os.path.join(datasets.config.HF_DATASETS_CACHE, "BeIR")
    data_path = util.download_and_unzip(url, download_path)
    corpus, queries, relevant_docs = BeirDataLoader(data_folder=data_path).load(
        split="test"
    )

    if rank == 0:
        print("Loading model...")
    # ����ģ��
    model = LLM2Vec.from_pretrained(
        "/data1/jiyifan/plm_dir/Sheared-LLaMA-1.3B",
        peft_model_name_or_path="/data1/jiyifan/llm2vec-main/output/mntp-supervised/Sheared-LLaMA-1.3B/E5_train_m-Sheared-LLaMA-1.3B_p-eos_token_b-256_l-512_bidirectional-False_e-1_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
        device_map=f"cuda:{rank}",
        torch_dtype=torch.bfloat16,
    )

    # ʹ��DDP��װģ��
    #model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.eval()

    # �����Ͽⰴ��������
    corpus_ids = sorted(
        corpus, 
        key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), 
        reverse=True
    )
    corpus = [corpus[cid] for cid in corpus_ids]

    # Ϊÿ��rank��������
    per_rank_size = len(corpus) // world_size
    start_idx = rank * per_rank_size
    end_idx = start_idx + per_rank_size if rank != world_size - 1 else len(corpus)
    rank_corpus = corpus[start_idx:end_idx]

    # ����Ŀ¼·��
    dir_path = "encoded_corpus"
    corpus_ids_mapping = {idx: cid for idx, cid in enumerate(corpus_ids)}
    
    if rank == 0:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, "corpus_ids_mapping.pkl"), "wb") as f:
            pickle.dump(corpus_ids_mapping, f)

    # �������벢������
    if rank == 0:
        print(f"Encoding Corpus on rank {rank}...")
    idx, lookup_indices, encoded = start_idx, [], []

    for i in tqdm(range(0, len(rank_corpus), batch_size), 
                 desc=f"Processing batches on rank {rank}", 
                 ncols=100):
        batch = rank_corpus[i:i + batch_size]
        batch_ids = list(range(start_idx + i, start_idx + i + len(batch)))
        lookup_indices.extend(batch_ids)

        with autocast() if torch.cuda.is_available() else nullcontext():
            model_output = encode_corpus(batch, batch_size=batch_size, 
                                      model=model, show_progress_bar=False)
            encoded.append(model_output.cpu().detach().to(torch.float).numpy())
##
        # ���ڱ���
        if len(lookup_indices) >= 100:
            encoded_data = np.concatenate(encoded)
            save_path = os.path.join(
                dir_path,
                f"embeddings.corpus.rank.{rank}.{idx}-{idx + len(lookup_indices)}.pkl",
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump((encoded_data, lookup_indices), f, protocol=4)

            encoded.clear()
            lookup_indices.clear()
            idx += len(batch_ids)

    # ����ʣ�������
    if len(lookup_indices) > 0:
        encoded_data = np.concatenate(encoded)
        save_path = os.path.join(
            dir_path,
            f"embeddings.corpus.rank.{rank}.{idx}-{idx + len(lookup_indices)}.pkl",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump((encoded_data, lookup_indices), f, protocol=4)

    # �ȴ����н������
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    freeze_support()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
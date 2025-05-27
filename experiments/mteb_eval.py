# -*- coding: utf-8 -*-
import json
import logging
from datasets import load_dataset
import mteb
import torch
from mteb import MTEB, ModelMeta
from mteb.models.llm2vec_models import _loader, LLM2VecWrapper

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    tasks = mteb.get_tasks(tasks=["ArguAna","NFCorpus","SCIDOCS","SciFact","FiQA2018","QuoraRetrieval","TRECCOVID","Touche2020","NQ","DBPedia","HotpotQA","ClimateFEVER","FEVER"], languages=["eng"])
    #tasks = mteb.get_tasks(tasks=["ArguAna","NFCorpus","SCIDOCS","SciFact","FiQA2018","QuoraRetrieval","TRECCOVID","Touche2020"])
    #tasks = mteb.get_tasks(tasks=["ArguAna","NFCorpus","SCIDOCS","SciFact"],languages=["eng"])
    # evaluation = MTEB(tasks=["ArguAna","FiQA2018","NFCorpus","QuoraRetrieval","SCIDOCS","SciFact","TRECCOVID","Touche2020","DBPedia","HotpotQA","NQ"])
    # tasks = mteb.get_tasks(tasks=["NQ"], languages=["eng"])
    # tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])
    evaluation = MTEB(tasks=tasks)
    # evaluation = MTEB(tasks=[
    #     "SprintDuplicateQuestions",
    #     "TwitterSemEval2015",
    #     "TwitterURLCorpus",
    #     "SummEval",
    #     "BIOSSES",
    #     "SICK-R",
    #     "STS12",
    #     "STS13",
    #     "STS14",
    #     "STS15",
    #     "STS16",
    #     "STSBenchmark",
    #     "STS17",
    #     "STS22",
    #     "AmazonPolarityClassification",
    #     "Banking77Classification",
    #     "EmotionClassification",
    #     "ImdbClassification",
    #     "ToxicConversationsClassification",
    #     "TweetSentimentExtractionClassification",
    #     "AmazonCounterfactualClassification",
    #     "AmazonReviewsClassification",
    #     "MassiveIntentClassification",
    #     "MassiveScenarioClassification",
    #     "MTOPDomainClassification",
    #     "MTOPIntentClassification",
    #     # "AskUbuntuDupQuestions",
    #     # "MindSmallReranking",
    #     # "SciDocsRR",
    #     # "StackOverflowDupQuestions",
    #     "ArxivClusteringP2P",
    #     "ArxivClusteringS2S",
    #     "BiorxivClusteringP2P",
    #     "BiorxivClusteringS2S",
    #     "MedrxivClusteringP2P",
    #     "MedrxivClusteringS2S",
    #     "RedditClustering",
    #     "RedditClusteringP2P",
    #     "StackExchangeClustering",
    #     "StackExchangeClusteringP2P",
    #     "TwentyNewsgroupsClustering"
    # ])

    # evaluation = MTEB(tasks=["CQADupstackAndroidRetrieval","CQADupstackEnglishRetrieval","CQADupstackGamingRetrieval","CQADupstackGisRetrieval","CQADupstackMathematicaRetrieval","CQADupstackPhysicsRetrieval","CQADupstackProgrammersRetrieval","CQADupstackStatsRetrieval","CQADupstackTexRetrieval","CQADupstackUnixRetrieval","CQADupstackWebmastersRetrieval","CQADupstackWordpressRetrieval"])
    # evaluation = MTEB(tasks=["NFCorpus","ArguAna","SCIDOCS","SciFact"])
    model_kwargs = {}
    with open("/data1/jiyifan/llm2vec-main/test_configs/mteb/task_to_instructions.json", "r") as f:
        task_to_instructions = json.load(f)
    model_kwargs["task_to_instructions"] = task_to_instructions
    model = mteb.get_model("SmolLM2-135M-Instruct-loss1-tech", **model_kwargs)
    #model = mteb.get_model("Mcpm-2B-loss1-tech", **model_kwargs)

    evaluation.run(model, output_folder="/data1/jiyifan/llm2vec-main/mteb_result/SmolLM2-135M-Instruct-loss1-tech",eval_splits=["test"])






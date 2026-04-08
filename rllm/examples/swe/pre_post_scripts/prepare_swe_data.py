from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry

import jsonlines
from loguru import logger


SWE_DATASETS = [
    # "R2E-Gym/R2E-Gym-Subset",
    # "R2E-Gym/R2E-Gym-Lite",
    # "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/SWE-Bench-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    # "r2e-edits/SweSmith-RL-Dataset",
    # "/mnt/82_store/tmp/SWE-bench-agent/storage/ByteDance-Seed/Multi-SWE-bench-flash/multi_swe_bench_flash.jsonl",
    # "/mnt/82_store/tmp/SWE-bench-agent/storage/ByteDance-Seed/Multi-SWE-bench-flash/multi_swe_bench_flash_subset_32.jsonl",
    # "/mnt/82_store/tmp/SWE-bench-agent/storage/ByteDance-Seed/Multi-SWE-bench-flash/multi_swe_bench_flash_subset_100.jsonl",
    # "/mnt/82_store/tmp/SWE-bench-agent/storage/R2E-Gym/SWE-Bench-Verified_subset_100.parquet"

]


MSWE_LANGUAGE_REPO_MAP = {
    "c": [
        "facebook/zstd",
        "jqlang/jq",
        "ponylang/ponyc",
    ],
    "cpp": [
        "catchorg/Catch2",
        "fmtlib/fmt",
        "nlohmann/json",
        "simdjson/simdjson",
        "yhirose/cpp-httplib",
    ],
    "go": [
        "cli/cli",
        "grpc/grpc-go",
        "zeromicro/go-zero",
    ],
    "java": [
        "alibaba/fastjson2",
        "apache/dubbo",
        "elastic/logstash",
        "fasterxml/jackson-core",
        "fasterxml/jackson-databind",
        "fasterxml/jackson-dataformat-xml",
        "google/gson",
        "googlecontainertools/jib",
        "mockito/mockito",
    ],
    "javascript": [
        "Kong/insomnia",
        "anuraghazra/github-readme-stats",
        "axios/axios",
        "expressjs/express",
        "iamkun/dayjs",
        "sveltejs/svelte",
    ],
    "rust": [
        "BurntSushi/ripgrep",
        "clap-rs/clap",
        "nushell/nushell",
        "rayon-rs/rayon",
        "serde-rs/serde",
        "sharkdp/bat",
        "sharkdp/fd",
        "tokio-rs/bytes",
        "tokio-rs/tokio",
        "tokio-rs/tracing",
    ],
    "typescript": [
        "darkreader/darkreader",
        "mui/material-ui",
        "vuejs/core",
    ],
}




def prepare_swe_data():
    """
    Prepare and register SWE datasets for training and testing.

    Returns:
        tuple: (train_datasets, test_datasets) - lists of registered datasets
    """

    def make_process_fn():
        def process_fn(row):
            row_dict = dict(row)
            # problem_statement = row_dict.get("problem_statement", "")
            return row_dict

        return process_fn

    process_fn = make_process_fn()
    train_datasets = []
    test_datasets = []

    for dataset_name in SWE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        dataset_key = dataset_name.split("/")[-1].split(".")[0].replace("-", "_")


        if dataset_name.endswith(".jsonl") and "multi" in dataset_name:
            with jsonlines.open(dataset_name, 'r') as reader:
                test_data = []
                for item in reader:
                    org = item.get("org", "")
                    repo = item.get("repo", "")
                    for language_type in MSWE_LANGUAGE_REPO_MAP:
                        if f"{org}/{repo}" in MSWE_LANGUAGE_REPO_MAP[language_type]:
                            language = language_type
                            break
                    else:
                        logger.warning(f"Language not found for {org}/{repo}")
                        language = "python"
                    number = str(item.get("number", "0"))  
                    new_item = {}
                    new_item["repo"] = f"{org}/{repo}"
                    new_item["number"] = number
                    new_item["language"] = language
                    new_item["instance_id"] = item["instance_id"]
                    new_item["problem_statement"] = item["resolved_issues"][0].get("title", "") + "\n" + item["resolved_issues"][0].get("body", "")
                    new_item["FAIL_TO_PASS"] = list(item["f2p_tests"].keys())
                    new_item["PASS_TO_PASS"] = list(item["p2p_tests"].keys())
                    new_item["patch"] = item.get("fix_patch", "")
                    new_item["test_patch"] = item["test_patch"]
                    new_item["base_commit"] = item['base'].get("sha","")
                    new_item["version"] = "0.1" # depends
                    new_item["docker_image"] = f"mswebench/{org.lower()}_m_{repo.lower()}:pr-{number}"
                    # only keep some languages for debugging
                    # if language not in ["javascript"]:
                    #     continue
                    test_data.append(new_item)
            test_split = DatasetRegistry.register_dataset(f"{dataset_key}", test_data, "test")
            test_datasets.append(test_split)
            logger.info(f"Registered test dataset with {len(test_data)} examples")
            continue



        try:
            # Load the dataset dictionary (which contains splits like 'train' or 'test')
            if dataset_name.endswith(".parquet"):
                dataset_splits = load_dataset("parquet", data_files=dataset_name)
            else:
                dataset_splits = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue


        # Process train split if it exists
        if "train" in dataset_splits:
            print(f"Processing 'train' split for {dataset_name}")
            train_data = [process_fn(row) for row in dataset_splits["train"]]
            logger.info(f"Train data: {train_data[0].keys()}")
            # some fields are nested data
            for data in train_data:
                data.pop("modified_entity_summaries")
            # if dataset_name == "R2E-Gym/R2E-Gym-Subset":
            #     dataset_key1 = "R2E_Gym_Subset_0_2000_SFT"
            #     dataset_key2 = "R2E_Gym_Subset_2000_4578_RL"
            #     dataset1 = train_data[:2000]
            #     dataset2 = train_data[2000:]
            #     train_datasets.append(DatasetRegistry.register_dataset(f"{dataset_key1}", dataset1, "train"))
            #     train_datasets.append(DatasetRegistry.register_dataset(f"{dataset_key2}", dataset2, "train"))
            train_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", train_data, "train")
            train_datasets.append(train_dataset)
            print(f"Registered train dataset with {len(train_data)} examples")

        # Process test split if it exists
        if "test" in dataset_splits:
            print(f"Processing 'test' split for {dataset_name}")
            test_data = [process_fn(row) for row in dataset_splits["test"]]
            test_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", test_data, "test")
            test_datasets.append(test_dataset)
            print(f"Registered test dataset with {len(test_data)} examples")

        # If neither train nor test exists, use the first available split as train
        if "train" not in dataset_splits and "test" not in dataset_splits:
            available_splits = list(dataset_splits.keys())
            if available_splits:
                split_name = available_splits[0]
                print(f"Using '{split_name}' split as train data for {dataset_name}")
                train_data = [process_fn(row) for row in dataset_splits[split_name]]
                train_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", train_data, "train")
                train_datasets.append(train_dataset)
                print(f"Registered train dataset with {len(train_data)} examples")

    return train_datasets, test_datasets


if __name__ == "__main__":
    train_datasets, test_datasets = prepare_swe_data()
    print("\nSummary:")
    print(f"Total train datasets: {len(train_datasets)}")
    print(f"Total test datasets: {len(test_datasets)}")

    # if train_datasets:
    #     print("Sample train example from first dataset:")
    #     print(train_datasets[0].get_data()[0])

    # if test_datasets:
    #     print("Sample test example from first dataset:")
    #     print(test_datasets[0].get_data()[0])

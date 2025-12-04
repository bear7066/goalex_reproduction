# GoalEx reproduction

# Setup

```
pip install -r requirements.txt
```
```
export OPENAI_API_KEY=
```

# Usage

1. execution

```
python src/iterative_cluster.py \
  --data_path processed_data/agnews \
  --exp_dir experiments/agnews \
  --proposer_model gpt-3.5-turbo \
  --assigner_name gpt-3.5-turbo \
  --proposer_num_descriptions_to_propose 30 \
  --assigner_for_final_assignment_template templates/t5_multi_assigner_one_output.txt \
  --iterative_max_rounds 1 \
  --verbose
  --with_labels
```

2. evaluation

```
python src/evaluate_only.py \
  --data_path processed_data/xxx \
  --exp_dir experiments/xxx
```

# Dataset
All datasets  are in `processed_data`, containing
- goal: A natural language goal
- texts: dataset texts
- example_descriptions(optional) [Optional] a list of strings, for example, 
  `['has a topic of healthcare', 'has a topic of education']`
- ```labels.json```


# PAS
Here, we show how to use our method, Propose-Assign-Select (PAS), for solving this task.
You can specify an `exp_dir` as an argument to it and the results will be saved in `exp_dir/cluster_result.json`. 

In most cases we use a flan-t5 model as the assigner, so it is reccomended to have a GPU.

## Running on a debug dataset
You will need to have set up the openai api key and organization account in the set up step.
The GPU memory required for the flan-t5-xl assigner is around 20GB.  
To test the code, we provide a debug dataset `processed_data/debug`. You can run the following command:
```
python src/iterative_cluster.py \
  --data_path processed_data/debug \
  --exp_dir experiments/debug \
  --proposer_model gpt-3.5-turbo \
  --assigner_name google/flan-t5-xl \
  --proposer_num_descriptions_to_propose 10 \
  --cluster_num_clusters 6 \
  --verbose
```

## Running on a clustering dataset
Here is also a list of recommended proposers and assigners:
- Proposers: gpt-3.5-turbo, gpt-4
- Assigners: google/flan-t5-xl, gpt-3.5-turbo

We will estimate the query cost if using openai proposer and assigners, before starting the algorithm.

test:

```
python src/iterative_cluster.py \
  --data_path processed_data/agnews \
  --exp_dir experiments/agnews \
  --subsample 1024 \
  --proposer_model gpt-3.5-turbo \
  --assigner_name google/flan-t5-xl \
  --proposer_num_descriptions_to_propose 30 \
  --assigner_for_final_assignment_template templates/t5_multi_assigner_one_output.txt \
  --cluster_num_clusters 4 \
  --verbose
  --with_labels
```

## Saving budget
1. We recommend using a subsample of the dataset to save budget.
2. Experiment with a FlanT5 assigner, and only use a GPT-3.5-turbo assigner if the proposed descriptions are good, but the found clusters are subpar.

## Creating a hierarchy

<p align="center"><img width="80%" src="goalex_hierarchy.png"/></p>
We can generate a taxonomy of model errors for human feedbacks of model-generated summaries, as shown in the figure above.

**IMPORTANT** We turn off asking for approval for the recursive clustering, so please be mindful of the cost when using this feature.

As an example with the datasets we expeirmented in the paper, run
```
python src/recursive_cluster.py
```

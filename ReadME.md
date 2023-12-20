## Description of the project
This project is mainly about fine-tuning LLM with limited computational/labor resources. 

Large language models have been shown to have powerful capabilities, including but not limited to Common Sense Reasoning, Closed-book Question Answering, Reading Comprehension, Mathematical reasoning, Code generation, and other areas of human interaction[https://arxiv.org/abs/2302.13971]. However, it has been found that it is difficult for large language models to understand human commands and give correct answers. This gap between training and real-world use often needs to be filled by fine-tuning the model.[https://arxiv.org/abs/2301.13688]

However, even the fine-tuning of the model is resource intensive, for example, fine-tuning all the parameters of the 7b model requires 70GB of memory.On top of that, the performance of the model depends heavily on the quality of the data, which is often manually labelled with significant labour costs.[https://arxiv.org/abs/2306.04751] Therefore, it is particularly important to improve the performance of the model with limited resources.

In this project, we proposed to fine tune Mistral 7b[https://arxiv.org/abs/2310.06825] by using QLoRA[https://arxiv.org/abs/2305.14314] and LIMA dataset[https://arxiv.org/abs/2305.11206], which contains only 1,000 carefully curated prompts and response. 


## Project milestones and their completion status
• Fine-tune Mistral 7b with QLoRA on a specific instruct dataset and test its performance [completed]
• Furthur investigates the existing benchmark for testing and analyzing the performance of the Instruct model [partilaly completed]
• Fine-tune Mistral MoE with QLoRA [didn’t have time]
• Conduct the experiments on multiple datasets and compare them with other existing instruct model [partilaly completed]

## A description of the repository and code structure
This repository contains the notebooks of the fine-tuning process and the scripts to process the data. 
script: contains the code needed to run the code on the HPC
tools: contains the code needed to analyze the result, pre-process the data 
data: the LIMA dataset used for training

## Example commands to execute the code         
To fine tune the model, only have to run the mistral-finetune notebbok. 

## Results (including charts/tables) and your observations 
Observation:
1 Validation loss maybe not a good metric. More experiments should be conducted to find the best place to stop the training. 
2 By only calculating the loss on the output(Pictures below), the model converges faster. 
3 It’s hard to compare model’s performance, due to the nature of open-ended question, we still need more experiments
4 Our experimental results show that 3 hours of fine-tuning under a single A100 GPU is sufficient to produce a usable instruction model by using QLoRA and LIMA datasets. 
5 However, the choice of hyperparameters and the evaluation of the model are still a problem for us, and we need to conduct more experiments to investigate. In the work of Wang et al[1]. they proposed to evaluate the 5 basic capabilities of the model, as these are the basic capabilities needed for the instruction model, and if there is still time this will be our direction.

More details, see the slides. 








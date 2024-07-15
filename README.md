# Intel Unnati Industrial Training Program 2024
Introduction to GenAI and Simple LLM Inference on CPU and finetuning of LLM Model to create a Custom Chatbot


### Team Details:

- **Team Name**: Tech Wizards 

- **Team Members**:
  - [Joy Mascarenhas](https://github.com/JoyM268) (Team Lead)
  - [AbdulBasith A Mulla](https://github.com/hauntedknight55)
  - [Kodeganti Bhanu Shankar](https://github.com/heyitsELECTRO)

- **Team Mentor**: Prathap Kumar

- **Institute Name**: Shri Dharmasthala Manjunatheshwara College of Engineering and Technology, Dharwad. 


### Project Details:

- **Project Statement**: Introduction to GenAI and Simple LLM Inference on CPU and finetuning of LLM Model to create a Custom Chatbot.

- **Project Description**: This problem statement is designed to introduce beginners to the exciting field of Generative Artificial Intelligence (GenAI) through a series of hands-on exercises Participants will learn the basics of GenAI, perform simple Large Language Model (LLM) inference on a CPU, and explore the process of fine-tuning an LLM model to create a custom Chatbots.

- **Major Challenges**:
    - Pre-trained Language Models can have large file sizes, which may require significant storage space and memory to load and run.  
    - Learn LLM inference on CPU.  
    - Understanding the concept of fine-tuning and its importance in customizing LLMs.  
    - Create a Custom Chatbot with fine-tuned pre-trained Large Language Models (LLMs) using Intel AI Tools. 

<br>

### Creating a Chatbot:
The following code is used to create the chatbot:
```python
from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.transformers import MixedPrecisionConfig
config = PipelineConfig(optimization_config=MixedPrecisionConfig())
chatbot = build_chatbot(config)
```

Once the chatbot is created we can prompt it as shown below:
```python
response = chatbot.predict(query="Tell me about Intel Xeon Scalable Processors.")
print(response)
``` 
Output:
```
The Intel Xeon Scalable Processors represent a family of high-performance central processing units (CPUs) designed for data centers, cloud computing, and other demanding workloads. These processors offer significant improvements in performance, efficiency, and scalability compared to their predecessors.

Key features include:
1. Advanced architecture: The Intel Xeon Scalable Processors utilize the latest microarchitecture, which enables better performance and power efficiency.
2. Enhanced security: With built-in hardware-level security features like Intel Software Guard Extensions (SGX), these processors provide robust protection against potential threats.
3. Scalability: The modular design allows for flexible configurations, enabling users to choose the right combination of cores, memory capacity, and I/O capabilities to meet specific needs.
4. Optimized for virtualization: The Intel Xeon Scalable Processors are optimized for virtual environments, supporting multiple operating systems and applications on a single server.
5. High-speed connectivity: Integrated support for high-speed interconnects such as Intel Omni-Path Architecture (OPA) and Intel QuickAssist Technology (QAT) ensures efficient communication between servers and devices
```
<br>

The Complete code with output for creating the chatbot is given in the file [build_chatbot_on_spr.ipynb](https://github.com/JoyM268/Intel-Unnati-Industrial-Training-Program-2024/blob/main/build_chatbot_on_spr.ipynb).

### Prompt Examples:
**Example-1**:<br>
Prompt: 
```python
response1 = chatbot.predict(query="What is difference between GPU and CPU.")
print(response1) 
```

Response:
```
The main difference between a Graphics Processing Unit (GPU) and a Central Processing Unit (CPU) lies in their primary functions and architectures.

A CPU is the brain of a computer, responsible for executing instructions and managing various tasks. It's designed to handle general-purpose computing tasks like processing data, running programs, and controlling peripheral devices. CPUs have a single instruction stream and can perform multiple tasks simultaneously through multithreading or multiprocessing.

On the other hand, a GPU is specialized hardware optimized for handling complex mathematical calculations and rendering visuals. GPUs excel at parallel processing, which means they can execute many tasks simultaneously using thousands of small processing units called CUDA cores. This makes them ideal for graphics-intensive applications such as gaming, video editing, and scientific simulations.

In summary, while both CPUs and GPUs are essential components of modern computers, they serve different purposes. CPUs focus on general-purpose computing tasks, whereas GPUs specialize in high-performance graphics and parallel processing.
```
<br>

**Example-2**:<br>
Prompt: 
```python
response2 = chatbot.predict(query="Explain finetuning of a ai model.")
print(response2)
 ```
Response:
```
Finetuning of an AI model is a process where you take an existing pre-trained model and further adapt it for a specific task or domain. This technique helps improve the performance of the model on new data sets or tasks without starting from scratch.

Imagine you have a smart robot that can perform various tasks like cooking, cleaning, and playing music. At first, this robot was designed to only cook. However, as time went by, you wanted it to learn how to clean your house too. Instead of building a completely new robot for cleaning, you decide to teach the existing cooking robot some additional skills. That's what we call finetuning - adapting the original model (robot) to perform better in a different area (cleaning).

To finetune an AI model, you typically follow these steps:

1. Choose a pre-trained model: Start with a model that has already been trained on a large dataset for a particular task. For example, if you want to create a model that can recognize images, you might use a pre-trained image recognition model.

2. Prepare the new dataset: Collect a new set of data related to the desired task or domain.
```
<br>

**Example-3**:<br>
Prompt: 
```python
response3 = chatbot.predict(query="What are TPU, are they better than GPU")
print(response3)
 ```
Response:
```
TPU stands for Tensor Processing Unit, which is a specialized hardware designed for machine learning tasks. It's optimized for running deep neural networks efficiently. GPUs (Graphics Processing Units) were initially developed for rendering graphics in video games but have since been repurposed for various computing tasks, including machine learning.

Comparing TPUs and GPUs can be complex as it depends on the specific application and workload. Generally, TPUs offer higher performance for certain machine learning tasks like training large models or performing inference at scale. However, GPUs still excel in other areas such as general-purpose computing, gaming, and handling diverse workloads. So, while TPUs may be better suited for some AI applications, GPUs remain versatile and valuable across multiple domains.
```
<br>

**Example-4**:<br>
Prompt: 
```python
response4 = chatbot.predict(query="What is a Transformer")
print(response4)
 ```
Response:
```
A Transformer is a type of neural network architecture designed for natural language processing tasks. It was introduced in 2017 by researchers at Google AI. The model has shown impressive results in various NLP applications such as machine translation, text summarization, and question answering. Transformers learn to represent words and sentences through self-attention mechanisms, which allow them to understand the context and relationships between different parts of the input data. This makes them highly effective in understanding complex linguistic structures and generating meaningful outputs.
```
<br>

**Example-5**:<br>
Prompt: 
```python
response5 = chatbot.predict(query="What is a AGI")
print(response5)
 ```
Response:
```
A General Artificial Intelligence (AGI) refers to a hypothetical future AI system that can perform any intellectual task as well as or better than a human being. It would possess a high level of intelligence, creativity, problem-solving abilities, and adaptability across various domains. AGIs are still in the realm of science fiction, with current AI systems focusing on specific tasks and limited capabilities.
```

### Fine-tuning your Chatbot:
The following code is used to fine-tune the `TinyLlama_v1.1` model on the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca).

```python
from transformers import TrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model
model_args = ModelArguments(model_name_or_path="TinyLlama/TinyLlama_v1.1")
data_args = DataArguments(train_file="alpaca_data.json", validation_split_percentage=1)
training_args = TrainingArguments(
    output_dir='./tmp',
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    save_strategy="no",
    log_level="info",
    save_total_limit=2,
    bf16=True,
)
finetune_args = FinetuningArguments()
finetune_cfg = TextGenerationFinetuningConfig(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetune_args=finetune_args,
        )
finetune_model(finetune_cfg)
```

We are using the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) from Stanford University as the general domain dataset to fine-tune the model. This dataset is provided in the form of a JSON file, [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). In Alpaca, researchers have manually crafted 175 seed tasks to guide `text-davinci-003` in generating 52K instruction data for diverse tasks.

The Complete code with output for finetuning the model is given in the file [single_node_finetuning_on_spr.ipynb](https://github.com/JoyM268/Intel-Unnati-Industrial-Training-Program-2024/blob/main/single_node_finetuning_on_spr.ipynb).

A video explanation for the Intel Project is available on google drive, click the link: [Intel Project Video](https://drive.google.com/file/d/1y0zHbecpfFLH25lg24_Wu-sAX7mu-kjY/view?usp=sharing).
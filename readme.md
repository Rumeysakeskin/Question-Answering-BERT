## Extractive Question-Answering with BERT on SQuAD v2.0 (Stanford Question Answering Dataset)

- The main goal of extractive question-answering is to find the most relevant and accurate answer to a given question within the provided text passage. 
In other words, the model extracts the answer directly from the passage, rather than generating a new answer. 
- This provides quick and accurate answers.
- However, it is important to note that extractive question-answering is limited by the information contained within the provided text passage 
and may not be able to generate novel or creative answers.
This approach has been applied to customer service chatbots, search engines, voice assistants etc.
---
### Pre-requisitions
**!!! You can run either this notebook locally (if you have all the dependencies and a GPU) or on Google Colab.**

Please make sure of the following software requirements if you will work on local:
```python
python 3.6.9
docker-ce > 19.03.5
docker-API 1.40
nvidia-container-toolkit > 1.3.0-1
nvidia-container-runtime > 3.4.0-1
nvidia-docker2 > 2.5.0-1
nvidia-driver >= 455.23
```
---
### SQuAD v2.0 and Data Format and Conversion

 [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially 
 by crowdworkers to look similar to answerable ones.
 
 Here's an example format for a squad question-answering dataset:
 
- Each `title` has one or multiple `paragraph` entries, each consisting of the `context` and `question-answer entries (qas)`.

- Each question-answer entry has a `question` and a globally unique `id`
 
- The Boolean flag `is_impossible`, which shows whether a question is answerable or not: If the question is answerable, one `answer` entry contains the text span and its starting character index in the context. If the question is not answerable, an empty `answers` list is provided.

**!!! For the QA task, NVIDIA toolkit accepts data in the SQuAD JSON format. 
If you have your data in any other format, be sure to convert it in the SQuAD format as below.**

```python
{
    "data": [
        {
            "title": "Super_Bowl_50",
            "paragraphs": [
                {
                    "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
                    "qas": [
                        {
                            "question": "Where did Super Bowl 50 take place?",
                            "is_impossible": "false",
                            "id": "56be4db0acb8001400a502ee",
                            "answers": [
                                {
                                    "answer_start": "403",
                                    "text": "Santa Clara, California"
                                }
                            ]
                        },
                        {
                            "question": "What was the winning score of the Super Bowl 50?",
                            "is_impossible": "true",
                            "id": "56be4db0acb8001400a502ez",
                            "answers": [
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}
```
---
### Configuration Files
 It's essentially just one command each to run data preprocessing, training, fine-tuning, evaluation, inference, and export! 
 All configurations happen through YAML spec files.
 There are sample spec files already [available](https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/question_answering/conf) for you to use directly or as reference to create your own. Through these spec files, you can tune many knobs like the model, dataset, hyperparameters, optimizer etc.

---
### Fine-tune BERT QA on the SQuAD Dataset
For training a QA model in NVIDIA format, we use following commands:
```python 
# set language model and tokenizer to be used
config.model.language_model.pretrained_model_name = "bert-base-uncased"
config.model.tokenizer.tokenizer_name = "bert-base-uncased"

# path where model will be saved
config.model.nemo_path = f"{WORK_DIR}/checkpoints/bert_squad_v2_0.nemo"

trainer = pl.Trainer(**config.trainer)
model = BERTQAModel(config.model, trainer=trainer)
trainer.fit(model)
trainer.test(model)

model.save_to(config.model.nemo_path)
```
More details about these arguments are present in the [Question_Answering.ipynb](https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/Question_Answering.ipynb)

---
### BERT QA Inference 

**!!! The evaluation files (for validation and testing) follow the above format except for it can provide more than one answer to the same question.** 
**!!!The inference file follows the above format except for it does not require the `answers` and `is_impossible` keywords.**

```python 
# Load saved model
model = BERTQAModel.restore_from(config.model.nemo_path)

eval_device = [config.trainer.devices[0]] if isinstance(config.trainer.devices, list) else 1
model.trainer = pl.Trainer(
    devices=eval_device,
    accelerator=config.trainer.accelerator,
    precision=16,
    logger=False,
)
config.exp_manager.create_checkpoint_callback = False
exp_dir = exp_manager(model.trainer, config.exp_manager)

def dump_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f)

def create_inference_data_format(context, question):

  squad_data = {"data": [{"title": "inference", "paragraphs": []}], "version": "v2.1"}
  squad_data["data"][0]["paragraphs"].append(
            {
                "context": context,
                "qas": [
                    {"id": 0, "question": question,}
                ],
            }
        )
  return squad_data

context = "The Amazon rainforest is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%."

question = "Which country has the most?"

inference_filepath = "inference.json"

inference_data = create_inference_data_format(context, question)
dump_json(inference_filepath, inference_data)

predictions = model.inference("inference.json")
question = predictions[1][0][0]["question"]
answer = predictions[1][0][0]["text"]
probability = predictions[1][0][0]["probability"]

print(f"\n> Question: {question}\n> Answer: {answer}\n Probability: {probability}")
```

```python
100%|██████████| 1/1 [00:00<00:00, 184.29it/s]
100%|██████████| 1/1 [00:00<00:00, 8112.77it/s]

> Question: Which country has the most?
> Answer: Brazil
Probability: 0.9688649039578262
```
---
In this study, it is utilized from [question-answering-training-final.ipynb](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/tao_question/version/1/files/question-answering-training-final.ipynb#training). You can find more information here.

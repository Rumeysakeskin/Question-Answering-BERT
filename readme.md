### Extractive Question-Answering with BERT on SQuAD v2.0 (Stanford Question Answering Dataset)

- The main goal of extractive question-answering is to find the most relevant and accurate answer to a given question within the provided text passage. 
In other words, the model extracts the answer directly from the passage, rather than generating a new answer. 
- This provides quick and accurate answers.
- However, it is important to note that extractive question-answering is limited by the information contained within the provided text passage 
and may not be able to generate novel or creative answers.
This approach has been applied to customer service chatbots, search engines, voice assistants etc.



You can run either this notebook locally (if you have all the dependencies and a GPU) or on Google Colab.

### SQuAD v2.0 Dataset

 [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially 
 by crowdworkers to look similar to answerable ones.
 
 Here's an example format for a squad question-answering dataset:

```python
{
  "data": [
    {
      "title": "Example Passage",
      "paragraphs": [
        {
          "context": "This is an example passage. It is used to demonstrate the format of a squad question-answering dataset. In order to create a squad dataset, you need to provide a passage of text and a list of questions and answers that relate to that passage.",
          "qas": [
            {
              "question": "What is the purpose of an example passage?",
              "id": "1",
              "answers": [
                {
                  "text": "It is used to demonstrate the format of a squad question-answering dataset.",
                  "answer_start": 32
                }
              ]
            },
            {
              "question": "What do you need to provide in order to create a squad dataset?",
              "id": "2",
              "answers": [
                {
                  "text": "A passage of text and a list of questions and answers that relate to that passage.",
                  "answer_start": 140
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

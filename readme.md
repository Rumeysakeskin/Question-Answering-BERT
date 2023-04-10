## Extractive Question-Answering with BERT on SQuAD v2.0 (Stanford Question Answering Dataset)

- The main goal of extractive question-answering is to find the most relevant and accurate answer to a given question within the provided text passage. 
In other words, the model extracts the answer directly from the passage, rather than generating a new answer. 
- This provides quick and accurate answers.
- However, it is important to note that extractive question-answering is limited by the information contained within the provided text passage 
and may not be able to generate novel or creative answers.
This approach has been applied to customer service chatbots, search engines, voice assistants etc.



You can run either this notebook locally (if you have all the dependencies and a GPU) or on Google Colab.

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

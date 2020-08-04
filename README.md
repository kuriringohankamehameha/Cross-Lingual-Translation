# Cross-Lingual-Translation
A Statistical model that translates across languages via phrase based translation using the `IBM Model` and the `Expectation Maximization (EM)` Algorithm.

## Prerequisites
* This model is implemented in Python3. No external libraries are required.
* The dataset used for training the model is in the subdirectory `dataset/`.

## Sample Queries
```bash
>> ...
>> import cldt
>> from cldt import CLDT
>> cldt.train(('english', 'dutch'), ('English.txt', 'Dutch.txt'))
>> cldt.prepare_translation(('english', 'dutch')) 
>> CLDT(input='Good morning', to_lang='dutch', is_file=False)
'Goedemorgen'
>> CLDT(input='How are you', to_lang='dutch')
'Hoe is het met je'
>> CLDT('Is het onmogelijk', 'english')
'Is it impossible?'
>> similarity = cldt.process_documents(langs=('english', 'dutch'), input=('Test1.txt', 'Test2.txt'), output=('Op1.txt', 'Op2.txt'))
>> print(similarity['cosine'], similarity['pearson']) 
>> avg = cldt.avg_similarity(similarity)
>> print(avg['cosine'], avg['pearson'])
>> ...
```

# Yelp Restaurant Review Named Entity Recognition Pipeline Building and Experiments

This project is consists of a pipeline building for NER on Yelp restaurant reviews,
along with some feature ablation experiments.

The entity types are as follow:
* Dish (DISH): food ("burger") or drink ("beer") that could be ordered from a menu or served.
* Ingredient (INGRED): specific references to ingredients as actual or potential sub-parts of dishes ("the [bun] was").
* Quality (QUAL): references to taste, consistency, or quality of dishes or ingredients ("yummy", "crumbly", "oily", "terrible").
* Business (BIZ): the names of businesses referred to ("Prime Deli"), either the one the review is about or other businesses.
* Service (SERV): Services or websites related to restaurant reviews, menus, and deliveries: GrubHub, Uber Eats, Postmates, Yelp.

### Data

The data is stored in batch_9_meiqw.jsonl file, the file is consists of 250 yelp
restaurant reviews I gathered online and annotated using Doccano.

### Pipeline and Experiments

The NER pipeline is in restaurant_ner.py, running the file will print out
evaluation of the model on the entity types.

Changing feature extractors in the main() section of the file will print out
different feature experiment results.

### Unit testing

The unit tests is in test.py.

### Prerequisites

The python packages listed in requirements.txt need to be installed for the program to run.

### Installing
Use pip to install the dependencies.

```
pip install -r requirements.txt
```

Then run

```
python -m spacy download en_core_web_sm
```

to install the spaCy models required for this assignment

### Contribution

Util_files folder, utils.py and annotation guideline is contributed by Professor
Constantine Lignos at Brandeis University.

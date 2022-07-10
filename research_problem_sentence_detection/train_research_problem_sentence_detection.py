import json, os, glob
import tensorflow as tf
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(data_dir):
    articles = []
    research_problem_sentences = []

    for category in os.listdir(data_dir):
        if category != 'README.md' and category != '.git':
            article_category = os.path.join(data_dir, category)
            if os.path.isfile(article_category):
                continue

            for folder_name in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category, folder_name)

                if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) == 0:
                    continue

                with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                    article = f.read()
                    articles.append(article.lower())

                with open(os.path.join(article_index, 'info-units/research-problem.json'), encoding='utf-8') as f:
                    research_problem_sentences.append([element[-1].get("from sentence", None) if isinstance(
                        element, list) else element.get("from sentence", None) if isinstance(element, dict) else None for element in json.load(f)["has research problem"]])
    return articles, research_problem_sentences


def article2SentenceAndLabels(articles, research_problem_sentences):
    sentences = []
    labels = []
    for i, article in enumerate(articles):
        for sentence in article.split('\n')[0:-1]:
            sentences.append(sentence)
            labels.append(int(sentence in research_problem_sentences[i]))
    return sentences, labels


train_data_dir = '/content/drive/MyDrive/Colab Notebooks/datasets/training-data'

train_articles, train_research_problem_sentences = loadArticles(train_data_dir)

train_sentences, train_labels = article2SentenceAndLabels(
    train_articles, train_research_problem_sentences)

train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_sentences, train_labels, test_size=.2)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
val_encodings = tokenizer(val_sentences, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
print('train data loaded:({0})'.format(len(train_labels)))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))
print('validation data loaded:({0})'.format(len(val_labels)))

model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', config=config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset.shuffle(len(train_labels)).batch(BATCH_SIZE),
          validation_data=val_dataset.batch(BATCH_SIZE),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

model.save_weights('/content/drive/MyDrive/Colab Notebooks/small-fun-learning-task/outcome/model-SC-BERT/')
print('Model saved.')

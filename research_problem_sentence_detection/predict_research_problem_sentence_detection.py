import os, glob
import tensorflow as tf
import numpy as np
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification

EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(data_dir):
    articles = []
    catalogues = []  # relative folder addresses

    for category in os.listdir(data_dir):
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
            article_category = os.path.join(data_dir, category)
            if os.path.isfile(article_category):
                continue

            for folder_name in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category, folder_name)
                indices = article_index.split('/')
                catalogue = indices[-2] + '/' + indices[-1]
                if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) != 0:
                    catalogues.append(catalogue)

                    with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                        article = f.read()
                        articles.append(article.lower())
    return articles, catalogues


def article2SentenceAndLabels(articles):
    article_sentences = []
    article_labels = []
    sentence_count = []
    for article in articles:
        article_sentence = []
        article_label = []
        count = 0
        for sentence in article.split('\n')[0:-1]:
            article_sentence.append(sentence)
            article_label.append(-1)
            count += 1
        article_sentences.append(article_sentence)
        article_labels.append(article_label)
        sentence_count.append(count)
    return article_sentences, article_labels, sentence_count


test_data_dir = 'test-data-master'

articles, catalogues = loadArticles(test_data_dir)

article_sentences, article_labels, sent_count = article2SentenceAndLabels(
    articles)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

test_sentences = []
test_labels = []
for i, article_sentence in enumerate(article_sentences):
    for j, sentence in enumerate(article_sentence):
        test_sentences.append(sentence)
        test_labels.append(article_labels[i][j])

test_encodings = tokenizer(test_sentences, truncation=True, padding=True)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))

print('test data loaded: ({0})'.format(len(test_labels)))

model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', config=config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.load_weights('small-fun-learning-task/outcome/model-SC-BERT/')
print('Model loaded.')

y_pred = model.predict(test_dataset.batch(BATCH_SIZE))
np.set_printoptions(threshold=1e6)
result = np.argmax(y_pred[0], axis=-1)

if not os.path.exists('small-fun-learning-task/outcome/predicted_sentences'):
    os.makedirs('small-fun-learning-task/outcome/predicted_sentences')

start = 0  # start index of any article sentences in the result array
for i, count in enumerate(sent_count):
    end = start + count
    article_label = result[start:end]
    article_catalogue = catalogues[i]

    if not os.path.exists('small-fun-learning-task/outcome/predicted_sentences/' + article_catalogue):
        os.makedirs(
            'small-fun-learning-task/outcome/predicted_sentences/' + article_catalogue)

    with open(os.path.join('small-fun-learning-task/outcome/predicted_sentences/' + article_catalogue, 'sentences_with_research_problem.txt'), 'w', encoding='utf-8') as f:
        for j in range(len(article_label)):
            if article_label[j] == 1:
                f.write(str(j + 1) + '\n')
    start = end

print("Predicted sentences with research problems saved.")

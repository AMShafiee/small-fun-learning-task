import json, os, glob
import tensorflow as tf
from transformers import BertConfig, BertTokenizerFast, TFBertForSequenceClassification
from keras import backend as K

EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(data_dir):
    articles = []
    research_problem_sentences = []

    for category in os.listdir(data_dir):
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
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

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value)/(precision_value + recall_value + K.epsilon()))

# class CategoricalTruePositives(keras.metrics.Metric):
#     def __init__(self, name="categorical_true_positives", **kwargs):
#         super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name="ctp", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
#         values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
#         values = tf.cast(values, "float32")
#         if sample_weight is not None:
#             sample_weight = tf.cast(sample_weight, "float32")
#             values = tf.multiply(values, sample_weight)
#         self.true_positives.assign_add(tf.reduce_sum(values))

#     def result(self):
#         return self.true_positives

#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.true_positives.assign(0.0)


# test_data_dir = '/content/drive/MyDrive/Colab Notebooks/datasets/trial-data'
test_data_dir = '/content/drive/MyDrive/Colab Notebooks/datasets/test-data'

articles, research_problem_sentences = loadArticles(test_data_dir)

test_sentences, test_labels = article2SentenceAndLabels(
    articles, research_problem_sentences)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

test_encodings = tokenizer(test_sentences, truncation=True, padding=True)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))
print('test data loaded:({0})'.format(len(test_labels)))

model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', config=config)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              # loss='cosine_similarity', #'categorical_crossentropy',
              # metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
              metrics=['accuracy', recall, precision, f1_score])
              #  metrics=[tf.keras.metrics.AUC(from_logits=True)])

# model.compile(optimizer='sgd',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=[tf.keras.metrics.AUC(from_logits=True)])               

model.load_weights('/content/drive/MyDrive/Colab Notebooks/small-fun-learning-task/outcome/model-SC-BERT/')
print('Model loaded.')

test_loss, test_accuracy, test_recall, test_precision, test_f1_score = model.evaluate(
    test_dataset.batch(BATCH_SIZE), batch_size=BATCH_SIZE)
print('test loss:', test_loss)
print('test accuracy:', test_accuracy)
print('test recall:', test_recall)
print('test precision:', test_precision)
print('test f1_score:', test_f1_score)
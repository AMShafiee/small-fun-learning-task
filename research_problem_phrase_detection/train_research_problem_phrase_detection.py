from keras import backend as K  
import os, glob, json
import numpy as np
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForTokenClassification
from sklearn.model_selection import train_test_split

MAX_TOKEN = 512
EPOCHS = 3
BATCH_SIZE = 8


def loadArticles(train_data_dir, trial_data_dir):
    articles = []
    research_problems = []

    for category in os.listdir(train_data_dir):
        if category != 'README.md' and category != '.git':
            article_category = os.path.join(train_data_dir, category)
            if os.path.isfile(article_category):
                continue

            for folder_name in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category,
                                             folder_name)
                if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) == 0:
                    continue
                with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                    article = f.read()
                    articles.append(article.lower())

                with open(os.path.join(article_index, 'info-units/research-problem.json'), encoding='utf-8') as f:
                    research_problem = []
                    json_data = json.load(f)["has research problem"]
                    if isinstance(json_data[0], list):
                        for each_element in json_data:
                            research_problem.append(
                                {'sentence': each_element[-1].get("from sentence", None), 'phrases': each_element[:-1]})
                    else:
                        research_problem.append(
                            {'sentence': json_data[-1].get("from sentence", None), 'phrases': json_data[:-1]})
                    research_problems.append(research_problem)

                # with open(os.path.join(article_index, 'entities.txt'), encoding='utf-8') as f:
                #     entity = {}
                #     for line in f.readlines():
                #         line = line.strip().split('\t')
                #         if len(line) > 0:
                #             article_sentence = int(line[0])
                #             span = (int(line[1]), int(line[2]))
                #             if article_sentence not in entity:
                #                 entity[article_sentence] = {'spans': []}
                #             entity[article_sentence]['spans'].append(span)
                #     entities.append(entity)

    for category in os.listdir(trial_data_dir):
        if category != 'README.md' and category != '.git':
            article_category = os.path.join(trial_data_dir, category)
            if os.path.isfile(article_category):
                continue

            for folder_name in sorted(os.listdir(article_category)):
                article_index = os.path.join(article_category,
                                             folder_name)
                if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) == 0:
                    continue
                with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                    article = f.read()
                    articles.append(article.lower())

                with open(os.path.join(article_index, 'info-units/research-problem.json'), encoding='utf-8') as f:
                    research_problem = []
                    json_data = json.load(f)["has research problem"]
                    if isinstance(json_data[0], list):
                        for each_element in json_data:
                            research_problem.append(
                                {'sentence': each_element[-1].get("from sentence", None), 'phrases': list(map(str.lower, each_element[:-1]))})
                    else:
                        research_problem.append(
                            {'sentence': json_data[-1].get("from sentence", None), 'phrases': list(map(str.lower, json_data[:-1]))})
                    research_problems.append(research_problem)

                # with open(os.path.join(article_index, 'entities.txt'), encoding='utf-8') as f:
                #     entity = {}
                #     for line in f.readlines():
                #         line = line.strip().split('\t')
                #         if len(line) > 0:
                #             article_sentence = int(line[0])
                #             span = (int(line[1]), int(line[2]))
                #             if article_sentence not in entity:
                #                 entity[article_sentence] = {'spans': []}
                #             entity[article_sentence]['spans'].append(span)
                #     entities.append(entity)
    return articles, research_problems


def article2ResearchProblemSentenceAndSpans(articles, research_problems):
    research_problems_sentences = []
    research_problems_spans = []
    for i, article in enumerate(articles):
        article_sentences = article.split('\n')[0:-1]

        for each_element in research_problems[i]:
            try:
                sentence_index = next(index for index, sentence in enumerate(
                    article_sentences) if each_element['sentence'] in sentence)
                research_problems_spans.append([(article_sentences[sentence_index].find(each_phrase), article_sentences[sentence_index].find(
                    each_phrase) + len(each_phrase)) for each_phrase in each_element['phrases']])
                research_problems_sentences.append(
                    article_sentences[sentence_index])
            except:
                for each_phrase in each_element['phrases']:
                    try:
                        sentence_index = next(index for index, sentence in enumerate(
                            article_sentences) if each_phrase in sentence)
                        research_problems_spans.append([(article_sentences[sentence_index].find(
                            each_phrase), article_sentences[sentence_index].find(each_phrase) + len(each_phrase))])
                        research_problems_sentences.append(
                            article_sentences[sentence_index])
                    except:
                        pass
    return research_problems_sentences, research_problems_spans


def alignSpansBySentence(research_problem_sentences, research_problem_spans):
    sent_tokens = []
    sent_token_ids = []
    sent_token_spans = []
    sent_token_tags = []

    maxTokenLen = 0
    for i, sent in enumerate(research_problem_sentences):
        sent_spans = research_problem_spans[i]
        tokens = tokenizer.tokenize(sent)
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        sent_tokens.append(tokens)

        token_spans = []
        token_ids = np.zeros(len(tokens), dtype='int32')
        token_tags = np.zeros(len(tokens), dtype='int')

        end = 0
        for index, token in enumerate(tokens):
            token_id = tokenizer.convert_tokens_to_ids(token)
            token_ids[index] = token_id

            if token in ['[CLS]', '[UNK]']:
                token_spans.append((end, end))
            elif token == '[SEP]':
                token_spans.append((end, end))
            else:
                token = token.replace('##', '')
                start = sent.find(token, end)
                end = start + len(token)
                if len(sent_spans) != 0:
                    for sent_span in sent_spans:
                        if start >= sent_span[0] and end <= sent_span[1]:
                            token_tags[index] = 1
                token_spans.append((start, end))

        sent_token_ids.append(token_ids)

        sent_token_spans.append(token_spans)

        sent_token_tags.append(token_tags)

        maxTokenLen = len(tokens) if maxTokenLen < len(tokens) else maxTokenLen

    return maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags


def chunkData(maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags):
    input_ids = []
    input_masks = []
    input_tags = []
    token_spans = sent_token_spans

    if maxTokenLen > MAX_TOKEN:
        maxTokenLen = MAX_TOKEN
    for i, token_ids in enumerate(sent_token_ids):
        ids = np.zeros(maxTokenLen, dtype=int)
        ids[0:len(token_ids)] = token_ids
        input_ids.append(ids)

        mask = np.copy(ids)
        mask[mask > 0] = 1
        input_masks.append(mask)

        token_tags = sent_token_tags[i]
        tags = np.zeros(maxTokenLen, dtype=int)
        tags[0:len(token_tags)] = token_tags
        input_tags.append(tags)

    return input_ids, input_masks, input_tags, token_spans


def buildData(research_problem_sentences, research_problem_spans):
    maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans, sent_token_tags \
        = alignSpansBySentence(research_problem_sentences, research_problem_spans)
    input_ids, input_masks, input_tags, token_spans = \
        chunkData(maxTokenLen, sent_tokens, sent_token_ids,
                  sent_token_spans, sent_token_tags)

    x = dict(
        input_ids=np.array(input_ids, dtype=np.int32),
        attention_mask=np.array(input_masks, dtype=np.int32),
        token_type_ids=np.zeros(shape=(len(input_ids), maxTokenLen))
    )
    y = np.array(input_tags, dtype=np.int32)
    return x, y


train_data_dir = 'training-data-master'
trial_data_dir = 'trial-data'
articles, research_problems = loadArticles(train_data_dir, trial_data_dir)

train_sentences, train_spans = article2ResearchProblemSentenceAndSpans(
    articles, research_problems)


train_sentences, val_sentences, train_spans, val_spans = train_test_split(
    train_sentences, train_spans, test_size=.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

train_x, train_y = buildData(train_sentences, train_spans)
train_x["token_type_ids"] = K.cast(
    train_x["token_type_ids"], dtype='int64')

train_dataset = tf.data.Dataset.from_tensor_slices((
    train_x,
    train_y
))
print('train data loaded:({0})'.format(len(train_y)))

val_x, val_y = buildData(val_sentences, val_spans)
val_x["token_type_ids"] = K.cast(
    val_x["token_type_ids"], dtype='int64')

val_dataset = tf.data.Dataset.from_tensor_slices((
    val_x,
    val_y
))
print('validation data loaded:({0})'.format(len(val_y)))

model = TFBertForTokenClassification.from_pretrained(
    'bert-base-uncased', config=config)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset.shuffle(len(train_y)).batch(BATCH_SIZE),
          validation_data=val_dataset.batch(BATCH_SIZE),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

model.save_weights('small-fun-learning-task/outcome/model-SI-BERT/')
print('Model saved.')

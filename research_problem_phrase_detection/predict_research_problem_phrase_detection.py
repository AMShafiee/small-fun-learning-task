import os, glob, json
import numpy as np
import tensorflow as tf
from transformers import BertConfig, BertTokenizerFast, TFBertForTokenClassification

EPOCHS = 3
BATCH_SIZE = 8
MAX_TOKEN = 512


def loadArticles(data_dir):
    articles = []
    articles_raw = []
    catalogues = []

    for category in sorted(os.listdir(data_dir)):
        if category != 'README.md' and category != '.git' and category != 'submission.zip':
            article_category = os.path.join(data_dir, category)
            if os.path.isfile(article_category):
                continue

            for folder_name in sorted(
                    os.listdir(article_category)):
                if folder_name != '.ipynb_checkpoints':
                    article_index = os.path.join(article_category, folder_name)
                    if len(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))) == 0:
                        continue

                    indices = article_index.split('/')
                    catalogue = indices[-2] + '/' + indices[-1]
                    catalogues.append(catalogue)

                    with open(glob.glob(os.path.join(article_index, '*-Stanza-out.txt'))[0], encoding='utf-8') as f:
                        article = f.read()
                        articles.append(article.lower())
                        articles_raw.append(article)

    return articles, articles_raw, catalogues


def loadSentences(data_dir, articles, catalogues):
    article_index = 0
    research_problem_lines = []

    # for category in os.listdir(data_dir):
    #     if category != 'README.md' and category != '.git' and category != 'submission.zip':
    #         article_category = os.path.join(data_dir, category)
    #         if os.path.isfile(article_category):
    #             continue

    #         for folder_name in sorted(os.listdir(article_category)):
    #             article_category_and_folder_name = os.path.join(article_category, folder_name)
    #             if os.path.isfile(article_category_and_folder_name):
    #                 continue
    for each_catalogue in catalogues:
                article_category_and_folder_name = os.path.join(data_dir, each_catalogue)
                article_sentences = articles[article_index].split('\n')[0:-1]
                article_index += 1

                with open(os.path.join(article_category_and_folder_name, 'info-units/research-problem.json'), encoding='utf-8') as f:
                    research_problem = []
                    research_problem_line = []
                    json_data = json.load(f)["has research problem"]
                    if isinstance(json_data[0], list):
                        for each_element in json_data:
                            research_problem.append(
                                {'sentence': each_element[-1].get("from sentence", None), 'phrases': each_element[:-1]})
                    else:
                        research_problem.append(
                            {'sentence': json_data[-1].get("from sentence", None), 'phrases': json_data[:-1]})

                    for each_element in research_problem:
                        try:
                            sentence_index = next(index for index, sentence in enumerate(
                                article_sentences) if each_element['sentence'].lower() in sentence)
                            research_problem_line.append(sentence_index)
                        except:
                            for each_phrase in each_element['phrases']:
                                try:
                                    sentence_index = next(index for index, sentence in enumerate(
                                        article_sentences) if each_phrase.lower() in sentence)
                                    research_problem_line.append(sentence_index)
                                except:
                                    pass
                    research_problem_lines.append(research_problem_line)
    return research_problem_lines


def article2ResearchProblemSentence(articles, articles_raw, research_problems):
    research_problem_sentences = []
    research_problem_sentences_raw = []
    research_problems_ascend = []
    sent_count = []
    for i, article in enumerate(articles):
        research_problem = research_problems[i]

        count = 0
        sentences = article.split('\n')[0:-1]
        for row, sentence in enumerate(sentences):
            if (row + 1) in research_problem:
                research_problem_sentences.append(sentence)
                count += 1
        sent_count.append(count)

        sentences_raw = articles_raw[i].split('\n')[0:-1]
        research_problem_sentence_raw = []
        research_problem_ascend = []
        for row, sentence_raw in enumerate(sentences_raw):
            if (row + 1) in research_problem:
                research_problem_sentence_raw.append(sentence_raw)
                research_problem_ascend.append(row + 1)
        research_problem_sentences_raw.append(research_problem_sentence_raw)
        research_problems_ascend.append(research_problem_ascend)
    return research_problem_sentences, research_problem_sentences_raw, research_problems_ascend, sent_count


def alignSpansBySentence(contribution_sentences):
    sent_tokens = []
    sent_token_ids = []
    sent_token_spans = []

    maxTokenLen = 0
    for i, sent in enumerate(contribution_sentences):
        tokens = tokenizer.tokenize(sent)
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        sent_tokens.append(tokens)

        token_spans = []
        token_ids = np.zeros(len(tokens), dtype='int32')

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

                token_spans.append((start, end))

        sent_token_ids.append(token_ids)

        sent_token_spans.append(token_spans)

        maxTokenLen = len(tokens) if maxTokenLen < len(tokens) else maxTokenLen

    return maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans


def chunkData(maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans):
    input_ids = []
    input_masks = []
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

    return input_ids, input_masks, token_spans


def buildData(contribution_sentences):
    maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans = alignSpansBySentence(
        contribution_sentences)
    input_ids, input_masks, token_spans = chunkData(
        maxTokenLen, sent_tokens, sent_token_ids, sent_token_spans)

    x = dict(
        input_ids=np.array(input_ids, dtype=np.int32),
        attention_mask=np.array(input_masks, dtype=np.int32),
        token_type_ids=np.zeros(shape=(len(input_ids), maxTokenLen))
    )
    return x, token_spans, maxTokenLen


test_article_dir = '/content/drive/MyDrive/Colab Notebooks/datasets/evaluation-phase1'
test_sentence_dir = '/content/drive/MyDrive/Colab Notebooks/datasets/evaluation-phase2'
articles, articles_raw, catalogues = loadArticles(test_article_dir)
research_problems = loadSentences(test_sentence_dir, articles, catalogues)
test_sentences, research_problem_sentences_raw, research_problems_ascend, sent_count = article2ResearchProblemSentence(articles,
                                                                                                            articles_raw,
                                                                                                            research_problems)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2

test_x, token_spans, maxTokenLen = buildData(test_sentences)
print('test data loaded:({0})'.format(len(test_sentences)))

model = TFBertForTokenClassification.from_pretrained(
    'bert-base-uncased', config=config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.load_weights('/content/drive/MyDrive/Colab Notebooks/small-fun-learning-task/outcome/model-SI-BERT/')
print('Model Loaded.')

np.set_printoptions(threshold=1e6)
y_pred = model.predict(test_x, batch_size=BATCH_SIZE)
result = np.argmax(y_pred[0], axis=-1)


start = 0
for article_category_and_folder_name, sent_num in enumerate(sent_count):
    end = start + sent_num

    article_tags = []
    article_spans = []
    article_catalogue = catalogues[article_category_and_folder_name]

    if not os.path.exists('/content/drive/MyDrive/Colab Notebooks/small-fun-learning-task/outcome/evaluation-phase2/' + article_catalogue + '/triples'):
        os.makedirs('/content/drive/MyDrive/Colab Notebooks/small-fun-learning-task/outcome/evaluation-phase2/' +
                    article_catalogue + '/triples')

    for j in range(start, end):
        article_tags.append(list(result[j]))
        article_spans.append(token_spans[j])

    predict_article_spans = []
    for m in range(len(article_tags)):
        predict_sentence_spans = []
        for n in range(maxTokenLen):
            if article_tags[m][n] == 1:
                spans = article_spans[m][n]
                predict_sentence_spans.append(spans)
        predict_article_spans.append(predict_sentence_spans)

    predict_article_spans_list = []
    for i in range(len(predict_article_spans)):
        sentence_spans = []
        for j in range(len(predict_article_spans[i])):
            sentence_spans.append(list(predict_article_spans[i][j]))
        predict_article_spans_list.append(sentence_spans)

    union_predict_article_spans = []
    for i in range(len(predict_article_spans_list)):
        sentence_spans = predict_article_spans_list[i]
        union_sentence_spans = []
        for sentence_span in sentence_spans:
            union_sentence_spans.append(list(sentence_span))
        for j in range(len(union_sentence_spans) - 1):
            for i in range(len(union_sentence_spans) - 1):
                if union_sentence_spans[i][1] == union_sentence_spans[i + 1][0] or union_sentence_spans[i + 1][0] - \
                        union_sentence_spans[i][1] == 1:
                    union_sentence_spans[i][1] = union_sentence_spans[i + 1][1]
                    del union_sentence_spans[i + 1]
                    break
        union_predict_article_spans.append(union_sentence_spans)

    contribution_sentence_row = []
    contribution_sentence_content = []
    for sentence_index in range(sent_num):
        contribution_sentence_row.append(
            research_problems_ascend[article_category_and_folder_name][sentence_index])
        contribution_sentence_content.append(
            research_problem_sentences_raw[article_category_and_folder_name][sentence_index])

    with open(os.path.join('/content/drive/MyDrive/Colab Notebooks/small-fun-learning-task/outcome/evaluation-phase2/' + article_catalogue, 'predicted_research_problem.txt'),
              'w', encoding='utf-8') as f:

        for m in range(len(union_predict_article_spans)):
            for n in range(len(union_predict_article_spans[m])):
                spans = union_predict_article_spans[m][n]
                contribution_sentence_content_spans = contribution_sentence_content[
                    m][spans[0]:spans[1]]
                f.write(str(contribution_sentence_row[m]) + '\t' + str(spans[0]) + '\t' + str(
                    spans[1]) + '\t' + contribution_sentence_content_spans + '\n')

    start = end

print("Predicted phrases with research problems saved.")
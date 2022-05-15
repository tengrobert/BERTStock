from tensorflow.contrib import predictor
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

df = pd.read_csv('./data.csv', index_col=0)
df['text'] = df['text'].astype(str)
df['date'] = pd.to_datetime(df['date'])
df_test = df[df['date'] >= '2019-09-30']
def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])
        
    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

# test_InputExamples = df_test.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
#                                                                 text_a = x['text'], 
#                                                                 text_b = None, 
#                                                                 label = x['label']), axis = 1)
MAX_SEQ_LENGTH = 25
# Convert our train and test features to InputFeatures that BERT understands.
# label_list = [0, 1]
# test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


# predict_input_fn = run_classifier.input_fn_builder(features=test_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

# estimator = tf.contrib.estimator.SavedModelEstimator('bert/1578230861')
# predictions = list(estimator.predict(predict_input_fn))
# for p in predictions:
#     print(p)
# predict_fn = predictor.from_saved_model('bert/1578232374')
# for test_feature in test_features:
#     test_feature = test_feature.__dict__
#     test_feature.pop("is_real_example", "None")
#     test_feature["label_ids"] = test_feature.pop("label_id")
#     test_feature = {key: [value] for key, value in test_feature.items()}
#     result = predict_fn(test_feature)
# print(test_InputExamples.values)


def getPrediction(in_sentences):
    predict_fn = predictor.from_saved_model('bert3/1578257305')
    label_list = [0, 1]
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    output = []
    for sentence, input_feature in zip(in_sentences, input_features):
        input_feature = input_feature.__dict__
        input_feature.pop("is_real_example", "None")
        input_feature["label_ids"] = input_feature.pop("label_id")
        input_feature = {key: [value] for key, value in input_feature.items()}
        result = predict_fn(input_feature)
        output.append(result['labels'])
    return output

# df_test['predicted_labels'] = getPrediction(df_test['text'].values)
# df_test.to_csv('./pred3.csv', index=False)
sss = [
    'China has been paying tarrifs to the USA'
]
print(getPrediction(sss))
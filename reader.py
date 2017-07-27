#!/usr/bin/env python
#coding=utf-8
import os
import random
import json


def train_reader(data_list, is_train=True):
    def reader():
        # every pass shuffle the data list again
        if is_train:
            random.shuffle(data_list)

        for train_sample in data_list:
            data = json.load(open(train_sample, "r"))
            '''
            the reader feeds data to the six data layers defined in paddle_model.py,
            they are:
            1. Questions:
                type: paddle.data_type.integer_value_sequence(vocab_size): []
            2. Documents:
                type: paddle.data_type.integer_value_sequence(vocab_size): []
            3. SameAsQuestion:
                type: paddle.data_type.dense_vector_sequence(1): [[]]
            4. CorrectSentence:
                type: paddle.data_type.integer_value_sequence(2): []
            5. CorrectStartWord:
                type: paddle.data_type.integer_value_sequence(2): []
            6. CorrentEndWord:
                type: paddle.data_type.integer_value_sequence(2): []
            '''

            doc_len = len(data['context'])
            same_as_question_word = [[[x]]
                                     for x in data['same_as_question_word']]

            ans_sentence = [0] * doc_len
            ans_sentence[data['ans_sentence']] = 1

            ans_start = [0] * doc_len
            ans_start[data['ans_start']] = 1

            ans_end = [0] * doc_len
            ans_end[data['ans_end']] = 1
            yield (data['question'], data['context'], same_as_question_word,
                   ans_sentence, ans_start, ans_end)

    return reader


if __name__ == "__main__":
    from paddle_train import choose_samples, load_config

    train_list, dev_list = choose_samples("featurized")
    for i, item in enumerate(train_reader(train_list)()):
        print(item)
        if i > 5: break

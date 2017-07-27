#!/usr/bin/env python
#coding=utf-8
import os
import random
import pdb
import json


def train_reader(data_list):
    def reader():
        # every pass shuffle the data list again
        random.shuffle(data_list)

        for train_sample in data_list:
            data = json.load(open(train_sample, "r"))
            '''
            the reader returen data for six data layers defined in
            paddle_model.py, they are:
            1. Questions
            2. Documents
            3. SameAsQuestion
            4. CorrectSentence
            5. CorrectStartWord
            6. CorrentEndWord
            '''
            yield data['question'], data['context'], \
                    data['same_as_question_word'], data['ans_sentence'] ,\
                    data['ans_start'], data['ans_end']

    return reader


if __name__ == "__main__":
    from paddle_train import choose_samples, load_config

    train_list, dev_list = choose_samples("featurized")

    for i, item in enumerate(train_reader(train_list)()):
        print item
        if i > 5: break

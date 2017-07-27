"""
A PaddlePaddle implementation of a question answering model.
"""
from __future__ import print_function

import json
import random
import collections
from collections import namedtuple
import pdb

import paddle.v2 as paddle
from paddle.v2.layer import parse_network

__all__ = ["build_model"]

UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
EMBEDDING_DIM = 300

Config = namedtuple("Config", [
    "question_layers",
    "document_layers",
    "layer_size",
    "embedding_dropout",
    "hidden_dropout",
    "learning_rate",
    "anneal_every",
    "anneal_rate",
    "epochs",
    "param_save_filename_format",
    "vocab_size",
    "data_dir",
    "batch_size",
])


def load_config(path):
    """
    Load the JSON config file from a file.
    """
    with open(path, "r") as handle:
        return Config(**json.load(handle))


def embedding_input(name, vocab_size, drop_rate=0.):
    """
    Create an embedding input to the network.

    Embeddings are static Glove vectors.
    """
    data = paddle.layer.data(
        name=name, type=paddle.data_type.integer_value_sequence(vocab_size))
    embeddings = paddle.layer.embedding(
        input=data,
        size=EMBEDDING_DIM,
        param_attr=paddle.attr.Param(name="GloveVectors", is_static=True),
        layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=drop_rate), )
    return embeddings


def binary_output(name):
    """
    Create a binary output for the network.
    """
    data = paddle.layer.data(
        name=name, type=paddle.data_type.integer_value_sequence(2))
    return data


def binary_input(name):
    """
    Create a binary input for the network.
    """
    data = paddle.layer.data(
        name=name, type=paddle.data_type.dense_vector_sequence(1))
    return data


def make_gru_step(size, static_input, prefix):
    """Make a gru step for recurrent_group"""
    boot = paddle.layer.fc(
        size=size,
        act=paddle.activation.Tanh(),
        bias_attr=False,
        input=static_input)

    def gru_with_static_input(current_input, static_input):
        mem = paddle.layer.memory(
            name='gru_decoder' + prefix, size=size, boot_layer=boot)
        gru_inputs = paddle.layer.fc(
            act=paddle.activation.Linear(),
            size=size * 3,
            bias_attr=False,
            input=[static_input, current_input])

        # without prefix, there may be problem when this function
        # is called more than one time,
        # because every layer in paddle have a unique name
        gru_step = paddle.layer.gru_step(
            name='gru_decoder' + prefix,
            input=gru_inputs,
            output_mem=mem,
            size=size)
        return gru_step

    return gru_with_static_input


def bidirectional_lstm(inputs, size, depth, prefix=""):
    """
    Run a bidirectional LSTM on the inputs.
    """
    if not isinstance(inputs, collections.Sequence):
        inputs = [inputs]

    lstm_last = []
    for dirt in ["fwd", "bwd"]:
        for i in range(depth):
            input_proj = paddle.layer.mixed(
                name="%s_in_proj_%0d_%s__" % (prefix, i, dirt),
                size=size * 4,
                bias_attr=paddle.attr.Param(initial_std=0.),
                input=[paddle.layer.full_matrix_projection(lstm)] if i else [
                    paddle.layer.full_matrix_projection(in_layer)
                    for in_layer in inputs
                ])
            lstm = paddle.layer.lstmemory(
                input=input_proj,
                bias_attr=paddle.attr.Param(initial_std=0.),
                param_attr=paddle.attr.Param(initial_std=5e-4),
                reverse=(dirt == "bwd"))
        lstm_last.append(lstm)

    final_states = paddle.layer.concat(input=[
        paddle.layer.last_seq(input=lstm_last[0]),
        paddle.layer.first_seq(input=lstm_last[1]),
    ])
    return final_states, paddle.layer.concat(input=lstm_last)


def build_document_embeddings(config, documents, same_as_question,
                              question_vector):
    """
    Build the document word embeddings.
    """
    hidden = paddle.layer.concat(input=[
        documents,
        same_as_question,
    ])

    # Half the question embedding is the final states of the LSTMs.
    question_expanded = paddle.layer.expand(
        input=question_vector, expand_as=documents)
    _, hidden = bidirectional_lstm([hidden, question_expanded],
                                   config.layer_size, config.document_layers,
                                   "__document__")

    return hidden


def build_question_vector(config, questions):
    """
    Build the question vector.
    """

    final, lstm_hidden = bidirectional_lstm(
        questions, config.layer_size, config.question_layers, "__question__")

    # The other half is created by doing an affine transform to generate
    # candidate embeddings, doing a second affine transform followed by a
    # sequence softmax to generate weights for the embeddings, and summing over
    # the weighted embeddings to generate the second half of the question
    # embedding.
    candidates = paddle.layer.fc(
        input=lstm_hidden, size=config.layer_size, act=None)
    weights = paddle.layer.fc(
        input=questions, size=1, act=paddle.activation.SequenceSoftmax())
    weighted = paddle.layer.scaling(input=candidates, weight=weights)
    embedding = paddle.layer.pooling(
        input=weighted, pooling_type=paddle.pooling.Sum())

    return paddle.layer.concat(input=[final, embedding])


def pick_word(config, word_embeddings):
    """
    For each word, predict a one or a zero indicating whether it is the chosen
    word.

    This is done with a two-class classification.
    """
    hidden = paddle.layer.dropout(
        input=word_embeddings, dropout_rate=config.hidden_dropout)
    predictions = paddle.layer.fc(
        input=hidden, size=2, act=paddle.activation.Softmax())
    return predictions


def build_classification_loss(predictions, classes):
    """
    Build a classification loss given predictions and desired outputs.
    """
    return paddle.layer.cross_entropy_cost(input=predictions, label=classes)


def build_model(config):
    """
    Build the PaddlePaddle model for a configuration.
    """
    questions = embedding_input("Questions", config.vocab_size,
                                config.embedding_dropout)
    documents = embedding_input("Documents", config.vocab_size,
                                config.embedding_dropout)

    same_as_question = binary_input("SameAsQuestion")

    correct_sentence = binary_output("CorrectSentence")
    correct_start_word = binary_output("CorrectStartWord")
    correct_end_word = binary_output("CorrectEndWord")

    # here the question vector is not a sequence
    question_vector = build_question_vector(config, questions)

    document_embeddings = build_document_embeddings(
        config, documents, same_as_question, question_vector)
    sentence_pred = pick_word(config, document_embeddings)
    start_word_pred = pick_word(config, document_embeddings)
    end_word_pred = pick_word(config, document_embeddings)

    losses = [
        build_classification_loss(sentence_pred, correct_sentence),
        build_classification_loss(start_word_pred, correct_start_word),
        build_classification_loss(end_word_pred, correct_end_word),
    ]
    return losses


if __name__ == "__main__":
    conf = load_config("paddle-config.json")
    costs = build_model(conf)
    print(parse_network(costs))

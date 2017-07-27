"""
A PaddlePaddle implementation of a question answering model.
"""
from __future__ import print_function

from collections import namedtuple
import glob
import gzip
import json
import os
import random
import sys

import click
import numpy as np

import paddle.v2 as paddle

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


def embedding_input(name, vocab_size):
    """
    Create an embedding input to the network.

    Embeddings are static Glove vectors.
    """
    data = paddle.layer.data(
        name=name,
        type=paddle.data_type.integer_value_sequence(vocab_size))
    embeddings = paddle.layer.embedding(
        input=data,
        size=EMBEDDING_DIM,
        param_attr=paddle.attr.Param(name="GloveVectors", is_static=True))
    return embeddings


def binary_output(name):
    """
    Create a binary output for the network.
    """
    data = paddle.layer.data(
        name=name,
        type=paddle.data_type.integer_value_sequence(2))
    return data


def binary_input(name):
    """
    Create a binary input for the network.
    """
    data = paddle.layer.data(
        name=name,
        type=paddle.data_type.dense_vector_sequence(1))
    return data

import random
def make_gru_step(size, static_input):
    """Make a gru step for recurrent_group"""
    boot = paddle.layer.fc(
      size=size,
      act=paddle.activation.Tanh(),
      bias_attr=False,
      input=static_input)

    def gru_with_static_input(current_input, static_input):
         mem = paddle.layer.memory(name='memory-'+str(random.randint(0, 1000000)), size=size, boot_layer=boot)
         gru_inputs = paddle.layer.fc(
             act=paddle.activation.Linear(),
             size=size * 3, bias_attr=False,
             input=[static_input, current_input])
         gru_step = paddle.layer.gru_step(
             name='gru_decoder',
             input=gru_inputs,
             output_mem=mem,
             size=size)
         return gru_step
    return gru_with_static_input


def lstm(inputs, size, layers, static=None):
    """
    Run a bidirectional LSTM on the inputs.
    """
    for i in range(layers):
	if static is not None:
	    inputs = [inputs, paddle.layer.StaticInput(static)]
            print("HELLO")
            fwd = paddle.layer.recurrent_group(input=inputs, step=make_gru_step(size, static), name="fwd")
            print("HELLO")
            bwd = paddle.layer.recurrent_group(input=inputs, step=make_gru_step(size, static), reverse=True, name="bwd")
            print("HELLO")
        else:
            fwd = paddle.networks.simple_lstm(input=inputs, size=size)
            bwd = paddle.networks.simple_lstm(input=inputs, size=size, reverse=True)
        inputs = paddle.layer.concat(input=[fwd, bwd])

    final_states = paddle.layer.concat(input=[
        paddle.layer.last_seq(input=fwd),
        paddle.layer.first_seq(input=bwd),
    ])
    return final_states, inputs


def build_document_embeddings(config, documents, same_as_question,
                              question_vector):
    """
    Build the document word embeddings.
    """
    hidden = paddle.layer.dropout(
        input=documents,
        dropout_rate=config.embedding_dropout)
    hidden = paddle.layer.concat(input=[
        hidden,
        same_as_question,
    ])

    # Half the question embedding is the final states of the LSTMs.
    final, hidden = lstm(hidden, config.layer_size,
                         config.document_layers,
			 static=question_vector)

    return hidden


def build_question_vector(config, questions):
    """
    Build the question vector.
    """
    hidden = paddle.layer.dropout(
        input=questions,
        dropout_rate=config.embedding_dropout)

    # Half the question embedding is the final states of the LSTMs.
    final, hidden = lstm(hidden, config.layer_size,
                         config.question_layers)

    # The other half is created by doing an affine transform to generate
    # candidate embeddings, doing a second affine transform followed by a
    # sequence softmax to generate weights for the embeddings, and summing over
    # the weighted embeddings to generate the second half of the question
    # embedding.
    candidates = paddle.layer.fc(
        input=hidden, size=config.layer_size, act=None)
    weights = paddle.layer.fc(
        input=hidden, size=1, act=paddle.activation.SequenceSoftmax())
    weighted = paddle.layer.scaling(
        input=candidates,
        weight=weights)
    embedding = paddle.layer.pooling(
        input=weighted,
        pooling_type=paddle.pooling.Sum())

    return paddle.layer.concat(input=[final, embedding])


def pick_word(config, word_embeddings):
    """
    For each word, predict a one or a zero indicating whether it is the chosen
    word.

    This is done with a two-class classification.
    """
    hidden = paddle.layer.dropout(
        input=word_embeddings,
        dropout_rate=config.hidden_dropout)
    predictions = paddle.layer.fc(
        input=hidden, size=2, act=paddle.activation.Softmax())
    return predictions


def build_classification_loss(predictions, classes):
    """
    Build a classification loss given predictions and desired outputs.
    """
    return paddle.layer.cross_entropy_cost(
        input=predictions,
        label=classes)


def build_model(config):
    """
    Build the PaddlePaddle model for a configuration.
    """
    questions = embedding_input("Questions", config.vocab_size)
    documents = embedding_input("Documents", config.vocab_size)
    same_as_question = binary_input("SameAsQuestion")
    correct_sentence = binary_output("CorrectSentence")
    correct_start_word = binary_output("CorrectStartWord")
    correct_end_word = binary_output("CorrectEndWord")

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

    parameters = paddle.parameters.create(losses)

    optimizer = paddle.optimizer.Adam(
        learning_rate=config.learning_rate,
        learning_rate_schedule="discexp",
        learning_rate_decay_a=config.anneal_rate,
        learning_rate_decay_b=config.anneal_every * config.batch_size)

    trainer = paddle.trainer.SGD(
        cost=losses,
        parameters=parameters,
        update_equation=optimizer)

    return trainer, parameters


def choose_samples(path):
    """
    Load filenames for train, dev, and augmented samples.
    """
    if not os.path.exists(os.path.join(path, "train")):
        print("Non-existent directory as input path: {}".format(path),
              file=sys.stderr)
        sys.exit(1)

    # Get paths to all samples that we want to load.
    train_samples = glob.glob(os.path.join(path, "train", "*"))
    valid_samples = glob.glob(os.path.join(path, "dev", "*"))

    train_samples.sort()
    valid_samples.sort()

    random.shuffle(train_samples)
    random.shuffle(valid_samples)

    return train_samples, valid_samples


def make_reader(samples):
    """
    Load samples from filenames and yield them.
    """
    random.shuffle(samples)

    for sample_file in samples:
        with open(sample_file, "rt") as handle:
            sample = json.load(handle)

        indices = np.arange(len(sample["context"]))

	# The values must all be Python lists, not NumPy arrays.
        yield (sample["context"],
               sample["same_as_question_word"],
	       sample["question"],
               (indices == sample["ans_sentence"]).tolist(),
               (indices == sample["ans_start"]).tolist(),
               (indices == sample["ans_end"]).tolist())


def build_reader(config):
    """
    Build the data reader for this model.
    """
    train_samples, valid_samples = choose_samples(config.data_dir)

    train_reader = lambda: make_reader(train_samples)
    train_reader = paddle.batch(
        paddle.reader.shuffle(train_reader, buf_size=1000),
        batch_size=config.batch_size)

    test_reader = lambda: make_reader(train_samples)
    test_reader = paddle.batch(
        paddle.reader.shuffle(test_reader, buf_size=100),
        batch_size=config.batch_size)
    return train_reader, test_reader


def build_event_handler(config, parameters, trainer, test_reader):
    """
    Build the event handler for this model.
    """
    # End batch and end pass event handler
    def event_handler(event):
        """The event handler."""
        if isinstance(event, paddle.event.EndIteration):
            print("\nPass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            param_path = config.param_save_filename_format % event.pass_id
            with gzip.open(param_path, 'w') as handle:
                parameters.to_tar(handle)

            result = trainer.test(reader=test_reader)
            print("\nTest with Pass %d, %s" % (event.pass_id, result.metrics))

    return event_handler


@click.group()
def main():
    """
    Train and run QA models with PaddlePaddle.
    """
    pass


@main.command("train")
@click.argument("config")
def train(config):
    """
    Train and run QA models with PaddlePaddle.
    """
    paddle.init(use_gpu=True, trainer_count=1)

    conf = load_config(config)
    trainer, parameters = build_model(conf)
    train_reader, test_reader = build_reader(conf)
    handler = build_event_handler(conf, parameters, trainer, test_reader)

    trainer.train(reader=train_reader, num_passes=conf.epochs,
                  event_handler=handler)


if __name__ == "__main__":
    main()

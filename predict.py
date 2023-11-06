import collections
import os
import pickle
from absl import flags, logging
import modeling
import metrics
import optimization
import tokenization
import tensorflow as tf
import numpy as np

FLAGS = flags.FLAGS
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

flags.DEFINE_string(
    "data_dir", "./data",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "./chinese_bert/chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "ner", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "./chinese_bert/chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./output",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", "./chinese_bert/chinese_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("crf", False, "use crf")


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class PaddingInputExample(object):
    pass


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids  # label_ids should be list
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Read a BIO data!"""
        rf = open(input_file, 'r')
        lines = [];
        words = [];
        labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            words.append(word)
            labels.append(label)
        l = ' '.join([label for label in labels if len(label) > 0])
        w = ' '.join([word for word in words if len(word) > 0])
        lines.append((l, w))
        rf.close()
        return lines


class NerProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "my_test.txt")), "test"
        )

    def get_labels(self):
        return ["[PAD]", "O", "B-OBJ", "I-OBJ", "B-PRO", "I-PRO", "B-NUM", "I-NUM", "B-MOV", "I-MOV", "[CLS]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.output_dir + "/label2id.pkl", 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')  # 在这里进行切分
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist, labellist)):
        tokens.append(word)
        labels.append(label)
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 把单词转换成在vocab中的id
    mask = [1] * len(input_ids)
    # use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)  # label_map["[PAD]"]==0
        ntokens.append("[PAD]")

    # 形如: ntokens:["[CLS]","当","希","望"], mask=[1,1,1,1],input_ids=[101,256,257,412],label_ids=[8,1,1,1]
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )

    return feature, ntokens, label_ids


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                                                             mode)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        if is_training:
            batch_size = params["train_batch_size"]
        else:
            batch_size = params["predict_batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.reduce_mean(-log_likelihood)
    return loss, transition


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])  # 计算loss的时候是不能考虑pad位置的，所以一定要乘以mask
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.nn.softmax(logits, axis=-1)  # (batch_size*seq_length,num_labels)
    predict = tf.argmax(probabilities, axis=-1)  # (batch_size*seq_length,)
    return loss, predict


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids
    )

    output_layer = model.get_sequence_output()
    # output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
    logits = hidden2tag(output_layer, num_labels)
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
    if FLAGS.crf:
        mask2len = tf.reduce_sum(mask, axis=1)
        loss, trans = crf_loss(logits, labels, mask, num_labels, mask2len)
        predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        assert len(predict.get_shape().as_list()) == 2  # because logits's rank is 3
        return (loss, logits, predict)

    else:
        loss, predict = softmax_layer(logits, labels, num_labels, mask)
        assert len(logits.get_shape().as_list()) == 3 and len(predict.get_shape().as_list()) == 1
        return (
        loss, logits, tf.reshape(predict, [-1, FLAGS.max_seq_length]))  # predict.shape==(batch_size,max_seq_length)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu=False):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                      mask, segment_ids, label_ids, num_labels)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predicts)  # predicts.shape==(batch_size,max_seq_length)
        return output_spec

    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    if token != "[PAD]" and token != "[CLS]":
        line = "{}\t{}\n".format(token, predict)
        wf.write(line)


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    # batch_tokens是所有token的集合
    # batch_labels是所有token的label_id的集合
    # result.shape==(len(pred_examples),max_seq_length)
    with open(output_predict_file, 'w') as wf:
        predictions = []
        for m, pred in enumerate(result):
            predictions.extend(pred)  # 相当于把predict所有的句子用一个列表表示出来
        for i, prediction in enumerate(predictions):
            # prediction就是一个id,id2label[prediction]就是预测的label
            _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)


def main(_):
    logging.set_verbosity(logging.INFO)
    processors = {"ner": NerProcessor}
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    run_config = tf.estimator.RunConfig().replace(model_dir=FLAGS.output_dir,
                                                  log_step_count_steps=100,
                                                  keep_checkpoint_max=5,
                                                  save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)
    params = {"train_batch_size": FLAGS.train_batch_size, "eval_batch_size": FLAGS.eval_batch_size,
              "predict_batch_size": FLAGS.predict_batch_size}
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params)

    if FLAGS.do_predict:
        with open(FLAGS.output_dir + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        batch_tokens, batch_labels = filed_based_convert_examples_to_features(predict_examples, label_list,
                                                                              FLAGS.max_seq_length, tokenizer,
                                                                              predict_file)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)  # result就是predictions,shape==(batch_size,max_seq_length)
        output_predict_file = os.path.join(FLAGS.output_dir, "my_test.txt")
        Writer(output_predict_file, result, batch_tokens, batch_labels, id2label)


if __name__ == "__main__":
    tf.app.run()

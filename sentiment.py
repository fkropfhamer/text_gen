import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras 
from official.nlp import optimization



def main():
    print("main")

    tf.get_logger().setLevel('ERROR')

    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 55

    num_classes = 2

    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='./data',
                                  cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    train_dir = os.path.join(dataset_dir, 'train')


    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'data/aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        'data/aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        'data/aclImdb/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)



    classfier = build_classfier_model()

    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_trains_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_trains_steps)
    
    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_trains_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw'
    )

    classfier.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = classfier.fit(x=train_ds, validation_data=val_ds, epochs=epochs)

    loss, accuracy = classfier.evaluate(test_ds)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    

def build_classfier_model():
    bert_preprocess_model_link = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_preprocess_model = hub.KerasLayer(bert_preprocess_model_link, name="preprocess")

    bert_model_link = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
    bert_model = hub.KerasLayer(bert_model_link, trainable=True, name='BERT_encoder')



    text_input = keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = bert_preprocess_model(text_input)
    outputs = bert_model(encoder_inputs)
    net = outputs['pooled_output']
    net = keras.layers.Dropout(0.1)(net)
    net = keras.layers.Dense(1, activation=None, name='classfier')(net)
    return keras.Model(text_input, net) 




if __name__ == '__main__':
    main()

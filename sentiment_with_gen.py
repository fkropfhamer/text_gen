import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras 
from official.nlp import optimization
import numpy as np
from sklearn.model_selection import train_test_split
import collections



def main():
    print("main")

    tf.get_logger().setLevel('ERROR')

    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 55

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





    xs = np.array([])
    ys = np.array([])

    for x, y in raw_train_ds:
        #print(x)
        #print(y)
        xs = np.concatenate([xs, x])
        ys = np.concatenate([ys, y])
        

    print(xs.shape)


    class_names = raw_train_ds.class_names


    use_generated = False
    if use_generated:
        raw_generated = keras.utils.text_dataset_from_directory(
        'data/imdb_generated',
        seed=seed
        )

        generated_xs = np.array([])
        generated_ys = np.array([])

        for x, y in raw_generated:
            generated_xs = np.concatenate([generated_xs, x])
            generated_ys = np.concatenate([generated_ys, y])

        x_train = np.concatenate([x_train, generated_xs])
        y_train = np.concatenate([y_train, generated_ys])

        print("Num of duplicates ", len([item for item, count in collections.Counter(list(generated_xs)).items() if count > 1]))



    classfier = build_classfier_model()

    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    epochs = 10
    steps_per_epoch = len(x_train)
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

    history = classfier.fit(x_train, y_train, epochs=epochs, validation_data=val_ds, batch_size=batch_size)

    loss, accuracy = classfier.evaluate(x_test, y_test, batch_size=batch_size)

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

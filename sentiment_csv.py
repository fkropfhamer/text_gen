import tensorflow as tf
import os
import shutil
import numpy as np
import csv

def main():
    name = "imdb"

    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='./data',
                                  cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    train_dir = os.path.join(dataset_dir, 'train')


    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    raw_ds = tf.keras.utils.text_dataset_from_directory(
        'data/aclImdb/train',
    )

    class_names = raw_ds.class_names

    classes = {}

    for class_name in class_names:
        classes[class_name] = np.array([])


    print(class_names)

    for x, y in raw_ds:
        class_name = class_names[y[0]]
        xs = classes[class_name]

        xs = np.concatenate([xs, x])

        classes[class_name] = xs

    for class_name, xs in classes.items():
        with open(f"./data/{name}_{class_name}.csv", "w", newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([])
            #file.write(','.join(map(lambda x: x.decode('utf-8'), xs)))

            for x in xs:
                csv_writer.writerow([x.decode('utf-8')])


if __name__ == '__main__':
    main()

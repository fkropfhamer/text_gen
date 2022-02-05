import os
from tensorflow import keras
from pathlib import Path
import json
import numpy as np  
import csv

def main():
    #create_dataset()
    create_csv()

def create_dataset():
    url = "https://figshare.com/ndownloader/files/6453753"


    dataset = keras.utils.get_file('pheme.tar.bz2', url,
                                  untar=True, cache_dir='./data',
                                  cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'pheme-rnr-dataset')

    only_text_dataset_dir = "./data/pheme_simple"

    for topic_dir in os.scandir(dataset_dir):
        if topic_dir.is_dir():
            non_rumor_dir = os.path.join(topic_dir, 'non-rumours')
            
            for non_rumor_conversation_dir in os.scandir(non_rumor_dir):
                source_tweet_dir = os.path.join(non_rumor_conversation_dir, 'source-tweet')
                for non_rumor_source_tweet_json in Path(source_tweet_dir).glob('*.json'):
                    with open(non_rumor_source_tweet_json, 'r') as file:
                        data = json.load(file)
                        out_filename = only_text_dataset_dir + f"/non_rumor/{non_rumor_source_tweet_json.stem}.txt"
                        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                        with open(out_filename, 'w+') as out_file:
                            out_file.write(data['text'])

            rumor_dir = os.path.join(topic_dir, 'rumours')

            for rumor_conversation_dir in os.scandir(rumor_dir):
                source_tweet_dir = os.path.join(rumor_conversation_dir, 'source-tweet')
                for rumor_source_tweet_json in Path(source_tweet_dir).glob('*.json'):
                    with open(rumor_source_tweet_json, 'r') as file:
                        data = json.load(file)
                        out_filename = only_text_dataset_dir + f"/rumor/{rumor_source_tweet_json.stem}.txt"
                        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                        with open(out_filename, 'w+') as out_file:
                            out_file.write(data['text'])

def create_csv():
    raw_ds = keras.utils.text_dataset_from_directory(
        'data/pheme_simple',
        batch_size=1
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
        with open(f"./data/{class_name}.csv", "w", newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([])
            #file.write(','.join(map(lambda x: x.decode('utf-8'), xs)))

            for x in xs:
                csv_writer.writerow([x.decode('utf-8')])

        
        continue

        with open(f"./data/{class_name}.txt", "w") as file:
            for x in xs:
                file.write(x.decode('utf-8')+"\n\n")





if __name__ == '__main__':
    main()
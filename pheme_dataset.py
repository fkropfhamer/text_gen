import os
from tensorflow import keras
from pathlib import Path
import json
import numpy as np  
import csv
import datetime

def main():
    create_dataset()
    create_csv('data/pheme_simple', 'simple')
    create_csv('data/pheme_with_reactions', 'reactions')

def create_dataset():
    url = "https://figshare.com/ndownloader/files/6453753"


    dataset = keras.utils.get_file('pheme.tar.bz2', url,
                                  untar=True, cache_dir='./data',
                                  cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'pheme-rnr-dataset')

    only_text_dataset_dir = "./data/pheme_simple"
    with_reactions_dir = "./data/pheme_with_reactions"

    for topic_dir in os.scandir(dataset_dir):
        if topic_dir.is_dir():
            non_rumor_dir = os.path.join(topic_dir, 'non-rumours')
            
            for non_rumor_conversation_dir in os.scandir(non_rumor_dir):
                source_tweet = None
                reactions_jsons = []

                source_tweet_dir = os.path.join(non_rumor_conversation_dir, 'source-tweet')
                for non_rumor_source_tweet_json in Path(source_tweet_dir).glob('*.json'):
                    with open(non_rumor_source_tweet_json, 'r') as file:
                        data = json.load(file)
                        source_tweet = data
                        out_filename = only_text_dataset_dir + f"/non_rumor/{non_rumor_source_tweet_json.stem}.txt"
                        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                        with open(out_filename, 'w+') as out_file:
                            out_file.write(data['text'])

                reactions_tweet_dir = os.path.join(non_rumor_conversation_dir, 'reactions')
                for non_rumor_reactions_json in Path(reactions_tweet_dir).glob('*.json'):
                    
                    with open(non_rumor_reactions_json) as file:
                        data = json.load(file)
                        created_date = datetime.datetime.strptime(data['created_at'], '%a %b %d %H:%M:%S %z %Y')
                        data['created_datetime'] = created_date

                        reactions_jsons.append(data)
                

                sorted_reaction = sorted(reactions_jsons, key=lambda x: x['created_datetime'])
                conversation_id = os.path.basename(non_rumor_conversation_dir)
                out_with_reaction_filename = with_reactions_dir + f"/non_rumor/{conversation_id}.txt"
                os.makedirs(os.path.dirname(out_with_reaction_filename), exist_ok=True)
                with open(out_with_reaction_filename, 'w+') as file:
                    texts = [source_tweet['text']]
                    texts += list(map(lambda x: x['text'], sorted_reaction))
                    file.write(' ### '.join(texts))
                

            rumor_dir = os.path.join(topic_dir, 'rumours')

            for rumor_conversation_dir in os.scandir(rumor_dir):
                source_tweet = None
                reactions_jsons = []

                source_tweet_dir = os.path.join(rumor_conversation_dir, 'source-tweet')
                for rumor_source_tweet_json in Path(source_tweet_dir).glob('*.json'):
                    with open(rumor_source_tweet_json, 'r') as file:
                        data = json.load(file)
                        source_tweet = data
                        out_filename = only_text_dataset_dir + f"/rumor/{rumor_source_tweet_json.stem}.txt"
                        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                        with open(out_filename, 'w+') as out_file:
                            out_file.write(data['text'])


                reactions_tweet_dir = os.path.join(rumor_conversation_dir, 'reactions')
                for rumor_reactions_json in Path(reactions_tweet_dir).glob('*.json'):
                    
                    with open(rumor_reactions_json) as file:
                        data = json.load(file)
                        created_date = datetime.datetime.strptime(data['created_at'], '%a %b %d %H:%M:%S %z %Y')
                        data['created_datetime'] = created_date

                        reactions_jsons.append(data)
                

                sorted_reaction = sorted(reactions_jsons, key=lambda x: x['created_datetime'])
                conversation_id = os.path.basename(rumor_conversation_dir)
                out_with_reaction_filename = with_reactions_dir + f"/rumor/{conversation_id}.txt"
                os.makedirs(os.path.dirname(out_with_reaction_filename), exist_ok=True)
                with open(out_with_reaction_filename, 'w+') as file:
                    texts = [source_tweet['text']]
                    texts += list(map(lambda x: x['text'], sorted_reaction))
                    file.write('###'.join(texts))

def create_csv(data_dir, name):
    raw_ds = keras.utils.text_dataset_from_directory(
        data_dir,
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
        with open(f"./data/{name}_{class_name}.csv", "w", newline='') as file:
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
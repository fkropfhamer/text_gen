import gpt_2_simple as gpt2
import os
import uuid
import collections

def main():
    #train("pheme_non_rumor_simple", "./data/simple_non_rumor.csv")
    #train("pheme_rumor_simple", "./data/simple_rumor.csv")   
    #train("pheme_non_rumor_reactions", "./data/reactions_non_rumor.csv", steps=3000)
    #train("pheme_rumor_reactions", "./data/reactions_rumor.csv", steps=3000) 

    #generate("pheme_non_rumor_simple", "./data/pheme_simple_generated", 'non_rumor')
    #generate("pheme_rumor_simple", "./data/pheme_simple_generated", 'rumor')

    #generate("pheme_non_rumor_reactions", "./data/reactions_generated", 'non_rumor')
    #generate("pheme_rumor_reactions", "./data/reactions_generated", 'rumor') 


    #train("imdb_neg", "./data/imdb_neg.csv")
    #train("imdb_pos", "./data/imdb_pos.csv")

    #generate("imdb_neg", "./data/imdb_generated", 'neg', num_samples=5000)
    #generate("imdb_pos", "./data/imdb_generated", 'pos', num_samples=5000)

    #train("pheme_split_non_rumor_simple", "./data/pheme_split_non_rumor.csv")
    #train("pheme_split_rumor_simple", "./data/pheme_split_rumor.csv")

    #generate("pheme_split_non_rumor_simple", "./data/pheme_split_simple_generated", "non_rumor", num_samples=1500)
    #generate("pheme_split_rumor_simple", "./data/pheme_split_simple_generated", "rumor", num_samples=1500)

    train("pheme_split_non_rumor_simple2", "./data/pheme_split_non_rumor.csv", steps=200)
    train("pheme_split_rumor_simple2", "./data/pheme_split_rumor.csv", steps=200)

    generate("pheme_split_non_rumor_simple2", "./data/pheme_split_simple_generated2", "non_rumor", num_samples=1500)
    generate("pheme_split_rumor_simple2", "./data/pheme_split_simple_generated2", "rumor", num_samples=1500)
    
def generate(run_name, dire, label, num_samples=3000):
    sess = gpt2.start_tf_sess()

    load_model(sess, run_name)

    ftexts = set()
    
    max_gen = num_samples

    while len(ftexts) < max_gen:
        print(f"{len(ftexts)} / {max_gen}")

        texts = gpt2.generate(sess, return_as_list=True,  nsamples=50)

        for text in texts:
            if "<|startoftext|>" in text:
                s = text.split("<|startoftext|>")
                #print(len(s))
                s.pop()
                s.pop(0)
                s = map(lambda x: x.split("<|endoftext|>")[0], s)

                #print(list(s))
                ftexts = set.union(ftexts, set(s))
            else:
                ftexts = set.union(ftexts, set(text))


    print("saving")

    only_text_dataset_dir = dire
    for text in ftexts:
        out_filename = only_text_dataset_dir + f"/{label}/{uuid.uuid4().hex}.txt"
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        with open(out_filename, "w+") as file:
            file.write(text)

    #print(texts)

    print("Duplicates ", len([item for item, count in collections.Counter(ftexts).items() if count > 1]))

    #print(len(texts))

    gpt2.reset_session(sess)

def train(run_name, file_name, steps=1000):
    sess = gpt2.start_tf_sess()

    finetune(sess, run_name, file_name, steps=steps)

    gpt2.reset_session(sess)


def download():
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
	    print(f"Downloading {model_name} model...")
	    gpt2.download_gpt2(model_name=model_name) 

def load_model(sess, run_name):
    gpt2.load_gpt2(sess, run_name=run_name)

def finetune(sess, run_name, file_name, steps=1000, model_name="124M"):
    gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=steps,
              run_name=run_name
              )   # steps is max number of training steps

if __name__ == '__main__':
    main()

import os
import sys

def write_training_notes(path, write_dict, append=False):
    if not os.path.exists(path):
        os.makedirs(path)

    path= str(path)+ "/training_notes.txt"

    if (append):
        with open(path, 'a') as file:
                for k,v in write_dict.items():
                    print("k: "+ str(k)+ " v: ", str(v))
                    file.write(str(k) +": " +str(v)+"\n")
    else:
        with open(path, 'w+') as file:
                for k,v in write_dict.items():
                    file.write(str(k) +": " +str(v)+"\n")
                print("TRAINING LOSS: \n\n")

def write_loss_history(path, list):
    if not os.path.exists(path):
        os.makedirs(path)

    path= str(path)+ "/loss_history.txt"
    with open(path, 'w+') as file:
            for i in list:
                    file.write(str(i)+"\n")

def write_model_json(path, model_json):
    if not os.path.exists(path):
        os.makedirs(path)

    model_path = str(path) + "/model.json"
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    # vae.save_weights("model.h5")
    print("Saved model to disk")


def write_model_architecture(directory, vae, name="model_architecture"):
    #first of all prints summary
    vae.summary()
    summary_path = directory + '/'+str(name)+'.txt'

    if not os.path.exists(directory):
        os.makedirs(directory)

    orig_stdout = sys.stdout

    with open(summary_path, 'a+') as f:
        sys.stdout = f
        print(vae.summary())
        sys.stdout = orig_stdout
        f.close()

    with open(summary_path, 'a+') as f:
        for l in vae.layers:
            f.write(str(l.name)+ "  trainable? "+str(l.trainable)+"\n")
    f.close()
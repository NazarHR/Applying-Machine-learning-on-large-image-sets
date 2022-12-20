import utility
import process
from filter_selector import select_filters, prepare_for_scores
from sklearn.neural_network import MLPClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

def train(model,Xfull,Yfull,name,kernels,lables):
    model.fit(Xfull,Yfull)
    with open("models/"+name+".model", 'wb') as file:
        pickle.dump(model, file)

def main_train():
    sample_files = utility.open_filedialog('вибір зображень-прикладів')
    other_files=utility.open_filedialog('вибір зображень, які не відповідають критеріям пошуку')
    images_df=utility.read_images(sample_files,other_files)
    kernels,lables=utility.generate_filters()
    scoresX, scoresY=prepare_for_scores(images_df,kernels,lables)
    selectedK,selectedL=select_filters(scoresX,scoresY,kernels,lables)
    training_class=sample_files.split("/")[-1]

    with open("models/"+training_class+".kernels", 'wb') as file:
        pickle.dump(selectedK, file)
    with open("models/"+training_class+".lables", 'wb') as file:
        pickle.dump(selectedL, file)
    Xtrain,Ytrain=process.form_data(images_df,selectedK,selectedL)
    print(Xtrain.shape)
    #model=MLPClassifier(activation='relu',solver='adam',hidden_layer_sizes=(256,),random_state=1, max_iter=300,verbose=True)
    model=MLPClassifier(activation='relu',solver='adam',hidden_layer_sizes=(512,),random_state=1, max_iter=300,verbose=True)
    train(model,Xtrain,Ytrain,training_class,selectedK,selectedL)

if __name__ == "__main__":
    main_train()

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import MobileNetV2, InceptionV3, ResNet50  # Vous pouvez ajouter d'autres modèles ici

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data(dataset_name):
    if dataset_name == 'CIFAR10':
        return tf.keras.datasets.cifar10.load_data()
    else:
        return None

# Fonction pour charger le modèle de base selon le choix de l'utilisateur
def get_base_model(model_name, input_shape):
    if model_name == 'MobileNetV2':
        return MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet',classifier_activation='softmax')
    elif model_name == 'InceptionV3':
        return InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_name == 'ResNet50':
        return ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        raise ValueError("Modèle non supporté")

# Liste des modèles disponibles pour le transfert d'apprentissage
transfer_learning_models = {
    'MobileNetV2': 'MobileNetV2',
    'InceptionV3': 'InceptionV3',  # Exemple d'ajout d'autres modèles
    'ResNet50': 'ResNet50'
}
simple_cnn={'simple_cnn'}

def plot_samples(images, labels, classes=None):
    """Affiche un échantillon d'images avec leurs étiquettes."""
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)   
        if images[i].shape[-1] == 1:
            plt.imshow(images[i].squeeze(), cmap=plt.cm.binary)
        else:
            plt.imshow(images[i])
            
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']   
        
        plt.xlabel(class_names[labels[i][0]])    
    st.pyplot()

# Interface utilisateur
if 'x_train' not in st.session_state:
    st.session_state['x_train'] = None
    st.session_state['y_train'] = None
    
# Interface utilisateur
if 'x_test' not in st.session_state:
    st.session_state['x_test'] = None
    st.session_state['y_test'] = None    

# Interface utilisateur
if 'x_val' not in st.session_state:
    st.session_state['x_val'] = None
    st.session_state['y_val'] = None   
    
st.title('Projet: Classification des images en utilisant l''pprentissage par transfert')

dataset_name = st.selectbox('Choisissez une base de données', ('CIFAR10',''))

data = load_data(dataset_name)

if data:
    if st.button('Charger la base de donnée'):
        # Message après le chargement complet
        st.success('Chargement est fini avec succès!')

split_ratio = st.slider('Choisissez le taux de split train/Valid datasets', 0.1, 0.9, 0.2, 0.1)


if data:    
    if st.button('Diviser la base de données (Train/Test/Valid)'):
        (x_train, y_train), (x_test, y_test) = data
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split_ratio, random_state=42)
        
        st.session_state['x_train'] = x_train/255.0
        st.session_state['y_train'] = y_train
        
        st.session_state['x_test'] = x_test/255.0
        st.session_state['y_test'] = y_test
        
        st.session_state['x_val'] = x_val
        st.session_state['y_val'] = y_val
        
        st.write(f"Taille des données d'entraînement: {x_train.shape[0]}")
        st.write(f"Taille des données de validation: {x_val.shape[0]}")
        st.write(f"Taille des données de test: {x_test.shape[0]}")
        
        
    if st.button('Afficher des échantillons'):
        plot_samples(st.session_state['x_train'][:25], st.session_state['y_train'][:25])
        
 

#model_choice = st.selectbox('Choisissez une architecture de modèle', list(model_choices.keys()))
option = list(transfer_learning_models.keys()) + ['simple_cnn']

# Sélection du modèle pour le transfert d'apprentissage
transfer_model_choice = st.selectbox('Choisissez une architecture pour le transfert d\'apprentissage ou Simple CNN', option)

# Création du modèle selon le choix de l'utilisateur
def create_model(model_type, input_shape, num_classes):
    if model_type in transfer_learning_models:
        base_model = get_base_model(transfer_model_choice, input_shape)
        base_model.trainable = False  # Congélation des couches du modèle de base
        model = Sequential([

            base_model,
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='relu')
        ])
    elif model_type == 'simple_cnn':
        model = Sequential([

            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='relu')
        ])
    return model

if dataset_name == 'CIFAR10':
    input_shape = (32, 32, 3)  # CIFAR-10 has color images of 32x32
    num_classes = 10  # CIFAR-10 has 10 classes
    model = create_model(transfer_model_choice, input_shape, num_classes)
else:
    st.error("Selected dataset is not supported.")
    
metrics = ['accuracy']
if st.button('Lancer l’entraînement'):
    if metrics:
        
        print(model.summary())
        
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=metrics)
        
        cnn = model.fit(st.session_state['x_train'],
                        st.session_state['y_train'], 
                        validation_data=(st.session_state['x_val'],
                                         st.session_state['y_val']),
                        epochs=20)
        
        print(model.summary())
        
        #évaluation
        
        st.write("Entraînement terminé avec succès.")
        plt.plot(cnn.history['accuracy'], label='accuracy')
        plt.plot(cnn.history['val_accuracy'], label = 'val_accuracy')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.ylim([0.1,1.0])
        plt.legend(loc='lower right')
        
        st.pyplot()
        loss, accuracy = model.evaluate(st.session_state['x_test'], st.session_state['y_test'], verbose=2)

        st.write(f"Perte (Loss) sur l'ensemble de test: {loss}")
        st.write(f"Précision (Accuracy) sur l'ensemble de test: {accuracy}")
        
    else:
        st.error("Veuillez choisir au moins une métrique.")
        

    
    
    
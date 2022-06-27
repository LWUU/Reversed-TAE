from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model, to_categorical
import tensorflow_addons as tfa

import sys
import numpy as np
import matplotlib.pyplot as plt

import Triplet_loss as losses

def CLR(training_num, batch_size):
    return tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-5,
                maximal_learning_rate=5e-3,
                scale_fn=lambda x: 1/(2.**(x-1)),
                step_size=2 * training_num // batch_size
            )

class TAE:
    # def __init__(self, base_model):        
                    
    def base_model(self):
        input_img = Input(shape=(784,))
        x = Dense(256, activation='relu')(input_img)
        embedding = Dense(self.encoding_dim, activation='relu')(x)
        x = Dense(256, activation='relu')(embedding)
        decoded = Dense(784, activation='sigmoid')(x)

        model = Model(input_img, [embedding, decoded])

        return model

    def build(self, input_shape, encoding_dim, weight_triplet, num_class, learning_rate=0.0001):
        self.encoding_dim = encoding_dim
        input_images = Input(shape=input_shape, name='input_image')
        input_labels = Input(shape=(1,), name='input_label')

        base_network = self.base_model()

        # output of network -> embeddings
        [embeddings, decoded] = base_network([input_images])

        # concatenating the labels + embeddings
        labels_plus_embeddings = Concatenate(axis=1)([input_labels, embeddings])

        # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
        self.model = Model(inputs=[input_images, input_labels], outputs=[labels_plus_embeddings, decoded])
        
        opt = Adam(learning_rate=learning_rate)  # choose optimiser. RMS is good too!

        # alpha value is used in my another paper. Disabled for now, so we use the standard semi-hard triplet loss
        # Triplet loss is also modified. See Triplet_loss.py L159 - L163 for details. 
        alpha_value = 0.0 
        margin = 0.1
        triplet_loss = losses.triplet_semihard_loss(alpha_value, margin, num_class)

        self.model.compile(loss=[triplet_loss, 'binary_crossentropy'], loss_weights=[weight_triplet, 1], optimizer=opt)

    def train(self, x_train, y_train, x_test, y_test, batch_size=256, epoch=500, verbose=2):
        dummy_gt_train = np.zeros((len(x_train), self.encoding_dim + 1))
        dummy_gt_val = np.zeros((len(x_test), self.encoding_dim + 1))

        history = self.model.fit(
            x=[x_train, y_train],
            y=[dummy_gt_train, x_train],
            batch_size=batch_size,
            epochs=epoch,
            shuffle=True,
            validation_data=([x_test, y_test], [dummy_gt_val, x_test]),
            verbose=verbose)
    
        return history

    def predict(self, x_test):
        dummy_gt_val = np.zeros((len(x_test), self.encoding_dim + 1))
        [embeddings, decoded_imgs] = self.model.predict([x_test, dummy_gt_val])
        return embeddings, decoded_imgs

    def visualize(x_test, decoded_imgs, saving_dir):
        n = 10  # How many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(saving_dir+'Reconstructed_images_WTriplet{}.png'.format(weight_triplet))

class Target_Classifier:
    def __init__(self, model=None):
        self.model = model

    def build(self, input_shape, num_class):
        self.num_class = num_class

        input_img = Input(shape=input_shape)
        x = Dense(256, activation='relu')(input_img)
        output = Dense(self.num_class, activation='softmax')(x)

        self.model = Model(input_img, output)
        opt = Adam(learning_rate=0.0001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    def train(self, x_train, y_train, x_test, y_test, batch_size, epochs, verbose):
        history = self.model.fit(
            x=x_train,
            y=to_categorical(y_train, num_classes=self.num_class),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=epochs,
            validation_data=(x_test, to_categorical(y_test, num_classes=self.num_class)),
            verbose=verbose)
        return history

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, to_categorical(y_test), verbose=2)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        return score

if __name__ == "__main__":

    result_saving_dir = '/home/nfs/lwu3/Project/Data_poisoning/Result/'
    model_saving_dir = '/home/nfs/lwu3/Project/Data_poisoning/Model/'
    # Weight of the triplet loss
    weight_triplet = float(sys.argv[1])
    # The latent space of the autoencoder
    encoding_dim = int(sys.argv[2])
    load_existing_model = bool(int(sys.argv[3]))

    # =================== Testbench ===================
    # header = 'F:/surfdrive/PhD/'
    # # header = header + '/usr/local/home/wu/ownCloud/PhD/'
    # result_saving_dir = header + 'Research/Poisoning attack/03 - Result/'
    # model_saving_dir = header + 'Research/Poisoning attack/02 - Model/'
    # weight_triplet = 0.5
    # encoding_dim = 32
    # load_existing_model = 1
    # =================== Testbench ===================

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    num_class = len(np.unique(y_train))
    input_shape = np.shape(x_train[0])

    clr = CLR(len(x_train), 256)

    # Build to train triplet-based autoencoder
    tae = TAE()
    tae.build(input_shape=input_shape, encoding_dim=encoding_dim, weight_triplet=weight_triplet, num_class=num_class)
    tae.train(x_train, y_train, x_test, y_test, batch_size=256, epoch=100, verbose=2)
    embeddings, decoded_imgs = tae.predict(x_test)
    TAE.visualize(x_test, decoded_imgs, result_saving_dir)

    # Load or train the target classier
    if load_existing_model:
        model = load_model(model_saving_dir+'target_classifier.h5')
        target_classifier = Target_Classifier(model)
    else:
        target_classifier = Target_Classifier()
        target_classifier.build(input_shape=input_shape, num_class=num_class)
        target_classifier.train(x_train, y_train, x_test, y_test, batch_size=256, epochs=100, verbose=2)

        save_model(target_classifier.model, model_saving_dir+'target_classifier_WTriplet{}.h5'.format(weight_triplet))

    print("Performance from real data:")
    target_classifier.evaluate(x_test, y_test)
    print("Performance from generated data:")
    target_classifier.evaluate(decoded_imgs, y_test)

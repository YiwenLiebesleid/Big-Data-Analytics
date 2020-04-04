import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, Input, Reshape, Flatten, Embedding, Dropout
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.optimizers import Adam, SGD

class GAN:
    def __init__(self, epochs, batch_size, early_stop, patience):
        (self.X_train, self.Y_train), (_, _) = mnist.load_data()
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_train = (self.X_train - 127.5) / 127.5

        self.batch_size = batch_size
        self.z_dim = 30
        self.num_classes = 10
        self.epochs = epochs
        self.early_stop = early_stop
        self.loss = []

        self.G, self.D, self.GAN = None, None, None

        self.epoch_track = 0
        self.epoch_loss_track = [float('inf'),float('inf')]
        self.patience = patience

    def train(self, style=0):
        self.build_model()
        for epoch in range(1, self.epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.epochs))
            num_batches = int(np.ceil(self.X_train.shape[0] / float(self.batch_size)))

            epoch_gen_loss = []
            epoch_disc_loss = []
            for index in range(num_batches):
                image_batch = self.X_train[index * self.batch_size:(index + 1) * self.batch_size]
                label_batch = self.Y_train[index * self.batch_size:(index + 1) * self.batch_size]

                noise = np.random.uniform(-1, 1, (len(image_batch), self.z_dim))
                sampled_labels = np.random.randint(0, self.num_classes, len(image_batch))
                generated_images = self.G.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)
                x = np.concatenate((image_batch, generated_images))
                soft_zero, soft_one = 0, 0.95
                y = np.array([soft_one] * len(image_batch) + [soft_zero] * len(image_batch))
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

                disc_sample_weight = [np.ones(2 * len(image_batch)),
                                      np.concatenate((np.ones(len(image_batch)) * 2, np.zeros(len(image_batch))))]
                #         disc_sample_weight = [np.ones(2*len(image_batch)),np.concatenate((np.ones(len(image_batch)),np.ones(len(image_batch))))]

                epoch_disc_loss.append(self.D.train_on_batch(x, [y, aux_y], sample_weight=disc_sample_weight))

                noise = np.random.uniform(-1, 1, (2 * len(image_batch), self.z_dim))
                sampled_labels = np.random.randint(0, self.num_classes, 2 * len(image_batch))
                trick = np.ones(2 * len(image_batch)) * soft_one
                epoch_gen_loss.append(
                    self.GAN.train_on_batch([noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

            print(np.mean(np.array(epoch_disc_loss), 0))
            print(np.mean(np.array(epoch_gen_loss), 0))
            self.Dloss.append(np.mean(np.array(epoch_disc_loss), 0)[1])
            self.Gloss.append(np.mean(np.array(epoch_gen_loss), 0)[1])
            self.show_for_epoch(style=style, epoch=str(epoch))
            if self.early_stop:
                if epoch != 0 and self.early_stopping(epoch, np.mean(np.array(epoch_gen_loss), 0)):
                    break

    def generator(self):
        latent = Input(shape=(self.z_dim,))
        num_class = Input(shape=(1,), dtype='int32')
        gen = Sequential()
        gen.add(Dense(3 * 3 * 512, input_dim=self.z_dim, activation='relu'))
        gen.add(Reshape((3, 3, 512)))
        gen.add(Conv2DTranspose(256, 5, strides=1, padding='valid', activation='relu',
                                kernel_initializer='glorot_normal'))
        gen.add(BatchNormalization())
        gen.add(Conv2DTranspose(128, 5, strides=2, padding='same', activation='relu',
                                kernel_initializer='glorot_normal'))
        gen.add(BatchNormalization())
        gen.add(
            Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh', kernel_initializer='glorot_normal'))
        cls = Embedding(self.num_classes, self.z_dim, embeddings_initializer='glorot_normal')(num_class)
        hid = layers.multiply([latent, cls])
        fake_image = gen(hid)
        return Model([latent, num_class], fake_image)

    def discriminator(self):
        image = Input(shape=(28, 28, 1))
        dis = Sequential()
        dis.add(Conv2D(32, 3, padding='same', strides=2,input_shape=(28, 28, 1)))
        dis.add(LeakyReLU(0.2))
        dis.add(Dropout(0.3))
        dis.add(Conv2D(64, 3, padding='same', strides=1))
        dis.add(LeakyReLU(0.2))
        dis.add(Dropout(0.3))
        dis.add(Conv2D(128, 3, padding='same', strides=2))
        dis.add(LeakyReLU(0.2))
        dis.add(Dropout(0.3))
        dis.add(Conv2D(256, 3, padding='same', strides=1))
        dis.add(LeakyReLU(0.2))
        dis.add(Dropout(0.3))
        dis.add(Flatten())
        features = dis(image)
        fake = Dense(1, activation='sigmoid', name='generation')(features)
        aux = Dense(self.num_classes, activation='softmax', name='auxiliary')(features)
        return Model(image, [fake, aux])

    def build_model(self):
        adam = Adam(lr=0.0002, beta_1=0.5)
        self.G = self.generator()
        self.D = self.discriminator()
        self.D.compile(optimizer=adam,loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
        latent = Input(shape=(self.z_dim, ))
        num_class = Input(shape=(1,), dtype='int32')
        fake_img = self.G([latent, num_class])
        self.D.trainable = False
        fake, aux = self.D(fake_img)
        self.GAN = Model([latent, num_class], [fake, aux])
        self.GAN.compile(optimizer=adam,loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

    def show_for_epoch(self, style=0, epoch='0'):  # 0: not generate images along training, 1: generate random, 2: generate fixed
        if style == 0:
            return
        noise0 = np.random.uniform(-1, 1, (3, self.z_dim))
        noise = np.empty((25, self.z_dim))
        if style == 1:  # random noise
            noise = np.random.uniform(-1, 1, (25, self.z_dim))
        elif style == 2:  # fix noise z for each group of numbers
            noise[0:10], noise[10:20], noise[20:] = noise0[0], noise0[1], noise0[2]

        labels = np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4])
        generated_images = self.G.predict([noise,labels], verbose=False)
        n = np.sqrt(25).astype(np.int32)
        I_generated = np.empty((28*n, 28*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*28:(i+1)*28, j*28:(j+1)*28] = generated_images[i*n+j, :].reshape(28, 28)
        plt.figure(figsize=(4, 4))
        plt.axis("off")
        plt.imshow(I_generated, cmap='gray')
        plt.savefig('epoch' + str(epoch))
        plt.show()

    def loss_history(self):
        fig = plt.figure(figsize=(10, 10))
        subp = fig.add_subplot(111)
        subp.set_title('Binary Loss')
        subp.plot(range(0, len(self.loss), 1), self.loss)
        fig.savefig("G-Loss")
        fig.show()
        fig = plt.figure(figsize=(10, 10))
        subp = fig.add_subplot(111)
        subp.set_title('Sparse Categorical Loss')
        subp.plot(range(0, len(self.loss), 1), self.loss)
        fig.savefig("D-Loss")
        fig.show()

    def save(self, epoch):
        with open("generator" + epoch + ".json", "w") as json_file:
            json_file.write(model_json)
            g.save_weights("generator" + epoch + ".h5")

    def early_stopping(self, epoch, loss):
        patience = self.patience
        if loss[0] <= self.epoch_loss_track[0] or loss[1] <= self.epoch_loss_track[1]:
            if loss[0] <= self.epoch_loss_track[0]:
                self.epoch_loss_track[0] = loss[0]
            else:
                self.epoch_loss_track[1] = loss[1]
            self.epoch_track = epoch
            return False
        else:
            if epoch - self.epoch_track == patience:
                return True

if __name__ == "__main__":
    my_GAN = GAN(epochs=30, batch_size=100, early_stop=False, patience=5)
    my_GAN.train(style=2)
    my_GAN.loss_history()
    g = my_GAN.G
    model_json = g.to_json()
    my_GAN.save("final")

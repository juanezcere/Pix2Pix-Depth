import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
import tqdm

PATH = "./"
#PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
LEARNING_RATE = 0.00015
BETA_1 = 0.5
LEAKY_RELU_ALPHA = 0.2
BN_MOMENTUM = 0.8
BATCH_SIZE = 32
BATCH_TEST = 5
EPOCHS = 500

class DataLoader():
    def __init__(self, path):
        self.path = path
        self.imagesPath = os.path.join(self.path, "images")
        self.targetPath = os.path.join(self.path, "target")
        urls = os.listdir(self.imagesPath)
        n = len(urls)
        train_n = round(n * 0.99)
        randurls = np.copy(urls)
        np.random.shuffle(randurls)
        self.trainurls = randurls[:train_n]
        self.testurls = randurls[train_n:n]
        self.shuffle()

    def resize(self, img, tgt, size=(IMAGE_SIZE,IMAGE_SIZE)):
        img = tf.image.resize(img, size)
        tgt = tf.image.resize(tgt, size)
        return img, tgt

    def normalize(self, img, tgt):
        img = (img / 127.5) - 1
        tgt = (tgt / 127.5) - 1
        return img, tgt

    @tf.function
    def randomize(self, img, tgt):
        self.resize(img, tgt, (286,286))
        stacked = tf.stack([img, tgt], axis=0)
        cropped = tf.image.random_crop(stacked, size=[2, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
        img, tgt = cropped[0], cropped[1]
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            tgt = tf.image.flip_left_right(tgt)
        return img, tgt

    @tf.function
    def load(self, url, augment=True):
        img = tf.image.decode_jpeg(tf.io.read_file(self.imagesPath + "/" + url))
        tgt = tf.image.decode_jpeg(tf.io.read_file(self.targetPath + "/" + url))
        #img, tgt = split(img)
        img = tf.cast(img, tf.float32)[..., :IMAGE_CHANNELS]
        tgt = tf.cast(tgt, tf.float32)[..., :IMAGE_CHANNELS]
        #img, tgt = resize(img, tgt)
        if augment:
            img, tgt = self.randomize(img, tgt)
        img, tgt = self.normalize(img, tgt)
        return img, tgt

    def loadtrain(self, url):
        return self.load(url, True)

    def loadtest(self, url):
        return self.load(url, False)
    
    def shuffle(self):
        np.random.shuffle(self.trainurls)
        np.random.shuffle(self.testurls)
        self.train = tf.data.Dataset.from_tensor_slices(self.trainurls)
        self.train = self.train.map(self.loadtrain, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train = self.train.batch(BATCH_SIZE)
        self.test = tf.data.Dataset.from_tensor_slices(self.testurls)
        self.test = self.test.map(self.loadtest, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test = self.test.batch(BATCH_TEST)

class Pix2Pix():
    def __init__(self, path):
        self.path = path
        self.inputPath = os.path.join(self.path, "input")
        self.outputPath = os.path.join(self.path, "output")
        if not os.path.exists(self.outputPath):
            os.mkdir(self.outputPath)
        self.modelsPath = os.path.join(self.path, "models")
        if not os.path.exists(self.modelsPath):
            os.mkdir(self.modelsPath)
        self.image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
        self.dataset = DataLoader(self.inputPath)
        patch = int(IMAGE_SIZE / 2**4)
        self.disc_patch = (patch, patch, 1)
        self.generator_filters = 64
        self.discriminator_filters = 64
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
        self.gen_file = os.path.join(self.modelsPath, "generator.h5")
        if os.path.exists(self.gen_file):
            self.generator = tf.keras.models.load_model(self.gen_file)
        else:
            self.generator = self.Generator()
        #tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi=64)
        #self.generator.summary()
        self.dis_file = os.path.join(self.modelsPath, "discriminator.h5")
        if os.path.exists(self.dis_file):
            self.discriminator = tf.keras.models.load_model(self.dis_file)
        else:
            self.discriminator = self.Discriminator()
        #tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)
        #self.discriminator.summary()
        self.com_file = os.path.join(self.modelsPath, "combined.h5")
        source_image = tf.keras.layers.Input(shape=self.image_shape)
        destination_image = tf.keras.layers.Input(shape=self.image_shape)
        generated_image = self.generator(destination_image)
        self.discriminator.trainable = False
        valid = self.discriminator([generated_image, destination_image])
        self.combined = tf.keras.models.Model(inputs=[source_image, destination_image], outputs=[valid, generated_image])
        self.combined.compile(loss=["mse", "mae"], loss_weights=[1, 100], optimizer=optimizer)
        #self.combined.summary()
        self.epoch_file = os.path.join(self.modelsPath, "epoch.txt")
        if os.path.exists(self.epoch_file):
            f = open(self.epoch_file, 'r')
            l = f.read()
            f.close()
            self.epoch = int(l)
        else:
            self.epoch = 0

    def Generator(self):
        def conv2d(layer_input, filters, bn=True):
            downsample = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same")(layer_input)
            downsample = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(downsample)
            if bn:
                downsample = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM)(downsample)
            return downsample

        def deconv2d(layer_input, skip_input, filters, dropout_rate=0):
            upsample = tf.keras.layers.UpSampling2D(size=2)(layer_input)
            upsample = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=1, padding="same", activation="relu")(upsample)
            if dropout_rate:
                upsample = tf.keras.layers.Dropout(dropout_rate)(upsample)
            upsample = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM)(upsample)
            upsample = tf.keras.layers.Concatenate()([upsample, skip_input])
            return upsample

        downsample_0 = tf.keras.layers.Input(shape=self.image_shape)
        downsample_1 = conv2d(downsample_0, self.generator_filters, bn=False)
        downsample_2 = conv2d(downsample_1, self.generator_filters*2)
        downsample_3 = conv2d(downsample_2, self.generator_filters*4)
        downsample_4 = conv2d(downsample_3, self.generator_filters*8)
        downsample_5 = conv2d(downsample_4, self.generator_filters*8)
        downsample_6 = conv2d(downsample_5, self.generator_filters*8)
        downsample_7 = conv2d(downsample_6, self.generator_filters*8)

        upsample_1 = deconv2d(downsample_7, downsample_6, self.generator_filters*8)
        upsample_2 = deconv2d(upsample_1, downsample_5, self.generator_filters*8)
        upsample_3 = deconv2d(upsample_2, downsample_4, self.generator_filters*8)
        upsample_4 = deconv2d(upsample_3, downsample_3, self.generator_filters*4)
        upsample_5 = deconv2d(upsample_4, downsample_2, self.generator_filters*2)
        upsample_6 = deconv2d(upsample_5, downsample_1, self.generator_filters)
        upsample_7 = tf.keras.layers.UpSampling2D(size=2)(upsample_6)

        output_image = tf.keras.layers.Conv2D(IMAGE_CHANNELS, kernel_size=4, strides=1, padding="same", activation="tanh")(upsample_7)
        return tf.keras.models.Model(downsample_0, output_image)

    def Discriminator(self):
        def discriminator_layer(layer_input, filters, bn=True):
            discriminator_layer = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same")(layer_input)
            discriminator_layer = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(discriminator_layer)
            if bn:
                discriminator_layer = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM)(discriminator_layer)
            return discriminator_layer

        source_image = tf.keras.layers.Input(shape=self.image_shape)
        destination_image = tf.keras.layers.Input(shape=self.image_shape)
        combined_images = tf.keras.layers.Concatenate(axis=-1)([source_image, destination_image])
        discriminator_layer_1 = discriminator_layer(combined_images, self.discriminator_filters, bn=False)
        discriminator_layer_2 = discriminator_layer(discriminator_layer_1, self.discriminator_filters*2)
        discriminator_layer_3 = discriminator_layer(discriminator_layer_2, self.discriminator_filters*4)
        discriminator_layer_4 = discriminator_layer(discriminator_layer_3, self.discriminator_filters*8)
        validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding="same")(discriminator_layer_4)
        return tf.keras.models.Model([source_image, destination_image], validity)
    
    def progress(self):
        for img, tgt in self.dataset.test.take(1):
            prediction = self.generator(img, training=True)
            discrimination = self.discriminator([tgt, prediction], training=True)
            fig, cols, h = plt.figure(figsize=(50,50)), 4, 0
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            for i in range(BATCH_TEST):
                title = ["Input", "Real", "Predicted", "Discrimination"]
                imgs = [img[i], tgt[i], prediction[i], discrimination[i]]
                for j in range(4):
                    fig.add_subplot(BATCH_TEST, cols, j+h+1)
                    plt.title(title[j])
                    if j < 3:
                        plt.imshow(imgs[j] * 0.5 + 0.5)
                    else:
                        plt.imshow(imgs[j][...,-1], vmin=-20, vmax=20, cmap="RdBu_r")
                        plt.colorbar()
                    plt.axis("off")
                h+=4
            plt.savefig(self.outputPath + "/graphic_" + str(self.epoch) + ".jpg")
            plt.show()

    def train(self):
        valid = np.ones((BATCH_SIZE,) + self.disc_patch)
        fake = np.zeros((BATCH_SIZE,) + self.disc_patch)
        self.d_losses = []
        self.g_losses = []
        self.progress()
        for epoch in range(self.epoch+1,EPOCHS+1):
            epoch_d_losses = []
            epoch_g_losses = []
            pbar = tqdm.tqdm(total=len(self.dataset.trainurls))
            for img, tgt in self.dataset.train:
                generated_images = self.generator.predict(img)
                d_loss_real = self.discriminator.train_on_batch([tgt, img], valid)
                d_loss_fake = self.discriminator.train_on_batch([generated_images, img], fake)
                d_losses = 0.5 * np.add(d_loss_real, d_loss_fake)
                g_losses = self.combined.train_on_batch([tgt, img], [valid, tgt])
                epoch_d_losses.append(d_losses)
                epoch_g_losses.append(g_losses)
                pbar.update(1)
            print("EPOCH: " + str(epoch) + ", D_LOSSES: " + str(d_losses) + ", G_LOSSES: " + str(g_losses))
            self.d_losses.append(np.average(epoch_d_losses, axis=0))
            self.g_losses.append(np.average(epoch_g_losses, axis=0))
            pbar.close()
            if epoch % 10 == 0:
                self.generator.save(self.gen_file)
                self.discriminator.save(self.dis_file)
                self.combined.save(self.com_file)
                f = open(self.epoch_file, 'w')
                f.write(str(epoch))
                f.close()
                self.epoch = epoch
            self.progress()
            self.dataset.shuffle()

    def test(self):
        cam = cv2.VideoCapture(0)
        time.sleep(5)
        #i = 0
        while True:
            _, image = cam.read()
            image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE), cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.float32)
            #image = np.array(imageio.imread(image_path))
            image = (image / 127.5) - 1
            generated_batch = self.generator.predict(np.array([image]))
            #concat = Helpers.unnormalize(np.concatenate([image_normalized, generated_batch[0]], axis=1))
            img = cv2.cvtColor(np.float32((0.5*generated_batch[0]+0.5)*255), cv2.COLOR_RGB2BGR)
            cv2.imshow("Zabavy", np.uint8(img))
            #cv2.imwrite(BASE_OUTPUT_PATH + "{}.png".format(i), cv2.cvtColor(np.float32(concat), cv2.COLOR_RGB2BGR))
            #i+=1
            cv2.waitKey(1)

gan = Pix2Pix(PATH)
gan.train()
gan.test()
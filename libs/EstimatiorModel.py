import datetime
from libs.BaseClass import BaseClass
from libs.UNet import *
from libs.Discriminators import ClassicalDiscriminator

# ======================================================================================================================

class TemplateEstimatior(BaseClass):
    def __init__(self, config, args, type="unet"):

        self.config = config
        self.type = type
        self.discriminator_weight = args.discriminator_weight if "discriminator_weight" in args else 1
        self.__layer_normalisation = args.unet_layer_normalisation if "unet_layer_normalisation" in args else self.config.models["unet"]["layer_normalisation"]

        self.__createResDirs(args)

        self.tensor_board = keras.callbacks.TensorBoard(
            log_dir=self.tensor_board_dir,
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            write_images=True
        )

        if self.type == "Dtt_Dxx":
            self.__initUnetXUnetT(args)
        elif self.type == "Dtt_Dt_Dxx_Dx":
            self.__initUnetDxUnetDt(args)
        else:
            self.__initUnetModel(args)

        self.tensor_board.set_model(self.EstimationModel)

    def __initUnetModel(self):
        input  = Input(shape=(self.config["dataset"]["args"]["target_size"][0],
                              self.config["dataset"]["args"]["target_size"][1],
                              self.config.models["unet"]["input_channels"]))

        self.UnetModel = UNet(filters=self.config.models["unet"]["filters"],
                              layer_normalisation=self.__layer_normalisation).init(input)
        self.EstimationModel = Model(input, self.UnetModel(input))

        optimizer = self.__getOptimizer(self.config.models["unet"]["optimizer"], self.config.models["unet"]["lr"])
        loss = self.__getLoss(self.config.models["unet"]["loss"])

        self.EstimationModel.compile(loss=loss, optimizer=optimizer)

    def __initUnetXUnetT(self):
        input_x = Input(shape=(self.config["dataset"]["args"]["target_size"][0],
                               self.config["dataset"]["args"]["target_size"][1],
                               self.config.models["unet"]["input_channels"]))

        input_t = Input(shape=(self.config["dataset"]["args"]["target_size"][0],
                               self.config["dataset"]["args"]["target_size"][1], 1))

        # --- init Models -------
        self.UnetXModel = UNet(filters=self.config.models["unet"]["filters"],
                               layer_normalisation=self.__layer_normalisation,
                               output_channels=1,
                               name="UNetX").init(input_x)

        self.UnetTModel = UNet(filters=self.config.models["UnetXUnetT"]["filters"],
                               layer_normalisation=self.__layer_normalisation,
                               output_channels=self.config.models["unet"]["input_channels"],
                               name="UNetT").init(input_t)

        # --- Nested estimator -------
        optimizer = self.__getOptimizer(self.config.models["UnetXUnetT"]["optimizer"], self.config.models["UnetXUnetT"]["lr"])
        loss_t = self.__getLoss(self.config.models["unet"]["loss"])
        loss_x = self.__getLoss(self.config.models["UnetXUnetT"]["loss"])

        estimation_t = self.UnetXModel(input_x)
        estimation_x = self.UnetTModel(estimation_t)

        self.EstimationModel = Model(inputs=[input_x],
                                     outputs=[estimation_t, estimation_x],
                                     name="estimator")

        self.EstimationModel.compile(loss=[loss_t, loss_x],
                                     loss_weights=[self.config.models["unet"]["loss_weight"], self.config.models["UnetXUnetT"]["loss_weight"]],
                                     optimizer=optimizer)

    def __initUnetDxUnetDt(self):
        input_x = Input(shape=(self.config["dataset"]["args"]["target_size"][0],
                               self.config["dataset"]["args"]["target_size"][1],
                               self.config.models["unet"]["input_channels"]))

        input_t = Input(shape=(self.config["dataset"]["args"]["target_size"][0],
                               self.config["dataset"]["args"]["target_size"][1], 1))

        # --- init Models -------
        self.UnetXModel = UNet(filters=self.config.models["unet"]["filters"],
                               layer_normalisation=self.__layer_normalisation,
                               output_channels=1,
                               name="UNetX").init(input_x)

        self.UnetTModel = UNet(filters=self.config.models["UnetXUnetT"]["filters"],
                               layer_normalisation=self.__layer_normalisation,
                               output_channels=self.config.models["unet"]["input_channels"],
                               name="UNetT").init(input_t)

        self.Dx = ClassicalDiscriminator(filters=self.config.models["ClassicalDiscriminator"]["filters"],
                                              name="Dx").init(input_x)
        self.Dt = ClassicalDiscriminator(filters=self.config.models["ClassicalDiscriminator"]["filters"],
                                              name="Dt").init(input_t)

        # --- DxModel -------
        optimizer_discr = self.__getOptimizer(self.config.models["ClassicalDiscriminator"]["optimizer"], self.config.models["ClassicalDiscriminator"]["lr"])
        loss_dx = self.__getLoss(self.config.models["ClassicalDiscriminator"]["loss"])

        self.DxModel = Model(inputs=input_x, outputs=self.Dx(input_x), name="discriminator")
        self.DxModel.compile(loss=loss_dx, optimizer=optimizer_discr)
        self.Dx.trainable = False

        # --- DtModel -------
        optimizer_discr = self.__getOptimizer(self.config.models["ClassicalDiscriminator"]["optimizer"], self.config.models["ClassicalDiscriminator"]["lr"])
        loss_dt = self.__getLoss(self.config.models["ClassicalDiscriminator"]["loss"])

        self.DtModel = Model(inputs=input_t, outputs=self.Dt(input_t), name="discriminator")
        self.DtModel.compile(loss=loss_dx, optimizer=optimizer_discr)
        self.Dt.trainable = False

        # --- Nested estimator -------
        optimizer  = self.__getOptimizer(self.config.models["UnetXUnetT"]["optimizer"], self.config.models["UnetXUnetT"]["lr"])

        loss_t     = self.__getLoss(self.config.models["unet"]["loss"])
        loss_x     = self.__getLoss(self.config.models["UnetXUnetT"]["loss"])

        estimation_t = self.UnetXModel(input_x)
        estimation_x = self.UnetTModel(estimation_t)

        self.EstimationModel = Model(inputs=[input_x],
                                     outputs=[estimation_t, estimation_x, self.Dx(estimation_x), self.Dt(estimation_t)],
                                     name="estimator")

        self.EstimationModel.compile(loss=[loss_t, loss_x, loss_dx, loss_dt],
                                     loss_weights=[self.config.models["unet"]["loss_weight"],
                                                   self.config.models["UnetXUnetT"]["loss_weight"],
                                                   self.config.models["ClassicalDiscriminator"]["loss_weight"],
                                                   self.config.models["ClassicalDiscriminator"]["loss_weight"]],
                                     optimizer=optimizer)

    def __getLoss(self, loss):
        if loss == "binary_crossentropy":
            return tf.keras.losses.binary_crossentropy
        elif loss == "mse":
            return tf.keras.losses.mean_squared_error
        elif loss == "l1":
            return tf.keras.losses.mean_absolute_error

    def __getOptimizer(self, optimazer, lr):
        if optimazer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)


    def __createResDirs(self, args):
        self.checkpoint_dir = self.makeDir(self.config.checkpoint_dir + "/" + args.checkpoint_dir)
        self.results_dir = self.makeDir(self.config.results_dir + "/" + args.dir)
        self.tensor_board_dir = self.makeDir("./TensorBoard/" + args.dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.makeDir(self.config.results_dir + "/" + args.dir + "/prediction/")


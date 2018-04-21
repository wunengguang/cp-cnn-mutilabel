import json#用于python编码和解码的对象

import shutil#是对OS的文件的补充，－－移动复制打包压缩解压
import os
import pickle #将序列化对象保存到磁盘中
from callback import MultipleClassAUROC, MultiGPUModelCheckpoint #
from configparser import ConfigParser#实现文件的增删改查
from generator import custom_image_generator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator #图像数据生成
from keras.utils import multi_gpu_model#多ＧＰＵ模式
from models.densenet121 import get_model
from utility import split_data, get_sample_counts, create_symlink
from weights import get_class_weights


def main():
    # parser config
    #－－－读取配置文件-------
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    #-----获取默认参数-------
    output_dir = cp["DEFAULT"].get("output_dir")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    train_patient_count = cp["DEFAULT"].getint("train_patient_count")
    dev_patient_count = cp["DEFAULT"].getint("dev_patient_count")
    data_entry_file = cp["DEFAULT"].get("data_entry_file")
    class_names = cp["DEFAULT"].get("class_names").split(",")

    # train config
    #----获取训练参数--------
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    train_steps = cp["TRAIN"].get("train_steps")
    patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
    validation_steps = cp["TRAIN"].get("validation_steps")
    positive_weights_multiply = cp["TRAIN"].getfloat("positive_weights_multiply")
    use_class_balancing = cp["TRAIN"].getboolean("use_class_balancing")
    use_default_split = cp["TRAIN"].getboolean("use_default_split")
    # if previously trained weights is used, never re-split

    if use_trained_model_weights:
        # resuming mode
        print("** use trained model weights, turn on use_skip_split automatically **")
        use_skip_split = True
        # load training status for resuming
        training_stats_file = os.path.join(output_dir, ".training_stats.json")
        if os.path.isfile(training_stats_file):
            # TODO: add loading previous learning rate?
            training_stats = json.load(open(training_stats_file))
        else:
            training_stats = {}
    else:
        # start over
        use_skip_split = cp["TRAIN"].getboolean("use_skip_split ")
        training_stats = {}

    split_dataset_random_state = cp["TRAIN"].getint("split_dataset_random_state")
    show_model_summary = cp["TRAIN"].getboolean("show_model_summary")
    # end parser config
    output_dir= output_dir+'2'
    # check output_dir, create it if not exists
    #判断文件夹是否存在，不存在建立一个文件夹
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    running_flag_file = os.path.join(output_dir, ".training.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()

    try:
        print(f"backup config file to {output_dir}")

        shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

        # split train/dev/test
        #数据集分割
        if use_default_split:
            datasets = ["train1", "dev1", "test1"]
            for dataset in datasets:
                shutil.copy(f"./data1/default_split1/{dataset}.csv", output_dir)#拷贝某个文件夹下面的文件
        elif not use_skip_split:
            print("** split dataset **")
            split_data(
                data_entry_file,
                class_names,
                train_patient_count,
                dev_patient_count,
                output_dir,
                split_dataset_random_state,
            )

        # get train/dev sample counts
        #获取样本数和正样本数
        train_counts, train_pos_counts = get_sample_counts(output_dir, "train", class_names)
        dev_counts, _ = get_sample_counts(output_dir, "dev", class_names)

        # compute steps
        if train_steps == "auto":
            train_steps = int(train_counts / batch_size)#训练步数等于样本数除以batchsize
        else:
            try:
                train_steps = int(train_steps)
            except ValueError:
                raise ValueError(f"""
                train_steps: {train_steps} is invalid,
                please use 'auto' or integer.
                """)
        print(f"** train_steps: {train_steps} **")

        if validation_steps == "auto":
            validation_steps = int(dev_counts / batch_size)
        else:
            try:
                validation_steps = int(validation_steps)
            except ValueError:
                raise ValueError(f"""
                validation_steps: {validation_steps} is invalid,
                please use 'auto' or integer.
                """)
        print(f"** validation_steps: {validation_steps} **")

        # compute class weights
        print("** compute class weights from training data **")
        class_weights = get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
            use_class_balancing=use_class_balancing
        )
        print("** class_weights **")
        #－－－－打印出每一个类别所对应的权重-----
        for c, w in class_weights.items():
            print(f"  {c}: {w}")

        print("** load model **")
        #------加载模型----------
        if use_base_model_weights:
            base_model_weights_file = cp["TRAIN"].get("base_model_weights_file")
        else:
            base_model_weights_file = None
        if use_trained_model_weights:
            if use_best_weights:
                model_weights_file = os.path.join(output_dir, f"best_{output_weights_name}")
            else:
                model_weights_file = os.path.join(output_dir, output_weights_name)
        else:
            model_weights_file = None
        model = get_model(class_names, base_model_weights_file, model_weights_file)
        if show_model_summary:
            print(model.summary())

        # recreate symlink folder for ImageDataGenerator
        symlink_dir_name = "image_links"
        create_symlink(image_source_dir, output_dir, symlink_dir_name)

        print("** create image generators **")
        train_data_path = f"{output_dir}/{symlink_dir_name}/train/"
        train_generator = custom_image_generator(
            ImageDataGenerator(horizontal_flip=True, rescale=1./255),
            train_data_path,
            batch_size=batch_size,
            class_names=class_names,
        )
        dev_data_path = f"{output_dir}/{symlink_dir_name}/dev/"
        dev_generator = custom_image_generator(
            ImageDataGenerator(horizontal_flip=True, rescale=1./255),
            dev_data_path,
            batch_size=batch_size,
            class_names=class_names,
        )

        output_weights_path = os.path.join(output_dir, output_weights_name)
        print(f"** set output weights path to: {output_weights_path} **")

        print("** check multiple gpu availability **")
        #进行多ＧＰＵ
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            #表示多ＧＰＵ模型
            model_train = multi_gpu_model(model, gpus)
            # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
            checkpoint = MultiGPUModelCheckpoint(
                filepath=output_weights_path,
                base_model=model,
            )
        else:
            model_train = model
            checkpoint = ModelCheckpoint(output_weights_path)

        print("** compile model with class weights **")
        optimizer = Adam(lr=initial_learning_rate)
        model_train.compile(optimizer=optimizer, loss="binary_crossentropy")

        auroc = MultipleClassAUROC(
            generator=dev_generator,
            steps=validation_steps,
            class_names=class_names,
            weights_path=output_weights_path,
            stats=training_stats,
        )
        callbacks = [
            checkpoint,
            TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr, verbose=1),
            auroc,
        ]

        print("** training start **")
        history = model_train.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=dev_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
        )

        # dump history
        print("** dump history **")
        with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
            pickle.dump({
                "history": history.history,
                "auroc": auroc.aurocs,
            }, f)
        print("** done! **")

    finally:
        os.remove(running_flag_file)


if __name__ == "__main__":
    main()

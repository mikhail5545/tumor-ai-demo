import tensorflow as tf


def __run():
    """
    
    """


    from core.ExtractData import _extract_data

    from core.SeparateDataFrames import _prepare_test_dataframes
    from core.SeparateDataFrames import _separate_dataframe

    from core.DataGenerators import _image_data_generator
    from core.DataGenerators import _train_dataframe_iterator
    from core.DataGenerators import _test_dataframe_iterator

    from console.visualize import visualize


    __xray_df, __classes, __low_accuracy_classes = _extract_data()

    print(f"<---- classes finded: {len(__classes), __classes} ; low accuray classes (less than 1000 findings in input data: {len(__low_accuracy_classes), __low_accuracy_classes}) ---->\n")

    __train_df, __test_df = _separate_dataframe(__xray_df, __classes)

    print(f"<---- train dataframe size: {len(__train_df)}; test dataframe size: {len(__test_df)} ---->\n")


    __image_data_generator = _image_data_generator()


    __train_dataframe_iterator = _train_dataframe_iterator(
        image_data_generator=__image_data_generator,
        train_df=__train_df,
        path_column='path',
        target_column='class',
        batch_size=32
    )

    print("<---- Train dataframe iterator was created. ---->")


    __test_dataframe_iterator = _test_dataframe_iterator(
        image_data_generator=__image_data_generator,
        test_df=__test_df,
        path_column='path',
        target_column='class',
        batch_size=256
    )

    print("<---- Test dataframe iterator was created. ---->")


    __test_x, __test_y = _prepare_test_dataframes(
        image_data_generator=__image_data_generator,
        input_df=__test_df,
        path_column='path',
        target_column='class'
    )


    __train_x, __train_y = next(__train_dataframe_iterator)


    from core.NrutalModel import neural_model

    __model = neural_model(__train_x)

    from config import WEIGHTS_PATH

    # __model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=WEIGHTS_PATH,
    #     monitor='val_accuracy',
    #     verbose=1,
    #     save_best_only=True,
    #     mode='max',
    #     save_weights_only=True
    # )


    # __early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     mode='min',
    #     patience=3
    # )


    __model.fit_generator(
        generator=__train_dataframe_iterator,
        steps_per_epoch=100,
        validation_data=(__test_x, __test_y),
        epochs=10,
        # callbacks=[__model_checkpoint, __early_stopping]
        # callbacks=__model_checkpoint
    )


    __model.save_weights(
        filepath=WEIGHTS_PATH,
        overwrite=True,
        save_format='h5'
    )


    predictions = __model.predict(
        x = __test_x,
        batch_size=1024,
        verbose=True
    )


    visualize(predictions, __classes)

    __model.load_weights(
        filepath=WEIGHTS_PATH
    )

    __predictions = __model.predict(
        x=__test_x,
        batch_size=1024
    )

    visualize(__predictions, __classes)


if __name__ == "__main__":

    import subprocess

    subprocess.run(['nvidia-smi'])

    print(tf.config.list_physical_devices('GPU'))
    tf.debugging.set_log_device_placement(False) # to see wich devices operations and tensors assigned to


    __run()

#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
import transformers
from transformers import ViTFeatureExtractor
import tensorflow.keras
from tensorflow import keras 
from tensorflow.keras import layers
import os
import datasets
from transformers.keras_callbacks import PushToHubCallback
from tensorflow.keras.callbacks import TensorBoard as TensorboardCallback, EarlyStopping
from huggingface_hub import HfFolder
from transformers import TFViTForImageClassification, create_optimizer
from transformers import DefaultDataCollator
import matplotlib.pyplot as plt

model_id = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

def create_image_folder_dataset(root_path):
    """creates `Dataset` from image folder structure"""
  
    # get class names by folders names
    _CLASS_NAMES= os.listdir(root_path)
    # defines `datasets` features`
    features=datasets.Features({
                  "img": datasets.Image(),
                  "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
              })
    # temp list holding datapoints for creation
    img_data_files=[]
    label_data_files=[]
    # load images into list for creation
    for img_class in os.listdir(root_path):
        for img in os.listdir(os.path.join(root_path,img_class)):
            path_=os.path.join(root_path,img_class,img)
            img_data_files.append(path_)
            label_data_files.append(img_class)
    # create dataset
    ds = datasets.Dataset.from_dict({"img":img_data_files,"label":label_data_files},features=features)
    return ds

# use keras image data augementation processing
def augmentation(examples):
    # print(examples["img"])
    examples["pixel_values"] = [data_augmentation(image) for image in examples["img"]]
    return examples


# basic processing (only resizing)
def process(examples):
    examples.update(feature_extractor(examples['img'], ))
    return examples

def main():
    #model_id = "google/vit-base-patch16-224-in21k"
    path = "EuroSAT/data/"
    eurosat_ds = create_image_folder_dataset(path)
    img_class_labels = eurosat_ds.features["label"].names
    print(img_class_labels)
    print(eurosat_ds)
    #feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
    # learn more about data augmentation here: https://www.tensorflow.org/tutorials/images/data_augmentation
    data_augmentation = keras.Sequential(
        [
            layers.Resizing(feature_extractor.size, feature_extractor.size),
            layers.Rescaling(1./255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    # we are also renaming our label col to labels to use `.to_tf_dataset` later
    eurosat_ds = eurosat_ds.rename_column("label", "labels")
    processed_dataset = eurosat_ds.map(process, batched=True)
    # test size will be 15% of train dataset
    test_size=.15
    processed_dataset = processed_dataset.shuffle().train_test_split(test_size=test_size)
    id2label = {str(i): label for i, label in enumerate(img_class_labels)}
    label2id = {v: k for k, v in id2label.items()}
    num_train_epochs = 5
    train_batch_size = 32
    eval_batch_size = 32
    learning_rate = 3e-5
    weight_decay_rate=0.01
    num_warmup_steps=0
    output_dir=model_id.split("/")[1]
    hub_token = HfFolder.get_token() # or your token directly "hf_xxx"
    hub_model_id = f'{model_id.split("/")[1]}-euroSat'
    fp16=True

    # Train in mixed-precision float16
    # Comment this line out if you're using a GPU that will not benefit from this
    if fp16:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # Data collator that will dynamically pad the inputs received, as well as the labels.
    data_collator = DefaultDataCollator(return_tensors="tf")

    # converting our train dataset to tf.data.Dataset
    tf_train_dataset = processed_dataset["train"].to_tf_dataset(
       columns=['pixel_values'],
       label_cols=["labels"],
       shuffle=True,
       batch_size=train_batch_size,
       collate_fn=data_collator)

    # converting our test dataset to tf.data.Dataset
    tf_eval_dataset = processed_dataset["test"].to_tf_dataset(
       columns=['pixel_values'],
       label_cols=["labels"],
       shuffle=True,
       batch_size=eval_batch_size,
       collate_fn=data_collator)
    # create optimizer wight weigh decay
    num_train_steps = len(tf_train_dataset) * num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=weight_decay_rate,
        num_warmup_steps=num_warmup_steps,
    )

    # load pre-trained ViT model
    model = TFViTForImageClassification.from_pretrained(
        model_id,
        num_labels=len(img_class_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # define loss
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # define metrics 
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
    ]

    # compile model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics
                  )
    '''
    callbacks=[]

    callbacks.append(TensorboardCallback(log_dir=os.path.join(output_dir,"logs")))
    callbacks.append(EarlyStopping(monitor="val_accuracy",patience=1))
    if hub_token:
        callbacks.append(PushToHubCallback(output_dir=output_dir,
                                         hub_model_id=hub_model_id,
                                         hub_token=hub_token))
    '''
    # print model summary
    model.summary()

    history = model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        #callbacks=callbacks,
        epochs=num_train_epochs
    )
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()









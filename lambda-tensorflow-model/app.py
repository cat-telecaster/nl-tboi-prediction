import os
import json
import numpy as np
import boto3
import zipfile
import tempfile
import logging

import tensorflow as tf

region = os.environ['AWS_REGION']
ACCESS_KEY_ID = os.environ['ACCESS_KEY_ID']
ACCESS_KEY = os.environ['ACCESS_KEY']
BUCKET_NAME = os.environ['BUCKET_NAME']
MODEL_NAME = os.environ['MODEL_NAME']

print('Downloading model...\n')
client_s3 = boto3.client("s3",
                         aws_access_key_id=ACCESS_KEY_ID,
                         aws_secret_access_key=ACCESS_KEY,
                         region_name=region)
result = client_s3.download_file(BUCKET_NAME, MODEL_NAME+'.zip', f"/tmp/{MODEL_NAME}.zip")

print('Loading model...\n')
with zipfile.ZipFile(f"/tmp/{MODEL_NAME}.zip") as zip_ref:
    zip_ref.extractall(f"/tmp/{MODEL_NAME}")
model = tf.keras.models.load_model(f"/tmp/{MODEL_NAME}/{MODEL_NAME}")

# Simple numpy numeric to one-hot function
def convertToOneHot(num, length):
    buff = np.zeros(length)
    buff[num - 1] = 1
    return buff

# Input conversion function from string to one-hot based on lookup table
def stringInput(character, end_objective, timed_objective):
    character_dict = {'Isaac': convertToOneHot(1, 34),
                      'Magdalene': convertToOneHot(2, 34),
                      'Cain': convertToOneHot(3, 34),
                      'Judas': convertToOneHot(4, 34),
                      'Blue Baby': convertToOneHot(5, 34),
                      'Eve': convertToOneHot(6, 34),
                      'Samson': convertToOneHot(7, 34),
                      'Azazel': convertToOneHot(8, 34),
                      'Lazarus': convertToOneHot(9, 34),
                      'Eden': convertToOneHot(10, 34),
                      'Lost': convertToOneHot(11, 34),
                      'Lilith': convertToOneHot(12, 34),
                      'Keeper': convertToOneHot(13, 34),
                      'Apollyon': convertToOneHot(14, 34),
                      'Forgotten': convertToOneHot(15, 34),
                      'Bethany': convertToOneHot(16, 34),
                      'Jacob&Esau': convertToOneHot(17, 34),
                      'Tainted Isaac': convertToOneHot(18, 34),
                      'Tainted Magdalene': convertToOneHot(19, 34),
                      'Tainted Cain': convertToOneHot(20, 34),
                      'Tainted Judas': convertToOneHot(21, 34),
                      'Tainted Blue Baby': convertToOneHot(22, 34),
                      'Tainted Eve': convertToOneHot(23, 34),
                      'Tainted Samson': convertToOneHot(24, 34),
                      'Tainted Azazel': convertToOneHot(25, 34),
                      'Tainted Lazarus': convertToOneHot(26, 34),
                      'Tainted Eden': convertToOneHot(27, 34),
                      'Tainted Lost': convertToOneHot(28, 34),
                      'Tainted Lilith': convertToOneHot(29, 34),
                      'Tainted Keeper': convertToOneHot(30, 34),
                      'Tainted Apollyon': convertToOneHot(31, 34),
                      'Tainted Forgotten': convertToOneHot(32, 34),
                      'Tainted Bethany': convertToOneHot(33, 34),
                      'Tainted Jacob': convertToOneHot(34, 34)
                      }
    objective_dict = {'???': convertToOneHot(1, 7),
                      'The Lamb': convertToOneHot(2, 7),
                      'Mega Satan': convertToOneHot(3, 7),
                      'Ultra Greed': convertToOneHot(4, 7),
                      'Delirium': convertToOneHot(5, 7),
                      'Mother': convertToOneHot(6, 7),
                      'The Beast': convertToOneHot(7, 7)
                      }
    timed_dict = {'None': np.zeros(2),
                  'Boss Rush': convertToOneHot(1, 2),
                  'Hush': convertToOneHot(2, 2)
                  }

    return np.concatenate((character_dict[character], objective_dict[end_objective], timed_dict[timed_objective]),axis=None).reshape(-1,43)

# Lambda handler code
def lambda_handler(event, context):
    try:
        character = event['character']
        main_obj = event['obj']
        timed_obj = event['timed']
    except KeyError:
        data = json.loads(event['body'])
        character = data['character']
        main_obj = data['obj']
        timed_obj = data['timed']

    prediction = model.predict(stringInput(character, main_obj, timed_obj))[0].tolist()

    return {
        'isBase64Encoded': False,
        "statusCode": 200,
        "body": json.dumps(
            {
                "predicted_label": prediction
            }
        ),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
            "Access-Control-Allow-Origin": "ENTER_YOUR_OWN",
            "X-Requested-With": "*"
        }
    }

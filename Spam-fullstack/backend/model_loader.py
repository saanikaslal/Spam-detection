from tensorflow.keras.models import load_model #type:ignore
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

#max_len read form config
with open("saved-model/config.json") as f:
    config = json.load(f)
    MAX_LEN = config.get("MAX_LEN", 20)   # fallback = 20

#loading model
model = load_model("saved-model/spam_model.keras")

# loading tokenizer

with open("saved-model/tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


#prediction function 

def predict_message(message:str, max_len=100):
    
    #convert text to integer sequence
    seq = tokenizer.texts_to_sequences([message])
    #padded to MAX_LEN
    padded = pad_sequences(seq, maxlen=MAX_LEN)


    #predict
    prediction = model.predict(padded)[0][0]
    label = "spam" if prediction > 0.5 else "not_spam"

    return {
        "label": label,
        "confidence": float(prediction)
    }

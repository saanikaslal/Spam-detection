from tensorflow.keras.models import load_model #type:ignore
import tensorflow as tf

model = load_model("saved-model/spam_model.keras")

def predict_message(message, tokenizer=None, max_len=100):
    if tokenizer:
        seq = tokenizer.texts_to_sequences([message])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    else:
        # Here you will later add real preprocessing
        padded = [message]

    prediction = model.predict(padded)[0][0]
    label = "spam" if prediction > 0.5 else "not_spam"

    return {
        "label": label,
        "confidence": float(prediction)
    }

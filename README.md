# Text Generation Using LSTM in TensorFlow

This project implements a word-level **text generation model** using **LSTM (Long Short-Term Memory)** networks in TensorFlow and Keras. The model learns patterns from input text and predicts the next word in a sequence.

## Features

* Tokenizes text and converts words to unique numerical indices.
* Prepares input sequences for training using sliding windows of words.
* Pads sequences to a fixed length for LSTM input.
* Uses a **two-layer LSTM model** to predict the next word.
* Generates text given a starting seed phrase.
* **Customizable:** You can add your own text data in the `text` variable, and the model will learn according to your writing style.

> Note: The dataset used in this demo is intentionally small to demonstrate functionality. For better results, provide a larger dataset.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/text-generation-lstm.git
cd text-generation-lstm
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install tensorflow numpy
```

## Usage

1. **Prepare your text data**
   Replace the variable `text` with your training text:

```python
text = """Your text goes here.
It can span multiple lines."""
```

2. **Tokenize and create input sequences**

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

input_sequences = []

for sentence in text.split("\n"):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])

max_len = max([len(x) for x in input_sequences])
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]
```

3. **One-hot encode the output**

```python
from tensorflow.keras.utils import to_categorical

y = to_categorical(y, num_classes=len(tokenizer.word_index)+1)
```

4. **Build and train the model**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 100, input_length=max_len-1))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(124))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=100, verbose=1)
```

5. **Generate text**

```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

seed_text = "hello"
for _ in range(5):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            seed_text += " " + word
            break
    print(seed_text)
```

## Notes

* Ensure your training text is sufficiently large for meaningful predictions.
* `max_len` determines the sequence length fed into LSTM. Adjust it as needed.
* The vocabulary size is determined by `len(tokenizer.word_index)+1`.
* You can replace the `text` variable with your own writing, and the model will generate text in your style.
* you may your own text in the data so the model will behave how you talk and learn from your talking style
* the dataset was kept small as this is the demonstration how model works

## License

This project is licensed under the MIT License.

# LSTM Text Generation

A text generation model built with LSTM (Long Short-Term Memory) 
neural networks, trained on Shakespeare's works.

## Overview
This project trains an LSTM model to generate Shakespeare-like text 
by learning character-level patterns from the dataset.

## Dataset
- **Source:** [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- **Size:** ~1MB of Shakespeare's works

## Model Architecture
- Embedding via One-Hot Encoding
- LSTM Layer 1 (128 units)
- LSTM Layer 2 (128 units)
- Dense Output Layer (Softmax)

## Requirements
- Python 3.12
- TensorFlow
- NumPy

## Installation
```bash
pip install tensorflow numpy
```

## Usage
```bash
python lstm_text_generation.py
```

## Results

Model generates Shakespeare-like text based on seed input.
Character names like `brutus`, `angelo`, `camillo` appear naturally.

### Temperature Effect
| Temperature | Result |
|------------|--------|
| 0.5 | Conservative, repetitive but readable |
| 1.0 | Balanced mix of real and invented words |
| 1.2 | Creative but more random |

### Sample Output

**Seed: "to be or not to be" | Temperature: 0.5**
```
there i cake
dorse sen the captar was the earl die them
menenius no see it is a very part
king edward iv the lords a good sir...
```

**Seed: "to be or not to be" | Temperature: 1.0**
```
i cannot be the unking that than the whife
cominius by my deed
to pleap of shewild an estay clood...
```

**Seed: "to be or not to be" | Temperature: 1.2**
```
un ryyd hipzents my hround
sicinius mischanger stic her courthy
like mens mouth tiblay this such...
```

**Seed: "shall i compare thee" | Temperature: 0.5**
```
brutus the consent me the conting
what doth that be consent of no man
angelo how that i would the bost...
```

**Seed: "shall i compare thee" | Temperature: 1.0**
```
rivers a courssape part
and who would the estate of twenty
on as the pirscy with hold persones...
```

**Seed: "shall i compare thee" | Temperature: 1.2**
```
comaus what he by bedvo unwrich of old wises
chasterd nature i cannot they nor
my hrul marruck so that call given...
```

**Seed: "all the world is a stage" | Temperature: 0.5**
```
o the people the common here of you
first murderer the lady that that stand
that such a streed at the right and sir...
```

**Seed: "all the world is a stage" | Temperature: 1.0**
```
camillo my name sir
that many down to go
servost and this hand shall be conquer...
```

**Seed: "all the world is a stage" | Temperature: 1.2**
```
fir trevingly give a stay to tongues
palime are a write worth a frince
to dody spild to decain whencerd...
```

## Bonus: Architecture Experiments

| Configuration | Effect |
|--------------|--------|
| Single LSTM (128 units) | Faster training, less coherent output |
| Double LSTM (128+128) ✅ | Better context, more coherent text |
| Temperature 0.5 | Safe, repetitive but readable |
| Temperature 1.2 | Creative but more gibberish |

## Dataset Link
- [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
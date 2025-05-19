import torch
from DataProcessing import ReadData, GetDicts
from TextProcessor import TextProcessor
import datetime
from model_RNN import RNN
from model_lstm_torch import LSTM
import torch.nn.functional as F
import pickle




def preprocess_data():
    # Get the raw data and unique characters
    data, unique_chars = ReadData("shakespeare.txt")

    with open("dicts.pkl", "rb") as f:
        vocab = pickle.load(f)
        char2ind = vocab['char2ind']
        ind2char = vocab['ind2char']

    # Convert data to a sequence of indices
    data_indices = [char2ind[char] for char in data]

    # Convert indices to tensor
    X_input = torch.tensor(data_indices, dtype=torch.long)

    # One-hot encode the data
    X_one_hot = F.one_hot(X_input, num_classes=len(unique_chars)).float()

    return X_one_hot, unique_chars




def load_model_and_synthesize(model_path, start_char="A", seq_len=100):
    X_data, unique_chars = preprocess_data()

    with open("dicts.pkl", "rb") as f:
        vocab = pickle.load(f)
        char2ind = vocab['char2ind']
        ind2char = vocab['ind2char']

    input_size = output_size = X_data.shape[1]
    hidden_size = 256
    num_layers = 2
    seq_len = 50
    batch_size = 32

    if "lstm" in model_path.lower():
        model = LSTM(input_size=input_size, hidden_size=hidden_size,
                     output_size=output_size, unique_chars=unique_chars,
                     num_layers=num_layers, batch_size=batch_size, seq_len=seq_len)
    else:
        print("RNN identified")
        model = RNN(input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size, num_layers=num_layers)
    first = print(model.state_dict())
    model.load_state_dict(torch.load(model_path))
    second = print("Loaded model")
    print(model.state_dict())

    #compare_state_dicts(first, second)
    model.eval()


    if isinstance(model, LSTM):
        return model.synth_text(start_char, seq_len=seq_len)
    elif isinstance(model, RNN):
        print("Synthesizing")
        start_idx = char2ind[start_char]
        one_hot = F.one_hot(torch.tensor(start_idx), num_classes=input_size).float().numpy()
        return model.synthesize_text(x0=one_hot, n=seq_len, ind_to_char=ind2char, char_to_ind=char2ind)



def run_text_quality_tests(synthesized_text, processor, model_type,  output_path="text_quality_report.txt"):

    spell_score = processor.correctly_spelt_count(synthesized_text)
    two_gram_score, three_gram_score = processor.measure_n_grams(synthesized_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Text Quality Evaluation ===\n\n")
        f.write(f"Tests run at {datetime.datetime.now()}\n")
        f.write(f"Model used: {model_type}\n")
        f.write("Synthesized Text:\n")
        f.write(synthesized_text + "\n\n")
        f.write(f"Correctly Spelt Word Fraction: {spell_score:.4f}\n")
        f.write(f"2-Gram Match Fraction: {two_gram_score:.4f}\n")
        f.write(f"3-Gram Match Fraction: {three_gram_score:.4f}\n")


if __name__ == "__main__":

    print("STARTINGS")
    data_str, _ = ReadData("shakespeare.txt")
    processor = TextProcessor(data_str)


    synthesized_text =load_model_and_synthesize("best_rnn_model.pth", seq_len= 200)


    run_text_quality_tests(synthesized_text, processor, "No Model, Just Test")
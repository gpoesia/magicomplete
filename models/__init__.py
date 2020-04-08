import torch
import json

from alphabet import *
from decoder import *

def load_model(model, is_baseline=True, load_best=False):
    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    encoder, alpha, dataset, lang_name = model

    if load_best:
        # Load the best model saved during training.
        filename = "models/{}_best_".format(encoder.name())
        loss_history_filename = "models/best_model_{}".format(encoder.name())
    else:
        # Load the final model.
        filename = "models/{}_{}_{}".format(encoder.name(), alpha, lang_name)
        loss_history_filename = "models/{}_{}_{}.json".format(encoder.name(), alpha, lang_name)

    alphabet = AsciiEmbeddedEncoding(device)
    decoder = AutoCompleteDecoderModel(alphabet, hidden_size=512, copy=None)
    decoder.load_state_dict(torch.load(filename + "decoder.model", map_location=device))
    alphabet.load_state_dict(torch.load(filename + "alphabet.model", map_location=device))

    decoder.to(device)
    alphabet.to(device)

    decoder.eval()

    if not is_baseline:
        encoder.load_state_dict(torch.load(filename + "encoder.model", map_location=device))
        encoder.to(device)

    with open(loss_history_filename) as f:
        j = json.load(f)
        loss_history = j["losses"]

    return (encoder, decoder, alphabet, lang_name, loss_history)

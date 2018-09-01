import midi

# pattern = midi.read_midifile("chno0902.mid")
# print pattern
from data import trainset
from train import get_midi_batch
test = get_midi_batch(trainset).next()
print test
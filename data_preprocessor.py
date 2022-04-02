"""data_preprocessor.py:
- Parsing the mid-data
- Plotting the notes
- Visualizing the octaves with most notes
"""
import pickle
from typing import List
import music21.stream
from music21 import *
import numpy
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from mid_reader import MIDReader


class PreProcessing(MIDReader):

    def __init__(self):
        """
        - Super class constructor call
        - Get parsed MIDs from the pickled files
        """
        super().__init__()
        with open(self.pic_name, 'rb') as f:
            self.parsed_midis = pickle.load(f)

    def notes_extraction_mid(self):
        """
        - Extracting notes from all the mid-files
        - If there are any tie notes, strip them and convert them into single notes with duration as the
        sum of the tied notes
        """
        parent_note = []
        c_note = []
        for p_mid_itr in self.parsed_midis:
            p_mid_itr = p_mid_itr.stripTies()
            """
            - Partition by instrument in the given mid files
            """
            instrument_div = instrument.partitionByInstrument(p_mid_itr)
            """
            - Iterate over all the Instruments
            """
            for instru_itr in instrument_div.parts:
                re_itr = instru_itr.recurse()
                """
                - Iterating over the notes of the given instrument
                """
                for itr in re_itr:
                    """
                    - If there are still some tie notes that were unresolved by tie.stripTie(), ignore them
                    """
                    if (isinstance(itr, note.Note) or isinstance(itr, chord.Chord)) and not itr.tie:
                        if isinstance(itr, note.Note):
                            c_note.append(max(0.0, itr.pitch.ps))
                            parent_note.append(itr)
                        elif isinstance(itr, chord.Chord):
                            parent_note.append(itr)
                            for single_pitch in itr.pitches:
                                c_note.append(max(0.0, single_pitch.ps))
                        else:
                            pass
        return c_note, parent_note

    @staticmethod
    def notes_extraction_instrument(parts: music21.stream.iterator.StreamIterator):
        """
        - Extracting notes from just the given instrument
        """
        parent_note = []
        c_note = []
        for itr in parts:
            if isinstance(itr, note.Note):
                c_note.append(max(0.0, itr.pitch.ps))
                parent_note.append(itr)
            elif isinstance(itr, chord.Chord):
                for pi in itr.pitches:
                    c_note.append(max(0.0, pi.ps))
                parent_note.append(itr)
            else:
                pass
        return c_note, parent_note

    @staticmethod
    def perform_end_note_correction(all_notes: List) -> None:
        for index, single_note in enumerate(all_notes):
            if not single_note.volume.velocity or not single_note.duration.quarterLength:
                all_notes.pop(index)
        return

    @staticmethod
    def perform_input_encoding(all_notes):
        notes_string = []
        for single_note_chord in all_notes:
            if isinstance(single_note_chord, music21.note.Note):
                if single_note_chord.nameWithOctave:
                    notes_string.append(single_note_chord.nameWithOctave)
            if isinstance(single_note_chord, chord.Chord):
                notes_string.append(','.join(note_itr.nameWithOctave for note_itr in single_note_chord.notes))

        """
        Get the string version of all the distinct notes and chords and store it in a set
        """
        distinct_notes_chords = set(notes_string)

        # create a dictionary of integers corresponding to the distinct notes and chords
        note_int_mapping = dict((note_itr, index) for index, note_itr in enumerate(distinct_notes_chords))
        return note_int_mapping, notes_string, distinct_notes_chords
        # get a list of all the values in the dictionary
        # mapped_values = list(note_int_mapping.values())

        # # one-hot encode the above dictionary
        # one_hot_encoded = to_categorical(mapped_values)
        # return one_hot_encoded

    @staticmethod
    def select_input_features(all_notes_string, integer_mapped_input):
        # select an input length
        input_length = 20
        x = []
        labels = []
        for notes_itr in range(0, len(all_notes_string) - input_length):
            input_as_strChars = all_notes_string[notes_itr:notes_itr + input_length]
            output_as_strChars = all_notes_string[notes_itr + input_length]
            x.append([integer_mapped_input[char] for char in input_as_strChars])
            labels.append(integer_mapped_input[output_as_strChars])

        # Reshape the input to 3D for it to be compatible with LSTM
        x = numpy.reshape(x, (len(x), input_length, 1))

        # normalize the input
        x = numpy.array(x) / float(len(integer_mapped_input))

        # one hot encode the output
        labels = to_categorical(labels)
        return x, labels

    @staticmethod
    def data_split(x, labels):
        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.15, random_state=42, shuffle=False)
        return x_train, x_test, y_train, y_test

""" mid_reader.py
-
"""
import pickle
from music21 import *
import os
from music21.converter import ConverterFileException
from music21.exceptions21 import StreamException


class MIDReader:
    def __init__(self):
        """
        - Constructor for MIDI Reader
        - Initialise path for mid-files directory
        - Initialise list for saving parsed mid - files
        - Initialise count for showing the corrupt files
        - Name of the pickle that will store the parsed MIDs for further usage
        """
        self.mid_files_path: str = "./midi_songs/"
        self.parsed_mid = []
        self.files_skipped: int = 0
        self.pic_name = 'parsed_midis.pkl'

    def parse_mid(self):
        """
        - Iterate through mid-files parsing data streams through them
        - Use music21's converter which is used to parse music from all kinds of files parses the data as a stream
        - If there are any tie notes, strip them and convert them into single notes with duration as the
        sum of the tied notes
        """
        for file_itr in os.listdir(self.mid_files_path):
            try:
                mid_itr = converter.parse(self.mid_files_path + file_itr)
                mid_itr.stripTies(inPlace=True)
                self.parsed_mid.append(mid_itr)

            except StreamException:
                # Skip the file when it cannot be parsed to a stream
                self.files_skipped += 1

            except ConverterFileException:
                # Skip all the other invalid music files
                pass
        print("<<<< MID Files parsed >>>>")

    def pickle_parsed_music(self):
        """
        - Save the parsed MIDI files with pickling to retrieve easily for further usage and avoid repetitive reading
        """
        with open(self.pic_name, 'wb') as f:
            pickle.dump(self.parsed_mid, f)
        print("<<<< Parsed MID's pickled, to be used later >>>>")

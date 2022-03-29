import pickle
from music21 import *
import os
from music21.converter import ConverterFileException
from music21.exceptions21 import StreamException


class MIDIReader:
    def __init__(self):
        """
        Constructor for MIDI Reader
        """
        # Initialise path for mid-files directory
        self.mid_files_path: str = "./midi_songs/"
        # Initialise list for saving parsed mid - files
        self.parsed_mid = []
        # Initialise count for showing the corrupt files
        self.files_skipped: int = 0
        self.pic_name = 'parsed_midis.pkl'

    def parse_mid(self):
        """
        Iterate through mid-files parsing data streams through them
        """
        for file_itr in os.listdir(self.mid_files_path):
            try:
                # Use music21's converter which is used to parse music from all kinds of files
                # Parses the data as a stream
                mid_itr = converter.parse(self.mid_files_path + file_itr)

                # If there are any tie notes, strip them and convert them into single notes with duration as the
                # sum of the tied notes
                mid_itr.stripTies(inPlace=True)

                self.parsed_mid.append(mid_itr)
            except StreamException:
                # Skip the file when it cannot be parsed to a stream
                self.files_skipped += 1
            except ConverterFileException:
                # Skip all the other invalid music files
                pass
        print("MID Files parsed................")

    def pickle_parsed_music(self):
        """
        Save the parsed MIDI files with pickling to retrieve easily for further usage and avoid repetitive reading
        """
        with open(self.pic_name, 'wb') as f:
            pickle.dump(self.parsed_mid, f)

        print("Parsed MID's pickled, to be used later............")

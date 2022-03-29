import os

import music21
from mido import MidiFile
from music21 import converter, instrument, note, chord
from typing import List
from music21.converter import ConverterFileException
from music21.exceptions21 import StreamException

mid_files_path: str = "./midi_songs/"
files_skipped: int = 0


class DataParser:
    def __init__(self, parsing_type: str):
        if parsing_type == 'music21':
            self.parsed_mid_music21: List[music21.stream.base.Score] = []
            self.__parse_mid()
        elif parsing_type == 'mido':
            self.parsed_mid_mido = []
            self.__mido_parser()

    def __parse_mid(self):
        """
        Iterate through mid-files parsing data streams through them
        """
        for file_itr in os.listdir(mid_files_path):
            try:
                # Use music21's converter which is used to parse music from all kinds of files
                # Parses the data as a stream
                mid_itr = converter.parse(mid_files_path + file_itr)
                self.parsed_mid_music21.append(mid_itr)
            except StreamException:
                # Skip the file when it cannot be parsed to a stream
                global files_skipped
                files_skipped += 1
            except ConverterFileException:
                # Skip all the other invalid music files
                pass

    def raw_data_exporter(self):
        all_notes = []
        for p_mid_itr in self.parsed_mid_music21:
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
                    if isinstance(itr, note.Note):
                        all_notes.append(str(itr.pitch))
                    elif isinstance(itr, chord.Chord):
                        all_notes.append('.'.join(str(n) for n in itr.normalOrder))
                    else:
                        pass

    def __mido_parser(self):
        for file_itr in os.listdir(mid_files_path):
            mid = MidiFile(mid_files_path + file_itr, clip=True)
            self.parsed_mid_mido.append(mid)
            print(type(mid))
            print("OK")

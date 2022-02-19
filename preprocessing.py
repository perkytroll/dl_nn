"""

"""
import collections
from typing import List
import music21.stream
from music21 import *
import os
from music21.converter import ConverterFileException
from music21.exceptions21 import StreamException
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

"""
Create a list of mid files for data extraction
"""
mid_files_path: str = "./midi_songs/"
parsed_mid: List[music21.stream.base.Score] = []
files_skipped: int = 0


def parse_mid():
    """
    Iterate through mid files parsing data streams through them
    """
    for file_itr in os.listdir(mid_files_path):
        try:
            # Use music21's converter which is used to parse music from all kinds of files
            # Parses the data as a stream
            mid_itr = converter.parse(mid_files_path + file_itr)
            parsed_mid.append(mid_itr)
        except StreamException:
            # Skip the file when it cannot be parsed to a stream
            global files_skipped
            files_skipped += 1
        except ConverterFileException:
            # Skip all the other invalid music files
            pass


def notes_extraction_mid(p_mid: List[music21.stream.Score]):
    parent_note = []
    c_note = []
    for p_mid_itr in p_mid:
        instrument_div = instrument.partitionByInstrument(p_mid_itr)
        for instru_itr in instrument_div.parts:
            re_itr = instru_itr.recurse()
            for itr in re_itr:
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


def notes_extraction_instrument(parts: music21.stream.iterator.StreamIterator):
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


def plot_parts(p_mid):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minPitch = pitch.Pitch('C10').ps
    maxPitch = 0
    xMax = 0

    for p_mid_itr in p_mid:
        instrument_div = instrument.partitionByInstrument(p_mid_itr)
        for instru_itr in range(len(instrument_div.parts)):
            top = instrument_div.parts[instru_itr].flat.notes
            y, parent_element = notes_extraction_instrument(top)
            if len(y) < 1:
                continue

            x = [n.offset for n in parent_element]
            ax.scatter(x, y, alpha=0.6, s=7)

            aux = min(y)
            if aux < minPitch:
                minPitch = aux

            aux = max(y)
            if aux > maxPitch:
                maxPitch = aux

            aux = max(x)
            if aux > xMax:
                xMax = aux

        for i in range(1, 10):
            linePitch = pitch.Pitch('C{0}'.format(i)).ps
            if minPitch < linePitch < maxPitch:
                ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))

    plt.ylabel("Note index (each octave has 12 notes)")
    plt.xlabel("Number of quarter notes (beats)")
    plt.title('Instrument in each Octave')
    plt.show()


parse_mid()
temp, parent_n = all_notes = notes_extraction_mid(parsed_mid)
plot_parts(parsed_mid)
print((files_skipped/(len(parsed_mid) + files_skipped) * 100), "percent of files have been skipped due to corrupt data")
print("Done")

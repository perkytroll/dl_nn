import collections
from typing import List

import matplotlib.pyplot as mpl
import music21
from matplotlib import pyplot as plt
from music21 import instrument, pitch


class DataVisualize:
    def __init__(self, plot_type):
        self.plot_type = plot_type
        self.fig: mpl.Figure = mpl.figure()
        self.axes = self.fig.add_axes([0, 0, 1, 1])

    def bar_plot(self, x, y, xlabel: str, ylabel: str, title: str, legend: list):
        self.axes.bar(x, y)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xlabel(xlabel)
        self.axes.set_title(title)
        self.axes.legend(labels=legend)

        def count_plots(p_mid: List[music21.stream.base.Score]):
            """
            - List of all the instruments being played
            - Bar plot for number of notes in each octave
            """
            pitches_per_octaves = {}
            pitch_class_counts = {}
            instruments = []
            for p_mid_itr in p_mid:
                this_instruments = instrument.partitionByInstrument(p_mid_itr)
                for itr_instru in this_instruments:
                    if itr_instru.partName in instruments:
                        pass
                    else:
                        instruments.append(itr_instru.partName)

                pitch_class_counts_this = collections.Counter(p_mid_itr.pitches)
                for p_key in pitch_class_counts_this.keys():
                    if p_key in pitch_class_counts:
                        pitch_class_counts[p_key] = pitch_class_counts[p_key] + pitch_class_counts_this[p_key]
                    else:
                        pitch_class_counts[p_key] = pitch_class_counts_this[p_key]

                for pit_itr in pitch_class_counts:
                    if pit_itr.octave in pitches_per_octaves:
                        pitches_per_octaves[pit_itr.octave] = pitches_per_octaves[pit_itr.octave] + 1
                    else:
                        pitches_per_octaves[pit_itr.octave] = 1
            plt.bar(pitches_per_octaves.keys(), pitches_per_octaves.values(), align='center')
            plt.xlabel('Octaves')
            plt.ylabel('Note Counts')
            plt.show()

        def plot_quarter_notes(p_mid):
            """
            - Plot quarter notes/beats on the note counts
            """
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(1, 1, 1)
            minPitch = pitch.Pitch('C10').ps
            maxPitch = 0
            xMax = 0

            for p_mid_itr in p_mid:
                instrument_div = instrument.partitionByInstrument(p_mid_itr)
                for instru_itr in range(len(instrument_div.parts)):
                    top = instrument_div.parts[instru_itr].flat.notes
                    # y, parent_element = notes_extraction_instrument(top)
                    if len(y) < 1:
                        continue

                    # x = [n.offset for n in parent_element]
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
                        pass
                        # ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))

            plt.ylabel("Note index")
            plt.xlabel("Quarter notes (beats)")
            plt.title('Each color is a different instrument, red lines show each octave')
            plt.show()

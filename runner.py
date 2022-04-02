from data_preprocessor import PreProcessing
from mid_reader import MIDReader
from model_training import ModelTraining
from music_generation import MusicGeneration

# reader = MIDReader()
# reader.parse_mid()
# reader.pickle_parsed_music()

"""
Pre-Processing Data
"""
pre_processing = PreProcessing()
c_note, parent_note = pre_processing.notes_extraction_mid()
pre_processing.perform_end_note_correction(parent_note)
integer_mapped_input, all_notes_string, distinct_notes_chords = pre_processing.perform_input_encoding(parent_note)
input_seq, output_vals = pre_processing.select_input_features(all_notes_string, integer_mapped_input)

"""
Splitting data
"""
x_train, x_test, y_train, y_test = pre_processing.data_split(input_seq, output_vals)

"""
Training Model
"""
model_training = ModelTraining(x_train, x_test, y_train, y_test)
model_architecture = model_training.build_architecture()
compiled_model = model_training.compile_model(model_architecture)
history, trained_model = model_training.model_training(compiled_model)
model_training.plot_training_results(history)

"""
Generation of Music
"""
gen_model = model_training.build_architecture()
weight_updated_model = model_training.weight_updated_model(gen_model)

gen_notes = MusicGeneration(distinct_notes_chords, integer_mapped_input, 500, x_train, x_test, y_train, y_test)
gen_notes.generate_new_notes(weight_updated_model)
melody_notes = gen_notes.decoder()
gen_notes.create_output_file(melody_notes)
print("OK")

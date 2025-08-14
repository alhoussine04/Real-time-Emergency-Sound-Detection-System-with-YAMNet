"""
YAMNet TF-Lite model wrapper for audio classification.
"""

import numpy as np
import zipfile
import logging
import os
from typing import List, Tuple, Optional

# Try to import TensorFlow Lite interpreters in order of preference
INTERPRETER_TYPE = None
Interpreter = None

try:
    # First try TensorFlow Lite Runtime (for Raspberry Pi)
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    INTERPRETER_TYPE = "tflite_runtime"
except ImportError:
    try:
        # Then try AI Edge LiteRT (Google's new runtime)
        from ai_edge_litert.interpreter import Interpreter
        INTERPRETER_TYPE = "ai_edge_litert"
    except ImportError:
        try:
            # Finally fallback to full TensorFlow
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            INTERPRETER_TYPE = "tensorflow"
        except ImportError:
            raise ImportError(
                "No TensorFlow Lite interpreter found. Please install one of:\n"
                "  - pip install tflite-runtime (Raspberry Pi)\n"
                "  - pip install tensorflow (PC)\n"
                "  - pip install ai-edge-litert (Future)"
            )


class YAMNetClassifier:
    """
    Wrapper class for YAMNet TF-Lite model to classify audio events.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the YAMNet classifier.
        
        Args:
            model_path: Path to the YAMNet TF-Lite model file
        """
        self.model_path = model_path
        self.interpreter = None
        self.labels = []
        self.input_index = None
        self.output_index = None
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
        self._load_labels()
    
    def _load_model(self) -> None:
        """Load the TF-Lite model and get input/output details."""
        try:
            # Initialize interpreter based on available runtime
            self.interpreter = Interpreter(model_path=self.model_path)
            self.logger.info(f"Using {INTERPRETER_TYPE}")

            self.interpreter.allocate_tensors()

            # Get input and output details
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            self.input_index = input_details[0]['index']
            self.output_index = output_details[0]['index']

            self.logger.info(f"YAMNet model loaded successfully from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load YAMNet model: {e}")
            raise
    
    def _load_labels(self) -> None:
        """Load class labels from the model file or external file."""
        # Try to load from embedded labels first
        try:
            with zipfile.ZipFile(self.model_path, 'r') as model_zip:
                labels_file = model_zip.open('yamnet_label_list.txt')
                self.labels = [
                    line.decode('utf-8').strip()
                    for line in labels_file.readlines()
                ]

            self.logger.info(f"Loaded {len(self.labels)} class labels from model file")
            return

        except Exception as e:
            self.logger.debug(f"Could not load labels from model file: {e}")

        # Try to load from external label file
        label_files = ['yamnet_labels.txt', 'yamnet_label_list.txt']
        for label_file in label_files:
            try:
                if os.path.exists(label_file):
                    with open(label_file, 'r', encoding='utf-8') as f:
                        self.labels = [line.strip() for line in f.readlines() if line.strip()]

                    self.logger.info(f"Loaded {len(self.labels)} class labels from {label_file}")
                    return

            except Exception as e:
                self.logger.debug(f"Could not load labels from {label_file}: {e}")

        # Fallback: create default labels
        self.logger.warning("Could not load labels, using default AudioSet classes")
        self.labels = self._get_default_labels()
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for YAMNet input.
        
        Args:
            audio_data: Raw audio data as numpy array
            
        Returns:
            Preprocessed audio data ready for model input
        """
        # Ensure audio is float32 and in range [-1.0, 1.0]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1.0, 1.0] if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Ensure exactly 15600 samples (0.975 seconds at 16kHz)
        target_length = 15600
        if len(audio_data) > target_length:
            # Truncate if too long
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # Pad with zeros if too short
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        return audio_data
    
    def classify(self, audio_data: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Classify audio data and return top predictions.
        
        Args:
            audio_data: Audio waveform as numpy array
            
        Returns:
            Tuple of (class_names, confidence_scores) for top predictions
        """
        try:
            # Validate input
            if not isinstance(audio_data, np.ndarray):
                raise ValueError("Input must be a numpy array")
            
            if audio_data.size == 0:
                raise ValueError("Input audio array is empty")
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data)

            # Set input tensor (YAMNet expects 1D input)
            self.interpreter.set_tensor(self.input_index, processed_audio)

            # Run inference
            self.interpreter.invoke()
            
            # Get output scores
            scores = self.interpreter.get_tensor(self.output_index)
            scores = scores.flatten()  # Shape: (521,)
            
            # Get top 10 predictions
            top_indices = np.argsort(scores)[-10:][::-1]
            top_classes = [self.labels[i] for i in top_indices]
            top_scores = [float(scores[i]) for i in top_indices]
            
            return top_classes, top_scores
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return [], []
    
    def get_class_score(self, audio_data: np.ndarray, target_class: str) -> float:
        """
        Get confidence score for a specific class.
        
        Args:
            audio_data: Audio waveform as numpy array
            target_class: Name of the target class
            
        Returns:
            Confidence score for the target class
        """
        try:
            # Find class index
            if target_class not in self.labels:
                self.logger.warning(f"Class '{target_class}' not found in labels")
                return 0.0
            
            class_index = self.labels.index(target_class)
            
            # Preprocess and run inference
            processed_audio = self.preprocess_audio(audio_data)

            self.interpreter.set_tensor(self.input_index, processed_audio)
            self.interpreter.invoke()
            
            scores = self.interpreter.get_tensor(self.output_index)
            return float(scores[0][class_index])
            
        except Exception as e:
            self.logger.error(f"Failed to get class score: {e}")
            return 0.0
    
    def get_available_classes(self) -> List[str]:
        """Get list of all available class names."""
        return self.labels.copy()

    def _get_default_labels(self) -> List[str]:
        """Get default AudioSet class labels for YAMNet."""
        return [
            "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
            "Conversation", "Narration, monologue", "Babbling", "Speech synthesizer", "Shout", "Bellow",
            "Whoop", "Yell", "Battle cry", "Children shouting", "Screaming", "Whispering", "Laughter",
            "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Crying, sobbing",
            "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh", "Singing", "Choir", "Yodeling",
            "Chant", "Mantra", "Male singing", "Female singing", "Child singing", "Synthetic singing",
            "Rapping", "Humming", "Groan", "Grunt", "Whistling", "Breathing", "Wheeze", "Snoring",
            "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff", "Run", "Shuffle",
            "Walk, footsteps", "Chewing, mastication", "Biting", "Gargling", "Stomach rumble", "Burping, eructation",
            "Hiccup", "Fart", "Hands", "Finger snapping", "Clapping", "Heart sounds, heartbeat", "Heart murmur",
            "Cheering", "Applause", "Chatter", "Crowd", "Hubbub, speech noise, speech babble", "Children playing",
            "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling",
            "Whimper (dog)", "Cat", "Purring", "Meow", "Hiss", "Caterwaul", "Livestock, farm animals, working animals",
            "Horse", "Clip-clop", "Neigh, whinny", "Cattle, bovinae", "Moo", "Cowbell", "Pig", "Oink",
            "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck", "Crowing, cock-a-doodle-doo",
            "Turkey", "Gobble", "Duck", "Quack", "Goose", "Honk", "Wild animals", "Roaring cats (lions, tigers)",
            "Roar", "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Squawk", "Pigeon, dove",
            "Coo", "Crow", "Caw", "Owl", "Hoot", "Bird flight, flapping wings", "Canidae, dogs, wolves",
            "Rodents, rats, mice", "Mouse", "Patter", "Insect", "Cricket", "Mosquito", "Fly, housefly",
            "Buzz", "Bee, wasp, etc.", "Frog", "Croak", "Snake", "Rattle", "Whale vocalization", "Music",
            "Musical instrument", "Plucked string instrument", "Guitar", "Electric guitar", "Bass guitar",
            "Acoustic guitar", "Steel guitar, slide guitar", "Tapping (guitar technique)", "Strum", "Banjo",
            "Sitar", "Mandolin", "Zither", "Ukulele", "Keyboard (musical)", "Piano", "Electric piano",
            "Organ", "Electronic organ", "Hammond organ", "Synthesizer", "Sampler", "Harpsichord", "Percussion",
            "Drum kit", "Drum machine", "Drum", "Snare drum", "Rimshot", "Drum roll", "Bass drum", "Timpani",
            "Tabla", "Cymbal", "Hi-hat", "Wood block", "Tambourine", "Rattle (instrument)", "Maraca",
            "Gong", "Tubular bells", "Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone",
            "Steelpan", "Orchestra", "Brass instrument", "French horn", "Trumpet", "Trombone", "Bugle",
            "Cornet", "Saxophone", "Clarinet", "Flute", "Whistle", "Kazoo", "Recorder", "Bagpipes",
            "Didgeridoo", "Shofar", "Theremin", "Singing bowl", "Scratching (performance technique)", "Pop music",
            "Hip hop music", "Beatboxing", "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock",
            "Rock and roll", "Psychedelic rock", "Rhythm and blues", "Soul music", "Reggae", "Country",
            "Swing music", "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz", "Disco",
            "Classical music", "Opera", "Electronic music", "House music", "Techno", "Dubstep", "Drum and bass",
            "Electronica", "Electronic dance music", "Ambient music", "Trance music", "Music of Latin America",
            "Salsa music", "Flamenco", "Blues", "Music for children", "New-age music", "Vocal music",
            "A capella", "Music of Africa", "Afrobeat", "Christian music", "Gospel music", "Music of Asia",
            "Carnatic music", "Music of Bollywood", "Ska", "Traditional music", "Independent music", "Song",
            "Background music", "Theme music", "Jingle (music)", "Soundtrack music", "Lullaby", "Video game music",
            "Christmas music", "Dance music", "Wedding music", "Happy music", "Funny music", "Sad music",
            "Tender music", "Exciting music", "Angry music", "Scary music", "Wind instrument, woodwind instrument",
            "Flute", "Saxophone", "Clarinet", "Harp", "Bell", "Church bell", "Jingle bell", "Bicycle bell",
            "Tuning fork", "Chime", "Wind chime", "Change ringing (campanology)", "Harmonica", "Accordion",
            "Bagpipes", "Didgeridoo", "Shofar", "Theremin", "Singing bowl", "Scratching (performance technique)",
            "Silence", "Noise", "White noise", "Pink noise", "Throbbing", "Vibration", "Television", "Radio",
            "Field recording", "Boat, Water vehicle", "Sailboat, sailing ship", "Rowboat, canoe, kayak",
            "Motorboat, speedboat", "Ship", "Motor vehicle (road)", "Car", "Vehicle horn, car horn, honking",
            "Toot", "Car alarm", "Power windows, electric windows", "Skidding", "Tire squeal", "Car passing by",
            "Race car, auto racing", "Truck", "Air brake", "Air horn, truck horn", "Reversing beeps",
            "Ice cream truck, ice cream van", "Bus", "Emergency vehicle", "Police car (siren)", "Ambulance (siren)",
            "Fire engine, fire truck (siren)", "Motorcycle", "Traffic noise, roadway noise", "Rail transport",
            "Train", "Train whistle", "Train horn", "Railroad car, train wagon", "Train wheels squealing",
            "Subway, metro, underground", "Aircraft", "Aircraft engine", "Jet engine", "Propeller, airscrew",
            "Helicopter", "Fixed-wing aircraft, airplane", "Bicycle", "Skateboard", "Engine", "Light engine (high frequency)",
            "Dental drill, dentist's drill", "Lawn mower", "Chainsaw", "Medium engine (mid frequency)",
            "Heavy engine (low frequency)", "Engine knocking", "Engine starting", "Idling", "Accelerating, revving, vroom",
            "Door", "Doorbell", "Ding-dong", "Sliding door", "Slam", "Knock", "Tap", "Squeak",
            "Cupboard open or close", "Drawer open or close", "Dishes, pots, and pans", "Cutlery, silverware",
            "Chopping (food)", "Frying (food)", "Microwave oven", "Blender", "Water tap, faucet",
            "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer", "Toilet flush",
            "Toothbrush", "Electric toothbrush", "Vacuum cleaner", "Zipper (clothing)", "Keys jangling",
            "Coin (dropping)", "Scissors", "Electric shaver, electric razor", "Shuffling cards", "Typing",
            "Typewriter", "Computer keyboard", "Writing", "Alarm", "Smoke detector, smoke alarm", "Fire alarm",
            "Foghorn", "Whistle", "Steam whistle", "Mechanisms", "Ratchet, pawl", "Clock", "Tick",
            "Tick-tock", "Gears", "Pulleys", "Sewing machine", "Mechanical fan", "Air conditioning",
            "Cash register", "Printer", "Camera", "Single-lens reflex camera", "Tools", "Hammer",
            "Jackhammer", "Sawing", "Filing (rasp)", "Sanding", "Power tool", "Drill", "Explosion",
            "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun", "Fireworks",
            "Firecracker", "Burst, pop", "Eruption", "Boom", "Wood", "Chop", "Splinter", "Crack",
            "Glass", "Chink, clink", "Shatter", "Liquid", "Splash, splatter", "Slosh", "Squish",
            "Drip", "Pour", "Trickle, dribble", "Gush", "Fill (with liquid)", "Spray", "Pump (liquid)",
            "Stir", "Boiling", "Sonar", "Arrow", "Whoosh, swoosh, swish", "Thump, thud", "Thunk",
            "Electronic tuner", "Effects unit", "Chorus effect", "Basketball bounce", "Bang", "Beep, bleep",
            "Sine wave", "Chirp tone", "Sound effect", "Pulse", "Inside, small room", "Inside, large room or hall",
            "Inside, public space", "Outside, urban or manmade", "Outside, rural or natural", "Reverberation",
            "Echo", "Noise", "Environmental noise", "Static", "Mains hum", "Distortion", "Sidetone",
            "Cacophony", "White noise", "Pink noise", "Throbbing", "Vibration", "Television", "Radio", "Field recording"
        ][:521]  # Ensure exactly 521 classes

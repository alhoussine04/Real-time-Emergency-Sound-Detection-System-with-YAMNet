#!/usr/bin/env python3
"""
Download YAMNet TF-Lite model with labels from TensorFlow Hub.
"""

import os
import urllib.request
import zipfile
import tempfile

def download_yamnet_model():
    """Download YAMNet TF-Lite model."""
    
    # YAMNet TF-Lite model URL from TensorFlow Hub
    model_url = "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite"
    model_path = "yamnet.tflite"
    
    print("Downloading YAMNet TF-Lite model...")
    print(f"URL: {model_url}")
    
    try:
        # Download the model
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✅ Model downloaded: {model_path}")
        
        # Check file size
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"📏 File size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def create_label_file():
    """Create YAMNet label file."""
    
    # YAMNet class names (AudioSet ontology)
    # This is a subset of the most common classes
    yamnet_labels = [
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
        "Pop music", "Hip hop music", "Beatboxing", "Rock music", "Heavy metal", "Punk rock", "Grunge",
        "Progressive rock", "Rock and roll", "Psychedelic rock", "Rhythm and blues", "Soul music", "Reggae",
        "Country", "Swing music", "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz",
        "Disco", "Classical music", "Opera", "Electronic music", "House music", "Techno", "Dubstep",
        "Drum and bass", "Electronica", "Electronic dance music", "Ambient music", "Trance music",
        "Music of Latin America", "Salsa music", "Flamenco", "Blues", "Music for children", "New-age music",
        "Vocal music", "A capella", "Music of Africa", "Afrobeat", "Christian music", "Gospel music",
        "Music of Asia", "Carnatic music", "Music of Bollywood", "Ska", "Traditional music", "Independent music",
        "Song", "Background music", "Theme music", "Jingle (music)", "Soundtrack music", "Lullaby",
        "Video game music", "Christmas music", "Dance music", "Wedding music", "Happy music", "Funny music",
        "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music", "Silence", "Noise",
        "White noise", "Pink noise", "Throbbing", "Vibration", "Television", "Radio", "Field recording",
        "Boat, Water vehicle", "Sailboat, sailing ship", "Rowboat, canoe, kayak", "Motorboat, speedboat",
        "Ship", "Motor vehicle (road)", "Car", "Vehicle horn, car horn, honking", "Toot", "Car alarm",
        "Power windows, electric windows", "Skidding", "Tire squeal", "Car passing by", "Race car, auto racing",
        "Truck", "Air brake", "Air horn, truck horn", "Reversing beeps", "Ice cream truck, ice cream van",
        "Bus", "Emergency vehicle", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
        "Motorcycle", "Traffic noise, roadway noise", "Rail transport", "Train", "Train whistle", "Train horn",
        "Railroad car, train wagon", "Train wheels squealing", "Subway, metro, underground", "Aircraft",
        "Aircraft engine", "Jet engine", "Propeller, airscrew", "Helicopter", "Fixed-wing aircraft, airplane",
        "Bicycle", "Skateboard", "Engine", "Light engine (high frequency)", "Dental drill, dentist's drill",
        "Lawn mower", "Chainsaw", "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Engine knocking",
        "Engine starting", "Idling", "Accelerating, revving, vroom", "Door", "Doorbell", "Ding-dong",
        "Sliding door", "Slam", "Knock", "Tap", "Squeak", "Cupboard open or close", "Drawer open or close",
        "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)", "Frying (food)", "Microwave oven",
        "Blender", "Water tap, faucet", "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer",
        "Toilet flush", "Toothbrush", "Electric toothbrush", "Vacuum cleaner", "Zipper (clothing)",
        "Keys jangling", "Coin (dropping)", "Scissors", "Electric shaver, electric razor", "Shuffling cards",
        "Typing", "Typewriter", "Computer keyboard", "Writing", "Alarm", "Smoke detector, smoke alarm",
        "Fire alarm", "Foghorn", "Whistle", "Steam whistle", "Mechanisms", "Ratchet, pawl", "Clock",
        "Tick", "Tick-tock", "Gears", "Pulleys", "Sewing machine", "Mechanical fan", "Air conditioning",
        "Cash register", "Printer", "Camera", "Single-lens reflex camera", "Tools", "Hammer", "Jackhammer",
        "Sawing", "Filing (rasp)", "Sanding", "Power tool", "Drill", "Explosion", "Gunshot, gunfire",
        "Machine gun", "Fusillade", "Artillery fire", "Cap gun", "Fireworks", "Firecracker", "Burst, pop",
        "Eruption", "Boom", "Wood", "Chop", "Splinter", "Crack", "Glass", "Chink, clink", "Shatter",
        "Liquid", "Splash, splatter", "Slosh", "Squish", "Drip", "Pour", "Trickle, dribble", "Gush",
        "Fill (with liquid)", "Spray", "Pump (liquid)", "Stir", "Boiling", "Sonar", "Arrow", "Whoosh, swoosh, swish",
        "Thump, thud", "Thunk", "Electronic tuner", "Effects unit", "Chorus effect", "Basketball bounce",
        "Bang", "Beep, bleep", "Sine wave", "Chirp tone", "Sound effect", "Pulse", "Inside, small room",
        "Inside, large room or hall", "Inside, public space", "Outside, urban or manmade", "Outside, rural or natural",
        "Reverberation", "Echo", "Noise", "Environmental noise", "Static", "Mains hum", "Distortion",
        "Sidetone", "Cacophony", "White noise", "Pink noise", "Throbbing", "Vibration", "Television",
        "Radio", "Field recording"
    ]
    
    # Write labels to file
    label_file = "yamnet_labels.txt"
    with open(label_file, 'w') as f:
        for label in yamnet_labels:
            f.write(label + '\n')
    
    print(f"✅ Created label file: {label_file} ({len(yamnet_labels)} labels)")
    return True

def main():
    """Main function."""
    print("YAMNet Model Downloader")
    print("=" * 30)
    
    # Check if model already exists
    if os.path.exists("1.tflite"):
        print("ℹ️ Model file '1.tflite' already exists")
        choice = input("Download new model? (y/N): ").lower()
        if choice != 'y':
            print("Keeping existing model")
            return
    
    # Download model
    if download_yamnet_model():
        # Rename to expected filename
        if os.path.exists("yamnet.tflite"):
            if os.path.exists("1.tflite"):
                os.remove("1.tflite")
            os.rename("yamnet.tflite", "1.tflite")
            print("✅ Model renamed to '1.tflite'")
    
    # Create label file
    create_label_file()
    
    print("\n🎉 Setup complete!")
    print("Next steps:")
    print("1. Run: python quick_test.py")
    print("2. Configure Telegram in config.env")
    print("3. Run: python main.py")

if __name__ == "__main__":
    main()

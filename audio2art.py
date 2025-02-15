import whisper
import os
import subprocess
import torch
import warnings
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline

warnings.filterwarnings("ignore", category=UserWarning)


# Load environment variables
load_dotenv()

# Load Whisper model (explicitly using CPU)
model = whisper.load_model("base", device="cpu")

# Load Stable Diffusion model (Use "cuda" for GPU, else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Function to ensure audio file is in the correct format
def convert_wav_to_standard_format(input_path, output_path):
    try:
        print(f"Converting {input_path} to standard WAV format...")
        
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",  # Convert to mono
            "-ar", "16000",  # Set sample rate to 16kHz
            "-sample_fmt", "s16",  # Use 16-bit PCM
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Conversion successful: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting WAV file: {e}")
        return None

# Function to convert speech to text using Whisper
def audio_to_text(audio_path):
    try:
        print(f"Processing audio file: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Error: Audio file not found at {audio_path}")

        # Convert to a standard format before processing
        converted_path = "converted_temp.wav"
        standard_audio_path = convert_wav_to_standard_format(audio_path, converted_path)
        
        if not standard_audio_path:
            raise RuntimeError("Audio conversion failed.")

        # Load and transcribe audio
        audio = whisper.load_audio(standard_audio_path)
        print("Audio loaded successfully!")
        
        result = model.transcribe(audio)
        text = result.get('text', '').strip()

        # Cleanup temporary file
        os.remove(converted_path)

        return text
    except Exception as e:
        print(f"Error in audio_to_text: {e}")
        return None

# Function to generate an image from text using Stable Diffusion
def generate_image_from_text(text_prompt):
    try:
        print("Generating image based on the extracted text...")

        # Generate image
        image = pipe(text_prompt).images[0]
        image_path = "generated_image.png"
        image.save(image_path)

        print(f"Image generated successfully: {image_path}")
        return image_path
    except Exception as e:
        print(f"Error in generate_image_from_text: {e}")
        return None

# Main function: Convert audio to text, then generate an image from text
def audio_to_art(audio_path):
    print("Starting the audio-to-art process...")
    
    # Step 1: Convert Audio to Text
    text = audio_to_text(audio_path)
    if text:
        print(f"Extracted Text: {text}")

        # Step 2: Generate Image from Text
        image_path = generate_image_from_text(text)
        if image_path:
            print(f"Art generated successfully! Saved at: {image_path}")
            return image_path
        else:
            print("Failed to generate art.")
    else:
        print("Failed to convert audio to text.")

# Example usage
if __name__ == "__main__":
    # Provide the correct file path
    audio_file_path = r"C:\Hackathon\audio2.wav"  # Ensure this path is correct
    
    # Run the main function
    audio_to_art(audio_file_path)
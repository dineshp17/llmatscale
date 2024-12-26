import streamlit as st
import os
import requests
import google.generativeai as genai
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips
import numpy as np
import re

# Function to extend the audio if it's shorter than the video duration
def extend_audio(audio_clip, target_duration):
    audio_duration = audio_clip.duration
    if audio_duration >= target_duration:
        return audio_clip.subclip(0, target_duration)
    else:
        loops = int(np.ceil(target_duration / audio_duration))
        extended_audio = concatenate_audioclips([audio_clip] * loops)
        return extended_audio.subclip(0, target_duration)

# Function to generate the story from the plot points
def generate_story(items):
    prompt = "Create a cohesive story based on the following plot points:\n" + "\n".join(items) + "\nShorten the story for a 1-minute video."
    response = model.generate_content([prompt], stream=True)
    response.resolve()
    return response.text

# Function to generate audio from text
def generate_audio_from_text(text, output_audio_path):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(output_audio_path)
        return True
    except Exception as e:
        st.error(f"Failed to generate audio: {e}")
        return False

# Function to generate and save images based on prompts
def generate_and_save_image(prompt, file_name, headers, API_URL):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    image.save(os.path.join("images", file_name))

# Function to sort image files numerically
def sort_numerically(files):
    return sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))

# Google Generative AI configuration
os.environ['GOOGLE_API_KEY'] = "AIzaSyB3tcyKxjewTaeNZOHEb6AXM9Pfpw5m6I4"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit UI
st.title("AI-Powered Image Prompt Generation & Video Creation")

# Step 1: Input Proverb and Theme
st.header("Enter a Proverb and Theme")
prompt = []
prompt.append(st.text_input("Enter the proverb"))
prompt.append(st.text_input("Enter the theme"))
video_duration = st.slider("Video Duration (seconds)", 10, 120, 60)

if st.button("Generate Video"):
    if len(prompt) == 2 and all(prompt):
        # Step 2: Generate Image Prompts
        prompt_text = (
            f"Generate 20 distinct image prompts that visually represent the concept of '{prompt[0]}' "
            f"in a single, continuous story. Each prompt should depict a different scene that builds upon "
            f"the previous one, creating a cohesive narrative emphasizing the central message of '{prompt[1]}'."
        )
        response = model.generate_content([prompt_text], stream=True)
        response.resolve()
        response_text=response.text
        st.text_area("Generated Image Prompts", response_text, height=300)
        # Extract items (image prompts) from response
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        items = [line[line.find("**") + 2: line.rfind("**")].strip() for line in lines if '**' in line]
        
        # Save items in session state
        st.session_state['items'] = items

        if not items:
            st.error("No items generated from the image prompt. Please check the input.")
        else:
            # Step 3: Generate Images from Prompts
            st.write("Generating images...")
            os.makedirs("images", exist_ok=True)
            API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
            headers = {"Authorization": "Bearer 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"}
            
            # Generate images using threading
            with ThreadPoolExecutor(max_workers=5) as executor:
                for i, scenario in enumerate(st.session_state['items']):
                    executor.submit(generate_and_save_image, scenario, f"image_{i+1}.png", headers, API_URL)
            st.success("Images generated successfully!")

            # Step 4: Generate Audio
            st.write("Generating audio from the story...")
            generated_story = generate_story(items)
            audio_path = "temp_audio.mp3"
            if not generate_audio_from_text(generated_story, audio_path):
                st.error("Audio generation failed. Please check the text.")

            # Step 5: Create Video with Audio
            st.write("Creating video...")
            image_folder = "images"
            video_path = "output_video_Final.mp4"
            image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files = sort_numerically(image_files)
            
            if not image_files:
                st.error("No images found in the specified folder.")
            else:
                # Convert images to numpy arrays
                image_clips = [np.array(Image.open(img).convert('RGB')) for img in image_files]
                fps = len(image_clips) / video_duration
                video_clip = ImageSequenceClip(image_clips, fps=fps)
                
                # Load audio and ensure it's long enough
                audio_clip = AudioFileClip(audio_path)
                audio_clip = extend_audio(audio_clip, video_duration)
                video_clip = video_clip.set_audio(audio_clip)
                
                # Export final video
                video_clip.write_videofile(video_path, codec='libx264', audio_codec='aac')
                st.video(video_path)

            # Step 6: Clean-up
            os.remove(audio_path)
            st.success("Video created successfully!")

    else:
        st.error("Please enter both a proverb and a theme.")
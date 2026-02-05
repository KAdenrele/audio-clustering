import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, high_pass_filter, low_pass_filter, normalize
from scipy.signal import butter, lfilter

class SocialAudioPipeline:
    def __init__(self, input_path):
        self.input_path = input_path
        self.audio = AudioSegment.from_file(input_path)
        self.sample_rate = self.audio.frame_rate
        
        # Load data for LUFS analysis
        self.data, self.rate = sf.read(input_path)
        # Handle stereo/mono for analysis
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
             pass # Stereo is fine
        else:
             # pyloudnorm expects stereo usually, or handle mono specifically
             pass 

    def _measure_lufs(self, data, rate):
        """Measures Integrated Loudness using ITU-R BS.1770-4 standard."""
        meter = pyln.Meter(rate) 
        loudness = meter.integrated_loudness(data)
        return loudness

    def _limit_audio(self, audio_segment, threshold_dbfs=-1.0):
        """Hard limiting to prevent clipping by reducing gain if needed."""
        if audio_segment.max_dBFS > threshold_dbfs:
            # Reduce gain so peaks don't exceed threshold
            gain_reduction = threshold_dbfs - audio_segment.max_dBFS
            return audio_segment.apply_gain(gain_reduction)
        return audio_segment

    # ==========================================
    # 1. TIKTOK PIPELINE
    # Logic: High Noise Floor -> Aggressive Compression -> -14 LUFS Norm -> Bass Cut
    # ==========================================
    def process_tiktok(self, output_path="tiktok_output.mp4"):
        print("--- Running TikTok Pipeline ---")
        pipeline = self.audio
        
        # 1. High Pass Filter (Phone mic simulation/cleanup)
        # TikTok often cuts mud below 100Hz
        pipeline = high_pass_filter(pipeline, 100)

        # 2. Dynamic Range Compression (The "Pumping" effect)
        # Threshold: -20dB, Ratio: 4.0 (Aggressive)
        pipeline = compress_dynamic_range(pipeline, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)

        # 3. Loudness Normalization to -14 LUFS
        # We calculate the gain needed to hit -14
        current_lufs = self._measure_lufs(self.data, self.rate)
        target_lufs = -14.0
        gain_change = target_lufs - current_lufs
        pipeline = pipeline.apply_gain(gain_change)

        # 4. Soft Clipper / Limiter at -1.0 dBTP
        if pipeline.max_dBFS > -1.0:
            pipeline = self._limit_audio(pipeline, -1.0)

        # 5. Export as AAC LC at 128k
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="128k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 2. INSTAGRAM PIPELINE (Reels)
    # Logic: Penalty for loud audio -> HE-AAC compression artifacts
    # ==========================================
    def process_instagram(self, output_path="insta_output.mp4"):
        print("--- Running Instagram Pipeline ---")
        pipeline = self.audio
        
        current_lufs = self._measure_lufs(self.data, self.rate)
        
        # 1. The "Loudness Penalty" Logic
        # If audio is hotter than -14 LUFS, Instagram attenuates it.
        # If it is quieter, they often leave it alone (unlike TikTok which boosts).
        if current_lufs > -14.0:
            gain_change = -14.0 - current_lufs
            pipeline = pipeline.apply_gain(gain_change)
        
        # 2. Spectral Band Replication Simulation (HE-AAC effect)
        # HE-AAC often rolls off real data around 16kHz
        pipeline = low_pass_filter(pipeline, 16000)

        # 3. Export at lower bitrate (64k - 96k common for video)
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="96k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 3. WHATSAPP / SIGNAL (Voice Note Mode)
    # Logic: Bandwidth Efficiency > Quality. Mono. Opus Codec.
    # ==========================================
    def process_whatsapp_voice(self, output_path="whatsapp_voice.ogg"):
        print("--- Running WhatsApp/Signal Pipeline ---")
        pipeline = self.audio

        # 1. Force Mono (Voice notes are rarely stereo)
        pipeline = pipeline.set_channels(1)

        # 2. Aggressive Bandpass (Human Voice Range)
        # Cuts mostly everything below 200Hz and above 7kHz to save data
        pipeline = high_pass_filter(pipeline, 200)
        pipeline = low_pass_filter(pipeline, 7000)

        # 3. Automatic Gain Control (AGC) simulation
        # Boosts quiet signals significantly so you can hear whispers
        pipeline = compress_dynamic_range(pipeline, threshold=-30.0, ratio=8.0, attack=10.0, release=200.0)
        pipeline = normalize(pipeline, headroom=1.0) # Normalize to max -1.0 dB

        # 4. Downsample (Whatsapp often uses 16kHz or 24kHz for voice)
        pipeline = pipeline.set_frame_rate(16000)

        # 5. Export as OGG (Opus) at very low VBR (approx 16k)
        # Note: 'libopus' must be available to ffmpeg
        pipeline.export(output_path, format="ogg", codec="libopus", bitrate="16k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 4. FACEBOOK (Watch/Feed)
    # Logic: Broadcast Standards (-18 LUFS)
    # ==========================================
    def process_facebook(self, output_path="facebook_output.mp4"):
        print("--- Running Facebook Pipeline ---")
        pipeline = self.audio

        # 1. Loudness Normalization to -18 LUFS (Standard Broadcast)
        current_lufs = self._measure_lufs(self.data, self.rate)
        target_lufs = -18.0
        gain_change = target_lufs - current_lufs
        pipeline = pipeline.apply_gain(gain_change)

        # 2. Standard Limiting
        if pipeline.max_dBFS > -2.0:
            pipeline = pipeline.apply_gain(-2.0 - pipeline.max_dBFS)

        # 3. Export High Quality AAC
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="192k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 5. SIGNAL (Voice Message Mode)
    # Logic: Privacy-focused, Opus codec, aggressive compression
    # Similar to WhatsApp but with metadata stripping emphasis
    # ==========================================
    def process_signal(self, output_path="signal_output.ogg"):
        print("--- Running Signal Pipeline ---")
        pipeline = self.audio

        # 1. Force Mono (Voice messages are mono)
        pipeline = pipeline.set_channels(1)

        # 2. Bandpass for voice frequencies
        # Signal uses similar voice optimization as WhatsApp
        # Cuts below 200Hz and above 8kHz
        pipeline = high_pass_filter(pipeline, 200)
        pipeline = low_pass_filter(pipeline, 8000)

        # 3. AGC (Automatic Gain Control) - boosts quiet audio
        pipeline = compress_dynamic_range(pipeline, threshold=-28.0, ratio=6.0, attack=10.0, release=150.0)
        pipeline = normalize(pipeline, headroom=1.0)

        # 4. Downsample to 24kHz (Signal uses slightly higher than WhatsApp)
        pipeline = pipeline.set_frame_rate(24000)

        # 5. Export as OGG Opus at ~24kbps VBR
        pipeline.export(output_path, format="ogg", codec="libopus", bitrate="24k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 6. TELEGRAM (Voice Message Mode)
    # Logic: Cloud-optimized, Opus codec, moderate compression
    # Telegram allows slightly higher quality than WhatsApp
    # ==========================================
    def process_telegram(self, output_path="telegram_output.ogg"):
        print("--- Running Telegram Pipeline ---")
        pipeline = self.audio

        # 1. Force Mono for voice messages
        pipeline = pipeline.set_channels(1)

        # 2. Light bandpass - Telegram preserves more bandwidth than WhatsApp
        # Cuts below 100Hz, preserves up to 12kHz
        pipeline = high_pass_filter(pipeline, 100)
        pipeline = low_pass_filter(pipeline, 12000)

        # 3. Light compression (Telegram is less aggressive)
        pipeline = compress_dynamic_range(pipeline, threshold=-24.0, ratio=3.0, attack=10.0, release=100.0)

        # 4. Normalize with headroom
        pipeline = normalize(pipeline, headroom=1.5)

        # 5. Keep reasonable sample rate (Telegram uses 48kHz for voice)
        pipeline = pipeline.set_frame_rate(48000)

        # 6. Export as OGG Opus at ~32kbps VBR (higher than WhatsApp)
        pipeline.export(output_path, format="ogg", codec="libopus", bitrate="32k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 7. WHATSAPP MEDIA (Audio in video/shared audio files)
    # Logic: Different from voice notes - used when sharing music/audio files
    # Standard vs HD quality modes
    # ==========================================
    def process_whatsapp_media(self, output_path="whatsapp_media.mp4", quality_mode='standard'):
        print(f"--- Running WhatsApp Media Pipeline ({quality_mode.upper()}) ---")
        pipeline = self.audio

        # Keep stereo for media (unlike voice notes)
        # WhatsApp media preserves stereo

        if quality_mode == 'standard':
            # Standard quality: 128kbps AAC, lowpass at 15kHz
            pipeline = low_pass_filter(pipeline, 15000)
            bitrate = "128k"

            # Light loudness normalization
            current_lufs = self._measure_lufs(self.data, self.rate)
            if current_lufs > -14.0:
                gain_change = -14.0 - current_lufs
                pipeline = pipeline.apply_gain(gain_change)

        else:  # 'high' / HD mode
            # HD quality: 192kbps AAC, full bandwidth (up to 18kHz)
            pipeline = low_pass_filter(pipeline, 18000)
            bitrate = "192k"

            # Lighter normalization for HD
            current_lufs = self._measure_lufs(self.data, self.rate)
            if current_lufs > -12.0:
                gain_change = -12.0 - current_lufs
                pipeline = pipeline.apply_gain(gain_change)

        # Limiting to prevent clipping
        if pipeline.max_dBFS > -1.0:
            pipeline = self._limit_audio(pipeline, -1.0)

        # Export as AAC
        pipeline.export(output_path, format="mp4", codec="aac", bitrate=bitrate)
        print(f"Saved to {output_path}")

    # ==========================================
    # 8. TELEGRAM MEDIA (Audio in video/shared audio)
    # Logic: Cloud-optimized media compression
    # ==========================================
    def process_telegram_media(self, output_path="telegram_media.mp4"):
        print("--- Running Telegram Media Pipeline ---")
        pipeline = self.audio

        # Telegram keeps stereo for media
        # Moderate compression - better than Instagram, worse than Facebook

        # 1. Light lowpass at 16kHz
        pipeline = low_pass_filter(pipeline, 16000)

        # 2. Loudness normalization to -16 LUFS
        current_lufs = self._measure_lufs(self.data, self.rate)
        target_lufs = -16.0
        gain_change = target_lufs - current_lufs
        pipeline = pipeline.apply_gain(gain_change)

        # 3. Limiting
        if pipeline.max_dBFS > -1.0:
            pipeline = self._limit_audio(pipeline, -1.0)

        # 4. Export as AAC at 128kbps
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="128k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 9. SIGNAL MEDIA (Audio in video)
    # Logic: Privacy-focused but quality-conscious for media
    # ==========================================
    def process_signal_media(self, output_path="signal_media.mp4"):
        print("--- Running Signal Media Pipeline ---")
        pipeline = self.audio

        # Signal keeps stereo for media sharing

        # 1. Lowpass at 16kHz (similar to Instagram)
        pipeline = low_pass_filter(pipeline, 16000)

        # 2. Light compression for dynamic range
        pipeline = compress_dynamic_range(pipeline, threshold=-20.0, ratio=2.0, attack=10.0, release=100.0)

        # 3. Loudness normalization to -16 LUFS
        current_lufs = self._measure_lufs(self.data, self.rate)
        target_lufs = -16.0
        gain_change = target_lufs - current_lufs
        pipeline = pipeline.apply_gain(gain_change)

        # 4. Limiting
        if pipeline.max_dBFS > -1.0:
            pipeline = self._limit_audio(pipeline, -1.0)

        # 5. Export as AAC at 128kbps
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="128k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 10. TIKTOK MEDIA (Audio in video posts)
    # Logic: Similar to voice but preserves stereo for music content
    # Optimized for music/entertainment content
    # ==========================================
    def process_tiktok_media(self, output_path="tiktok_media.mp4"):
        print("--- Running TikTok Media Pipeline ---")
        pipeline = self.audio

        # Keep stereo for music content

        # 1. High Pass Filter (remove sub-bass rumble)
        pipeline = high_pass_filter(pipeline, 60)

        # 2. Light compression (less aggressive than voice)
        # Preserve dynamics for music
        pipeline = compress_dynamic_range(pipeline, threshold=-18.0, ratio=2.5, attack=10.0, release=80.0)

        # 3. Loudness Normalization to -14 LUFS (TikTok standard)
        current_lufs = self._measure_lufs(self.data, self.rate)
        target_lufs = -14.0
        gain_change = target_lufs - current_lufs
        pipeline = pipeline.apply_gain(gain_change)

        # 4. Soft Clipper / Limiter at -1.0 dBTP
        if pipeline.max_dBFS > -1.0:
            pipeline = self._limit_audio(pipeline, -1.0)

        # 5. Export as AAC LC at 192k (higher quality for music)
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="192k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 11. INSTAGRAM MEDIA (Reels/Stories with music)
    # Logic: Preserves stereo, higher quality than voice posts
    # ==========================================
    def process_instagram_media(self, output_path="instagram_media.mp4"):
        print("--- Running Instagram Media Pipeline ---")
        pipeline = self.audio

        # Keep stereo for music/media content

        current_lufs = self._measure_lufs(self.data, self.rate)

        # 1. The "Loudness Penalty" Logic (same as voice)
        if current_lufs > -14.0:
            gain_change = -14.0 - current_lufs
            pipeline = pipeline.apply_gain(gain_change)

        # 2. Light compression to even out dynamics
        pipeline = compress_dynamic_range(pipeline, threshold=-18.0, ratio=2.0, attack=10.0, release=100.0)

        # 3. Spectral Band Replication Simulation (HE-AAC effect)
        # Media preserves slightly more bandwidth than voice
        pipeline = low_pass_filter(pipeline, 17000)

        # 4. Limiting
        if pipeline.max_dBFS > -1.0:
            pipeline = self._limit_audio(pipeline, -1.0)

        # 5. Export at higher bitrate than voice (128k vs 96k)
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="128k")
        print(f"Saved to {output_path}")

    # ==========================================
    # 12. FACEBOOK MEDIA (Watch/Feed videos with music)
    # Logic: Broadcast standards with stereo preservation
    # ==========================================
    def process_facebook_media(self, output_path="facebook_media.mp4"):
        print("--- Running Facebook Media Pipeline ---")
        pipeline = self.audio

        # Keep stereo for media content

        # 1. Loudness Normalization to -18 LUFS (Standard Broadcast)
        current_lufs = self._measure_lufs(self.data, self.rate)
        target_lufs = -18.0
        gain_change = target_lufs - current_lufs
        pipeline = pipeline.apply_gain(gain_change)

        # 2. Light compression for broadcast consistency
        pipeline = compress_dynamic_range(pipeline, threshold=-16.0, ratio=2.0, attack=10.0, release=100.0)

        # 3. Standard Limiting
        if pipeline.max_dBFS > -2.0:
            pipeline = pipeline.apply_gain(-2.0 - pipeline.max_dBFS)

        # 4. Export High Quality AAC (same as voice - Facebook uses good quality)
        pipeline.export(output_path, format="mp4", codec="aac", bitrate="192k")
        print(f"Saved to {output_path}")

# =======================
# USAGE EXAMPLE
# =======================
if __name__ == "__main__":
    # Ensure you have a file named 'input.wav' in the directory
    # or change this path.
    input_file = "input.wav"

    if os.path.exists(input_file):
        processor = SocialAudioPipeline(input_file)

        # Run all platform simulations
        # Voice note style (mono, low bitrate Opus)
        processor.process_tiktok()
        processor.process_instagram()
        processor.process_whatsapp_voice()
        processor.process_facebook()
        processor.process_signal()
        processor.process_telegram()

        # Media style (stereo, higher bitrate AAC)
        processor.process_whatsapp_media(quality_mode='standard')
        processor.process_whatsapp_media("whatsapp_media_hd.mp4", quality_mode='high')
        processor.process_telegram_media()
        processor.process_signal_media()
    else:
        print("Please place an 'input.wav' file in the directory to test.")
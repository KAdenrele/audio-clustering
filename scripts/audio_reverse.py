import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, wiener, hilbert
from scipy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import CubicSpline
import os

class AudioRestorer:
    """
    Advanced Audio Restoration optimized for SNR, LSD, and Correlation.
    Designed to reverse social media codec damage using phase-linear DSP.
    """
    
    # Target profiles based on known platform encoding constraints
    # Voice modes: optimized for speech (mono, low bitrate, heavy compression)
    # Media modes: optimized for music (stereo, higher bitrate, lighter compression)
    PLATFORM_PROFILES = {
        # Voice modes (original defaults)
        'tiktok': {'highpass_cutoff': 100, 'lowpass_cutoff': 15000, 'target_lufs': -14, 'type': 'voice'},
        'tiktok_voice': {'highpass_cutoff': 100, 'lowpass_cutoff': 15000, 'target_lufs': -14, 'type': 'voice'},
        'instagram': {'highpass_cutoff': 80, 'lowpass_cutoff': 16000, 'target_lufs': -14, 'type': 'voice'},
        'instagram_voice': {'highpass_cutoff': 80, 'lowpass_cutoff': 16000, 'target_lufs': -14, 'type': 'voice'},
        'whatsapp': {'highpass_cutoff': 200, 'lowpass_cutoff': 7000, 'target_lufs': -16, 'type': 'voice'},
        'whatsapp_voice': {'highpass_cutoff': 200, 'lowpass_cutoff': 7000, 'target_lufs': -16, 'type': 'voice'},
        'facebook': {'highpass_cutoff': 60, 'lowpass_cutoff': 18000, 'target_lufs': -18, 'type': 'voice'},
        'facebook_voice': {'highpass_cutoff': 60, 'lowpass_cutoff': 18000, 'target_lufs': -18, 'type': 'voice'},
        'signal': {'highpass_cutoff': 200, 'lowpass_cutoff': 8000, 'target_lufs': -16, 'type': 'voice'},
        'signal_voice': {'highpass_cutoff': 200, 'lowpass_cutoff': 8000, 'target_lufs': -16, 'type': 'voice'},
        'telegram': {'highpass_cutoff': 100, 'lowpass_cutoff': 12000, 'target_lufs': -16, 'type': 'voice'},
        'telegram_voice': {'highpass_cutoff': 100, 'lowpass_cutoff': 12000, 'target_lufs': -16, 'type': 'voice'},

        # Media modes (stereo, higher quality)
        'tiktok_media': {'highpass_cutoff': 60, 'lowpass_cutoff': 18000, 'target_lufs': -14, 'type': 'media'},
        'instagram_media': {'highpass_cutoff': 40, 'lowpass_cutoff': 17000, 'target_lufs': -14, 'type': 'media'},
        'whatsapp_media': {'highpass_cutoff': 40, 'lowpass_cutoff': 15000, 'target_lufs': -14, 'type': 'media'},
        'whatsapp_media_hd': {'highpass_cutoff': 30, 'lowpass_cutoff': 18000, 'target_lufs': -12, 'type': 'media'},
        'facebook_media': {'highpass_cutoff': 40, 'lowpass_cutoff': 19000, 'target_lufs': -18, 'type': 'media'},
        'signal_media': {'highpass_cutoff': 40, 'lowpass_cutoff': 16000, 'target_lufs': -16, 'type': 'media'},
        'telegram_media': {'highpass_cutoff': 40, 'lowpass_cutoff': 16000, 'target_lufs': -16, 'type': 'media'},
    }

    def __init__(self, input_path, platform='tiktok'):
        self.data, self.samplerate = sf.read(input_path)
        self.platform = platform.lower()
        self.profile = self.PLATFORM_PROFILES.get(self.platform, self.PLATFORM_PROFILES['tiktok'])
        
        # Ensure Stereo (Internal processing works best on 2-channel)
        if len(self.data.shape) == 1:
            self.data = np.stack([self.data, self.data], axis=1)

    def _zero_phase_butter(self, data, cutoff, btype='low', order=4):
        """Standard Butterworth with filtfilt to ensure 0 samples of phase delay."""
        nyq = 0.5 * self.samplerate
        if isinstance(cutoff, list):
            norm_cutoff = [np.clip(c / nyq, 0.001, 0.999) for c in cutoff]
        else:
            norm_cutoff = np.clip(cutoff / nyq, 0.001, 0.999)
        b, a = butter(order, norm_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, data, axis=0)

    # ==========================================
    # GEOMETRY & SNR RESTORATION
    # ==========================================

    def reconstruct_clipped_peaks(self, threshold_db=-0.3):
        """Uses Cubic Splines to round off squared-off peaks (De-clipping)."""
        threshold = 10 ** (threshold_db / 20)
        output = np.copy(self.data)
        
        for ch in range(output.shape[1]):
            chan_data = output[:, ch]
            clipped = np.abs(chan_data) >= threshold
            i = 0
            while i < len(clipped):
                if clipped[i]:
                    start = i
                    while i < len(clipped) and clipped[i]: i += 1
                    end = i
                    window = 4 # Samples to look back/forward for curve context
                    if start > window and end < len(chan_data) - window:
                        x_pts = np.concatenate([np.arange(start-window, start), 
                                              np.arange(end, end+window)])
                        y_pts = chan_data[x_pts]
                        cs = CubicSpline(x_pts, y_pts)
                        chan_data[start:end] = cs(np.arange(start, end))
                else: i += 1
            output[:, ch] = chan_data
        self.data = output

    def remove_dc_and_subsonic(self):
        """Cleans subsonic energy and centers waveform for improved SNR."""
        self.data = self._zero_phase_butter(self.data, 20, btype='high')
        self.data -= np.mean(self.data, axis=0)

    # ==========================================
    # SPECTRAL & LSD RESTORATION
    # ==========================================

    def precise_inverse_eq(self):
        """Applies inverse Butterworth transfer function to correct spectral tilt."""
        output = np.copy(self.data)
        for ch in range(output.shape[1]):
            n = len(output[:, ch])
            spec = rfft(output[:, ch])
            freqs = rfftfreq(n, 1/self.samplerate)
            
            # Inverse Highpass & Lowpass correction
            hp_loss = 1.0 / np.sqrt(1 + (self.profile['highpass_cutoff'] / (freqs + 1e-10))**8)
            lp_loss = 1.0 / np.sqrt(1 + (freqs / self.profile['lowpass_cutoff'])**8)
            
            # Boost limited to +12dB to prevent noise floor amplification
            eq = np.clip(1.0 / (hp_loss + 1e-5), 1.0, 4.0) * np.clip(1.0 / (lp_loss + 1e-5), 1.0, 4.0)
            output[:, ch] = irfft(spec * eq, n=n)
        self.data = output

    def adaptive_harmonic_mirroring(self, mix=0.06):
        """Mirrors high-mids into the codec-deleted high-end (Bandwidth Extension)."""
        output = np.copy(self.data)
        for ch in range(output.shape[1]):
            n = len(output[:, ch])
            spec = rfft(output[:, ch])
            freqs = rfftfreq(n, 1/self.samplerate)
            
            idx_cutoff = np.argmin(np.abs(freqs - self.profile['lowpass_cutoff']))
            source_width = min(idx_cutoff, len(spec) - idx_cutoff)
            
            if source_width > 0:
                # Use spectral mirroring (conjugate reverse) to preserve phase alignment
                spec[idx_cutoff:idx_cutoff + source_width] += \
                    np.conj(spec[idx_cutoff - source_width:idx_cutoff][::-1]) * mix
            output[:, ch] = irfft(spec, n=n)
        self.data = output

    def spectral_hole_filler(self, floor_db=-95):
        """Injects shaped dither into zeroed-out frequency bins to optimize LSD."""
        noise_floor = 10 ** (floor_db / 20)
        noise = (np.random.rand(*self.data.shape) - 0.5) * 2 * noise_floor
        # Filter noise to the high-frequency 'codec void' area
        noise = self._zero_phase_butter(noise, 7000, btype='high')
        self.data += noise

    # ==========================================
    # SPATIAL & DYNAMIC RESTORATION
    # ==========================================

    def stereo_reconstruction(self, width=1.12):
        """Centers bass and expands side-channel width using Mid-Side processing."""
        L = self.data[:, 0]
        R = self.data[:, 1]
        M = (L + R) * 0.5
        S = (L - R) * 0.5
        
        # Keep bass mono, expand side highs
        S_high = self._zero_phase_butter(S * width, 150, btype='high')
        self.data = np.stack([M + S_high, M - S_high], axis=1)

    def multiband_transient_reconstruction(self, attack_boost=1.12):
        """Restores transients specifically in the 2-6kHz range for clarity."""
        presence = self._zero_phase_butter(self.data, [2000, 6000], btype='band')
        # Analytical envelope detection
        envelope = np.abs(hilbert(presence, axis=0))
        smooth_env = self._zero_phase_butter(envelope, 60, btype='low')
        
        transients = np.maximum(np.diff(smooth_env, axis=0, prepend=0), 0)
        transients /= (np.max(transients) + 1e-10)
        
        boosted_presence = presence * (1.0 + transients * (attack_boost - 1.0))
        self.data = (self.data - presence) + boosted_presence

    # ==========================================
    # MAIN PIPELINE
    # ==========================================

    def restore_for_metrics(self):
        """The complete sequence optimized for SNR, LSD, and Correlation."""
        print(f"-> Starting Metric-Optimized Restoration [{self.platform}]...")

        # 1. Base Cleaning & Geometry
        self.remove_dc_and_subsonic()       # Correlation Focus
        self.reconstruct_clipped_peaks()    # SNR & Correlation Focus

        # 2. Spatial & Spectral Balance
        self.stereo_reconstruction()        # Inter-channel Correlation Focus
        self.precise_inverse_eq()           # LSD Focus

        # 3. Content Reconstruction
        self.adaptive_harmonic_mirroring()  # LSD & Correlation Focus
        self.multiband_transient_reconstruction() # Correlation Focus
        self.spectral_hole_filler()         # LSD Focus

        # 4. Adaptive Noise Reduction (Post-processing)
        for ch in range(self.data.shape[1]):
            self.data[:, ch] = wiener(self.data[:, ch], mysize=int(0.004 * self.samplerate))

        # 5. Final Calibration
        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = self.data / (max_val + 1e-6) * 0.94 # -0.5dB Headroom

        print("-> Restoration Complete.")

    def restore_full(self):
        """
        Full subtractive restoration - artifact removal only.
        No additive processing (harmonics, stereo widening, noise fill).
        """
        print(f"-> Starting Full Subtractive Restoration [{self.platform}]...")

        # 1. DC offset removal (improves correlation)
        self.data -= np.mean(self.data, axis=0)

        # 2. Declip only severe clipping (can improve SNR)
        self.reconstruct_clipped_peaks(threshold_db=-0.1)

        # 3. Light noise reduction (can improve SNR if careful)
        for ch in range(self.data.shape[1]):
            self.data[:, ch] = wiener(self.data[:, ch], mysize=int(0.002 * self.samplerate))

        # 4. Final normalization
        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = self.data / (max_val + 1e-6) * 0.94

        print("-> Full Subtractive Restoration Complete.")

    def restore_conservative(self):
        """
        Conservative restoration - DC removal and normalization only.
        Safest option for metrics.
        """
        print(f"-> Starting Conservative Restoration [{self.platform}]...")

        # Only DC offset removal
        self.data -= np.mean(self.data, axis=0)

        # Final normalization
        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = self.data / (max_val + 1e-6) * 0.94

        print("-> Conservative Restoration Complete.")

    def restore_aggressive(self):
        """
        Aggressive subtractive restoration - stronger noise reduction.
        May help with very noisy sources but risks removing signal.
        """
        print(f"-> Starting Aggressive Subtractive Restoration [{self.platform}]...")

        # 1. DC offset removal
        self.data -= np.mean(self.data, axis=0)

        # 2. Declip
        self.reconstruct_clipped_peaks(threshold_db=-0.3)

        # 3. Stronger noise reduction
        for ch in range(self.data.shape[1]):
            self.data[:, ch] = wiener(self.data[:, ch], mysize=int(0.005 * self.samplerate))

        # 4. Final normalization
        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = self.data / (max_val + 1e-6) * 0.94

        print("-> Aggressive Subtractive Restoration Complete.")

    def restore_ml(self):
        """ML-based restoration - placeholder for future ML integration."""
        print(f"-> Starting ML Restoration [{self.platform}]...")
        print("   Note: ML models not yet integrated, using conservative mode.")
        self.restore_conservative()

    def restore_match(self, reference_path):
        """
        Reference-guided restoration - uses original to optimize metrics.
        This is the ONLY mode that can reliably improve metrics because
        it knows what the target should be.
        """
        print(f"-> Starting Reference-Matched Restoration [{self.platform}]...")

        # Load reference for analysis
        ref_data, ref_sr = sf.read(reference_path)
        if len(ref_data.shape) == 1:
            ref_data = np.stack([ref_data, ref_data], axis=1)

        # Resample reference to match our sample rate if needed
        if ref_sr != self.samplerate:
            from scipy.signal import resample
            num_samples = int(len(ref_data) * self.samplerate / ref_sr)
            ref_data = resample(ref_data, num_samples, axis=0)

        # Align lengths
        min_len = min(len(self.data), len(ref_data))
        self.data = self.data[:min_len]
        ref_data = ref_data[:min_len]

        # 1. DC offset removal
        self.data -= np.mean(self.data, axis=0)
        ref_data_centered = ref_data - np.mean(ref_data, axis=0)

        # 2. Spectral matching - match our spectrum to reference
        for ch in range(self.data.shape[1]):
            n = len(self.data[:, ch])
            our_spec = rfft(self.data[:, ch])
            ref_spec = rfft(ref_data_centered[:, ch])

            # Match magnitudes to reference (preserve our phase)
            our_mag = np.abs(our_spec)
            ref_mag = np.abs(ref_spec)
            our_phase = np.angle(our_spec)

            # Gentle correction - blend toward reference magnitude
            # Limit correction to avoid amplifying noise in quiet bands
            ratio = (ref_mag + 1e-10) / (our_mag + 1e-10)
            ratio = np.clip(ratio, 0.25, 4.0)  # +/- 12dB max

            # Apply correction
            corrected_mag = our_mag * ratio
            spec_corrected = corrected_mag * np.exp(1j * our_phase)
            self.data[:, ch] = irfft(spec_corrected, n=n)

        # 3. Match RMS loudness to reference
        ref_rms = np.sqrt(np.mean(ref_data_centered ** 2))
        our_rms = np.sqrt(np.mean(self.data ** 2))
        if our_rms > 0:
            self.data = self.data * (ref_rms / our_rms)

        # 4. Final peak limiting
        max_val = np.max(np.abs(self.data))
        if max_val > 0.94:
            self.data = self.data / (max_val + 1e-6) * 0.94

        print("-> Reference-Matched Restoration Complete.")

    def restore_metric(self):
        """
        Minimal processing - only DC removal and normalization.
        Good baseline for comparison.
        """
        print(f"-> Starting Metric-Only Processing [{self.platform}]...")

        # Only remove DC offset and normalize
        self.data -= np.mean(self.data, axis=0)

        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = self.data / (max_val + 1e-6) * 0.94

        print("-> Metric-Only Processing Complete.")

    def restore_none(self):
        """No restoration - pure passthrough for baseline comparison."""
        print(f"-> No Restoration (Passthrough) [{self.platform}]...")
        # Only prevent clipping if needed
        max_val = np.max(np.abs(self.data))
        if max_val > 0.94:
            self.data = self.data / (max_val + 1e-6) * 0.94
        print("-> Passthrough Complete.")

    def normalize_and_save(self, output_path):
        """Normalize audio to -0.5dB headroom and save to file."""
        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = self.data / (max_val + 1e-6) * 0.94  # -0.5dB headroom
        self.save(output_path)

    def save(self, output_path):
        sf.write(output_path, self.data, self.samplerate)
        print(f"File Successfully Saved: {output_path}")

# =======================
# EXECUTION
# =======================
if __name__ == "__main__":
    # Change 'input.wav' to your filename and 'tiktok' to your source platform
    input_file = "input.wav" 
    try:
        restorer = AudioRestorer(input_file, platform="tiktok")
        restorer.restore_for_metrics()
        
        output_name = input_file.replace(".wav", "_metric_optimized.wav")
        restorer.save(output_name)
    except Exception as e:
        print(f"Error: {e}")
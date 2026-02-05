import numpy as np
import librosa
from scipy.signal import coherence
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ForensicEngine")

class ForensicAnalysisError(Exception):
    """Custom exception for audio analysis failures."""
    pass

class ForensicAudioEngine:
    def __init__(self, sr=44100):
        self.sr = sr

    def load_and_align(self, orig_path, proc_path):
        """
        Loads, validates, and aligns signals.
        Returns: y_org, y_prc, alignment_score
        """
        try:
            # 1. Load Original (Assume clean WAV)
            y_org, _ = librosa.load(orig_path, sr=self.sr, mono=True)

            # 2. Load Processed (Robust Pydub loading for MP4/Deepfakes)
            try:
                proc_segment = AudioSegment.from_file(proc_path)
                if proc_segment.frame_rate != self.sr:
                    proc_segment = proc_segment.set_frame_rate(self.sr)
                if proc_segment.channels > 1:
                    proc_segment = proc_segment.set_channels(1)
                
                # Convert to float32 numpy array normalized to [-1, 1]
                y_prc = np.array(proc_segment.get_array_of_samples()).astype(np.float32)
                y_prc /= (2**(proc_segment.sample_width * 8 - 1))
            except Exception:
                # Fallback to librosa if pydub fails
                y_prc, _ = librosa.load(proc_path, sr=self.sr, mono=True)

            # --- Validation ---
            if y_org.size == 0 or y_prc.size == 0:
                raise ForensicAnalysisError("One or both audio files contain no data.")
            if len(y_org) < 2048 or len(y_prc) < 2048:
                raise ForensicAnalysisError("Audio signals too short (min 2048 samples).")

            # Standardise lengths before correlation
            min_len = min(len(y_org), len(y_prc))
            y_org = y_org[:min_len]
            y_prc = y_prc[:min_len]

            # --- Correlation Alignment ---
            # Use a slice for speed (first 2 seconds), but align the whole file
            align_slice = min(len(y_org), self.sr * 2)
            corr = np.correlate(y_org[:align_slice], y_prc[:align_slice], mode='full')
            
            # Get Lag
            lag = np.argmax(corr) - (align_slice - 1)
            
            # Get Score (Normalized Cross-Correlation)
            norm_factor = (np.linalg.norm(y_org[:align_slice]) * np.linalg.norm(y_prc[:align_slice]))
            alignment_score = float(np.max(corr) / norm_factor) if norm_factor > 0 else 0.0

            # Apply Linear Shift
            if lag > 0:
                # y_prc is delayed; shift it left
                y_prc = y_prc[lag:]
                y_org = y_org[:len(y_prc)]
            elif lag < 0:
                # y_prc is early; shift it right
                y_org = y_org[-lag:]
                y_prc = y_prc[:len(y_org)]
            
            # Final trim
            min_final = min(len(y_org), len(y_prc))
            return y_org[:min_final], y_prc[:min_final], alignment_score

        except FileNotFoundError:
            raise ForensicAnalysisError(f"File not found: {orig_path} or {proc_path}")
        except Exception as e:
            raise ForensicAnalysisError(f"Loading/Alignment failed: {str(e)}")

    def get_lsd(self, y_org, y_prc):
        """Calculates Log-Spectral Distance."""
        try:
            S_org = np.abs(librosa.stft(y_org))
            S_prc = np.abs(librosa.stft(y_prc))
            log_spec_diff = (np.log10(S_org**2 + 1e-10) - np.log10(S_prc**2 + 1e-10))**2
            return float(np.mean(np.sqrt(np.mean(log_spec_diff, axis=0))))
        except Exception as e:
            logger.error(f"LSD Error: {e}")
            return None

    def get_si_sdr(self, reference, estimation):
        """Calculates Scale-Invariant SDR."""
        try:
            ref_energy = np.linalg.norm(reference)**2
            if ref_energy < 1e-10: return -np.inf
                
            alpha = np.dot(reference, estimation) / (ref_energy + 1e-8)
            target = alpha * reference
            res = estimation - target
            
            res_energy = np.linalg.norm(res)**2
            if res_energy < 1e-10: return 100.0
                
            sdr = 10 * np.log10(np.linalg.norm(target)**2 / (res_energy + 1e-8))
            return float(sdr)
        except Exception as e:
            logger.error(f"SI-SDR Error: {e}")
            return None

    def get_coherence_data(self, y_org, y_prc):
        """Returns frequency and coherence arrays."""
        try:
            f, coh_val = coherence(y_org, y_prc, fs=self.sr, nperseg=2048)
            return f.tolist(), coh_val.tolist()
        except Exception as e:
            logger.error(f"Coherence Error: {e}")
            return [], []
    def get_spectral_centroid_shift(self, y_org, y_prc):
        """
        Simplistic Metric: Measures change in 'Brightness' or Timbre.
        Returns: float (Shift in Hz). Negative value means the audio became 'dull' or 'muffled'.
        """
        try:
            # Spectral Centroid: The "center of mass" of the spectrum
            cent_org = np.mean(librosa.feature.spectral_centroid(y=y_org, sr=self.sr))
            cent_prc = np.mean(librosa.feature.spectral_centroid(y=y_prc, sr=self.sr))
            
            # Return the difference (How much did it move?)
            return float(cent_prc - cent_org)
        except Exception as e:
            logger.error(f"Spectral Centroid Error: {e}")
            return None

    def get_spectral_rolloff_change(self, y_org, y_prc):
        """
        Simplistic Metric: Measures change in Bandwidth / High-Freq Cutoff.
        Returns: dict containing the Hz shift and percentage of bandwidth retained.
        """
        try:
            # Spectral Rolloff: The frequency below which 85% of energy lies.
            # Good for detecting low-pass filtering common in compression.
            roll_org = np.mean(librosa.feature.spectral_rolloff(y=y_org, sr=self.sr, roll_percent=0.85))
            roll_prc = np.mean(librosa.feature.spectral_rolloff(y=y_prc, sr=self.sr, roll_percent=0.85))

            # Avoid division by zero
            if roll_org == 0:
                roll_org = 1e-8

            return {
                "rolloff_shift_hz": float(roll_prc - roll_org),
                "bandwidth_retained_pct": float((roll_prc / roll_org) * 100)
            }
        except Exception as e:
            logger.error(f"Spectral Rolloff Error: {e}")
            return None
        
    def get_flatness_change(self, y_org, y_prc):
        """Measures change in 'Noisiness' vs 'Tonality'."""
        try:
            flat_org = np.mean(librosa.feature.spectral_flatness(y=y_org))
            flat_prc = np.mean(librosa.feature.spectral_flatness(y=y_prc))
            return float(flat_prc - flat_org)
        except Exception:
            return None
        
    def run_full_analysis(self, orig_path, proc_path):
        """Generates the full forensic report."""
        response = {
            "status": "pending",
            "metrics": {},
            "charts": {},
            "errors": []
        }

        try:
            # 1. Align and Get Confidence
            y_org, y_prc, align_score = self.load_and_align(orig_path, proc_path)
            
            response["metrics"]["alignment_confidence"] = align_score
            
            if align_score < 0.8:
                response["errors"].append("Warning: Low alignment confidence. Metrics may be unreliable.")

            # 2. Standard Forensic Metrics
            response["metrics"]["lsd"] = self.get_lsd(y_org, y_prc)
            response["metrics"]["si_sdr"] = self.get_si_sdr(y_org, y_prc)
            
            # 3. *** NEW: Simplistic Spectral Analysis (The missing part) ***
            # Brightness/Timbre Check
            centroid_shift = self.get_spectral_centroid_shift(y_org, y_prc)
            response["metrics"]["centroid_shift_hz"] = centroid_shift

            flatness_change = self.get_flatness_change(y_org, y_prc)
            response["metrics"]["flatness_change"] = flatness_change
            
            # Bandwidth/Cutoff Check
            rolloff_data = self.get_spectral_rolloff_change(y_org, y_prc)
            if rolloff_data:
                response["metrics"].update(rolloff_data)

            # 4. Coherence (Detailed Frequency Analysis)
            f, coh = self.get_coherence_data(y_org, y_prc)
            response["metrics"]["avg_coherence"] = float(np.mean(coh)) if coh else None
            
            response["charts"]["coherence_f"] = f
            response["charts"]["coherence_val"] = coh
            
            response["status"] = "success"

        except ForensicAnalysisError as fae:
            response["status"] = "failed"
            response["errors"].append(str(fae))
        except Exception as e:
            response["status"] = "error"
            response["errors"].append(f"Unexpected System Error: {str(e)}")

        return response
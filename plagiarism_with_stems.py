"""
Music Plagiarism Detection with Stem Separation
================================================
Використовує Demucs для розділення на stems (vocals, drums, bass, other)
і порівнює ТІЛЬКИ мелодійні компоненти.

Встановлення:
    pip install demucs torch librosa

Або для легшої версії (Spleeter):
    pip install spleeter
"""

import numpy as np
import librosa
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    sr: int = 22050
    use_beat_sync: bool = True
    use_oti: bool = True
    
    # Stem separation
    use_stems: bool = True
    stem_model: str = 'htdemucs'  # 'htdemucs', 'htdemucs_ft', 'mdx_extra'
    stems_to_compare: List[str] = None  # ['vocals', 'other'] or ['melody']
    
    # Chunk settings
    chunk_duration_sec: float = 8.0
    chunk_overlap: float = 0.5
    chunk_threshold: float = 0.55  # Нижчий поріг бо stems чистіші
    
    # Weights
    w_chunk: float = 0.55
    w_dtw: float = 0.25
    w_cos: float = 0.20
    
    # Decision
    threshold: float = 0.50
    
    def __post_init__(self):
        if self.stems_to_compare is None:
            self.stems_to_compare = ['vocals', 'other']  # other = melody/instruments


# =============================================================================
# STEM SEPARATION
# =============================================================================

class StemSeparator:
    """
    Розділяє аудіо на stems використовуючи Demucs.
    
    Stems:
    - vocals: вокал
    - drums: ударні
    - bass: бас
    - other: все інше (мелодія, гітара, синтезатор, etc.)
    """
    
    def __init__(self, model: str = 'htdemucs', device: str = 'cpu'):
        """
        Args:
            model: 'htdemucs' (швидкий), 'htdemucs_ft' (точніший), 'mdx_extra'
            device: 'cpu' або 'cuda'
        """
        self.model = model
        self.device = device
        self._check_demucs()
    
    def _check_demucs(self):
        """Перевіряє чи встановлений demucs"""
        try:
            import demucs
            self.use_python_api = True
        except ImportError:
            # Пробуємо CLI
            result = subprocess.run(['demucs', '--help'], capture_output=True)
            if result.returncode != 0:
                raise ImportError(
                    "Demucs not installed! Run:\n"
                    "  pip install demucs torch\n"
                    "Or for lighter alternative:\n"
                    "  pip install spleeter"
                )
            self.use_python_api = False
    
    def separate(self, audio_path: str, output_dir: str = None) -> Dict[str, str]:
        """
        Розділяє аудіо на stems.
        
        Args:
            audio_path: шлях до аудіо файлу
            output_dir: куди зберегти stems (якщо None - temp dir)
        
        Returns:
            Dict з шляхами до stem файлів:
            {'vocals': '/path/vocals.wav', 'drums': '...', 'bass': '...', 'other': '...'}
        """
        audio_path = Path(audio_path)
        
        # Створюємо temp dir якщо потрібно
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_python_api:
            return self._separate_python(audio_path, output_dir)
        else:
            return self._separate_cli(audio_path, output_dir)
    
    def _separate_python(self, audio_path: Path, output_dir: Path) -> Dict[str, str]:
        """Separation через Python API"""
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio
        
        # Завантажуємо модель
        model = get_model(self.model)
        model.to(self.device)
        model.eval()
        
        # Завантажуємо аудіо
        wav, sr = torchaudio.load(audio_path)
        
        # Resample якщо потрібно
        if sr != model.samplerate:
            wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
        
        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        
        # Apply model
        with torch.no_grad():
            wav = wav.unsqueeze(0).to(self.device)
            sources = apply_model(model, wav, device=self.device)
        
        # sources shape: (1, n_sources, 2, time)
        sources = sources.squeeze(0).cpu().numpy()
        
        # Зберігаємо stems
        stems = {}
        source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
        
        for i, name in enumerate(source_names):
            stem_path = output_dir / f"{name}.wav"
            # Convert to mono and save
            stem_mono = np.mean(sources[i], axis=0)
            import soundfile as sf
            sf.write(stem_path, stem_mono, model.samplerate)
            stems[name] = str(stem_path)
        
        return stems
    
    def _separate_cli(self, audio_path: Path, output_dir: Path) -> Dict[str, str]:
        """Separation через CLI"""
        cmd = [
            'demucs',
            '-n', self.model,
            '-d', self.device,
            '-o', str(output_dir),
            str(audio_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed: {result.stderr}")
        
        # Шукаємо output files
        stem_dir = output_dir / self.model / audio_path.stem
        
        stems = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_path = stem_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = str(stem_path)
        
        return stems
    
    def separate_to_array(self, audio_path: str, sr: int = 22050) -> Dict[str, np.ndarray]:
        """
        Розділяє і повертає numpy arrays замість файлів.
        """
        # Separate to temp files
        temp_dir = tempfile.mkdtemp()
        try:
            stem_paths = self.separate(audio_path, temp_dir)
            
            # Load as arrays
            stems = {}
            for name, path in stem_paths.items():
                y, _ = librosa.load(path, sr=sr, mono=True)
                stems[name] = y
            
            return stems
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)


class SpleeterSeparator:
    """
    Альтернатива: Spleeter (легша, але менш точна)
    
    pip install spleeter
    """
    
    def __init__(self, stems: int = 4):
        """
        Args:
            stems: 2 (vocals/accompaniment), 4 (vocals/drums/bass/other), 
                   5 (vocals/drums/bass/piano/other)
        """
        self.stems = stems
        self._check_spleeter()
    
    def _check_spleeter(self):
        try:
            from spleeter.separator import Separator
            self.separator = Separator(f'spleeter:{self.stems}stems')
        except ImportError:
            raise ImportError("Spleeter not installed! Run: pip install spleeter")
    
    def separate_to_array(self, audio_path: str, sr: int = 22050) -> Dict[str, np.ndarray]:
        """Розділяє аудіо на stems"""
        from spleeter.audio.adapter import AudioAdapter
        
        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(audio_path, sample_rate=sr)
        
        prediction = self.separator.separate(waveform)
        
        stems = {}
        for name, data in prediction.items():
            # Convert to mono
            stems[name] = np.mean(data, axis=1)
        
        return stems


# =============================================================================
# FEATURE EXTRACTION WITH STEMS
# =============================================================================

class StemFeatureExtractor:
    """Екстракція features з stems"""
    
    def __init__(self, config: Config):
        self.cfg = config
        
        # Ініціалізуємо separator
        if config.use_stems:
            try:
                self.separator = StemSeparator(model=config.stem_model)
                self.separator_type = 'demucs'
            except:
                try:
                    self.separator = SpleeterSeparator(stems=4)
                    self.separator_type = 'spleeter'
                except:
                    print("Warning: No stem separator available, using full mix")
                    self.separator = None
                    self.separator_type = None
    
    def extract_hpcp(self, y: np.ndarray) -> np.ndarray:
        """Extract HPCP"""
        return librosa.feature.chroma_cens(y=y, sr=self.cfg.sr)
    
    def beat_sync(self, features: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Beat synchronization"""
        _, beats = librosa.beat.beat_track(y=y, sr=self.cfg.sr)
        if len(beats) > 2:
            return librosa.util.sync(features, beats, aggregate=np.median)
        return features
    
    def extract_features(self, audio_path: str) -> Dict:
        """
        Екстракція features з stem separation.
        
        Returns:
            {
                'combined_hpcp': HPCP від комбінації stems,
                'vocals_hpcp': HPCP тільки вокалу,
                'melody_hpcp': HPCP мелодії (other stem),
                'full_hpcp': HPCP повного міксу (fallback)
            }
        """
        # Завантажуємо повний мікс
        y_full, _ = librosa.load(audio_path, sr=self.cfg.sr, mono=True)
        y_full = librosa.util.normalize(y_full)
        
        result = {
            'full_mix': y_full,
            'full_hpcp': None,
            'stems': {},
            'stem_hpcp': {},
            'combined_hpcp': None
        }
        
        # Extract full mix features (fallback)
        full_hpcp = self.extract_hpcp(y_full)
        if self.cfg.use_beat_sync:
            full_hpcp = self.beat_sync(full_hpcp, y_full)
        result['full_hpcp'] = full_hpcp
        
        # Stem separation
        if self.cfg.use_stems and self.separator is not None:
            try:
                stems = self.separator.separate_to_array(audio_path, sr=self.cfg.sr)
                result['stems'] = stems
                
                # Extract HPCP для кожного stem
                for stem_name, stem_audio in stems.items():
                    if stem_name in self.cfg.stems_to_compare:
                        stem_audio = librosa.util.normalize(stem_audio)
                        hpcp = self.extract_hpcp(stem_audio)
                        if self.cfg.use_beat_sync:
                            hpcp = self.beat_sync(hpcp, stem_audio)
                        result['stem_hpcp'][stem_name] = hpcp
                
                # Комбінуємо stems для порівняння
                if result['stem_hpcp']:
                    # Беремо середнє HPCP від усіх вибраних stems
                    hpcps = list(result['stem_hpcp'].values())
                    
                    # Вирівнюємо довжини
                    min_len = min(h.shape[1] for h in hpcps)
                    hpcps = [h[:, :min_len] for h in hpcps]
                    
                    # Зважене середнє (vocals важливіші для мелодії)
                    weights = []
                    for name in result['stem_hpcp'].keys():
                        if name == 'vocals':
                            weights.append(1.5)  # Вокал важливіший
                        elif name == 'other':
                            weights.append(1.2)  # Мелодія важлива
                        else:
                            weights.append(1.0)
                    
                    combined = np.zeros_like(hpcps[0])
                    total_weight = sum(weights)
                    for hpcp, w in zip(hpcps, weights):
                        combined += hpcp * (w / total_weight)
                    
                    result['combined_hpcp'] = combined
                else:
                    result['combined_hpcp'] = full_hpcp
                    
            except Exception as e:
                print(f"Stem separation failed: {e}, using full mix")
                result['combined_hpcp'] = full_hpcp
        else:
            result['combined_hpcp'] = full_hpcp
        
        return result


# =============================================================================
# OTI
# =============================================================================

def compute_oti(h1: np.ndarray, h2: np.ndarray) -> int:
    """Optimal Transposition Index"""
    g1 = np.mean(h1, axis=1)
    g2 = np.mean(h2, axis=1)
    g1 = g1 / (np.linalg.norm(g1) + 1e-8)
    g2 = g2 / (np.linalg.norm(g2) + 1e-8)
    corr = np.real(np.fft.ifft(np.fft.fft(g1) * np.conj(np.fft.fft(g2))))
    return int(np.argmax(corr))

def apply_oti(h: np.ndarray, oti: int) -> np.ndarray:
    """Apply OTI shift"""
    return np.roll(h, oti, axis=0) if oti != 0 else h


# =============================================================================
# CHUNK ANALYSIS
# =============================================================================

def segment_features(features: np.ndarray, chunk_frames: int, hop_frames: int) -> List[np.ndarray]:
    """Split features into chunks"""
    n = features.shape[1]
    chunks = []
    start = 0
    while start + chunk_frames <= n:
        chunks.append(features[:, start:start+chunk_frames])
        start += hop_frames
    if start < n and (n - start) >= chunk_frames * 0.5:
        chunks.append(features[:, start:])
    return chunks


def compare_chunks(c1: np.ndarray, c2: np.ndarray) -> float:
    """Compare two chunks"""
    # Pad if needed
    l1, l2 = c1.shape[1], c2.shape[1]
    if l1 != l2:
        ml = max(l1, l2)
        if l1 < ml:
            c1 = np.pad(c1, ((0,0), (0, ml-l1)), mode='edge')
        if l2 < ml:
            c2 = np.pad(c2, ((0,0), (0, ml-l2)), mode='edge')
    
    # DTW
    try:
        D, _ = librosa.sequence.dtw(c1, c2, metric='cosine')
        dtw_sim = max(0, 1 - D[-1,-1] / (D.shape[0] + D.shape[1]))
    except:
        dtw_sim = 0
    
    # Cosine
    g1, g2 = np.mean(c1, axis=1), np.mean(c2, axis=1)
    cos_sim = max(0, np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-8))
    
    return 0.6 * dtw_sim + 0.4 * cos_sim


def chunk_analysis(q_hpcp: np.ndarray, r_hpcp: np.ndarray, cfg: Config) -> Dict:
    """Chunk-based analysis"""
    # Estimate fps
    fps = 2 if q_hpcp.shape[1] < 100 else 43
    chunk_fr = max(5, int(cfg.chunk_duration_sec * fps))
    hop_fr = max(2, int(chunk_fr * (1 - cfg.chunk_overlap)))
    
    q_chunks = segment_features(q_hpcp, chunk_fr, hop_fr)
    r_chunks = segment_features(r_hpcp, chunk_fr, hop_fr)
    
    nq, nr = len(q_chunks), len(r_chunks)
    if nq == 0 or nr == 0:
        return {'score': 0, 'n_plag': 0, 'n_total': 0, 'matches': []}
    
    # Similarity matrix
    matrix = np.array([[compare_chunks(q_chunks[i], r_chunks[j]) 
                        for j in range(nr)] for i in range(nq)])
    
    # Best matches
    matches = []
    for i in range(nq):
        best_j = int(np.argmax(matrix[i, :]))
        best_sim = float(matrix[i, best_j])
        is_plag = best_sim >= cfg.chunk_threshold
        matches.append({'ref_idx': best_j, 'sim': best_sim, 'is_plag': is_plag})
    
    n_plag = sum(1 for m in matches if m['is_plag'])
    sims = [m['sim'] for m in matches]
    
    # Consecutive bonus
    max_con, cur = 0, 0
    for m in matches:
        if m['is_plag']:
            cur += 1
            max_con = max(max_con, cur)
        else:
            cur = 0
    bonus = min(0.15, (max_con - 1) * 0.05) if max_con > 1 else 0
    
    score = min(1.0, 0.4 * np.mean(sims) + 0.35 * (n_plag/nq) + 0.25 * np.max(sims) + bonus)
    
    return {
        'score': score,
        'n_plag': n_plag,
        'n_total': nq,
        'plag_ratio': n_plag / nq,
        'matches': matches,
        'matrix': matrix
    }


# =============================================================================
# GLOBAL METRICS
# =============================================================================

def global_dtw(f1: np.ndarray, f2: np.ndarray) -> float:
    """Global DTW similarity"""
    try:
        D, _ = librosa.sequence.dtw(f1, f2, metric='cosine')
        return max(0, 1 - D[-1,-1] / (D.shape[0] + D.shape[1]))
    except:
        return 0.0


def global_cosine(f1: np.ndarray, f2: np.ndarray) -> float:
    """Global cosine similarity"""
    g1, g2 = np.mean(f1, axis=1), np.mean(f2, axis=1)
    return max(0, np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-8))


# =============================================================================
# MAIN COMPARATOR
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparison"""
    score: float
    is_plagiarism: bool
    
    # Chunk analysis
    chunk_score: float
    n_plag_chunks: int
    n_total_chunks: int
    plag_percentage: float
    
    # Global
    global_dtw: float
    global_cosine: float
    
    # OTI
    oti: int
    
    # Stems used
    stems_used: List[str]
    
    # Details
    chunk_matches: List = None


class StemPlagiarismDetector:
    """
    Детектор плагіату з stem separation.
    
    Використання:
        detector = StemPlagiarismDetector()
        result = detector.compare('query.mp3', 'reference.mp3')
        print(f"Score: {result.score}, Plagiarism: {result.is_plagiarism}")
    """
    
    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self.extractor = StemFeatureExtractor(self.cfg)
    
    def compare(self, query_path: str, ref_path: str) -> ComparisonResult:
        """
        Порівнює два аудіо файли.
        
        Args:
            query_path: потенційний плагіат
            ref_path: оригінал
        
        Returns:
            ComparisonResult
        """
        # 1. Extract features (with stem separation)
        print("Extracting features from query...")
        query_feat = self.extractor.extract_features(query_path)
        
        print("Extracting features from reference...")
        ref_feat = self.extractor.extract_features(ref_path)
        
        # 2. Get HPCP to compare
        q_hpcp = query_feat['combined_hpcp']
        r_hpcp = ref_feat['combined_hpcp']
        
        # 3. OTI alignment
        oti = compute_oti(q_hpcp, r_hpcp) if self.cfg.use_oti else 0
        r_hpcp_aligned = apply_oti(r_hpcp, oti)
        
        # 4. Chunk analysis
        print("Running chunk analysis...")
        chunk_res = chunk_analysis(q_hpcp, r_hpcp_aligned, self.cfg)
        
        # 5. Global metrics
        g_dtw = global_dtw(q_hpcp, r_hpcp_aligned)
        g_cos = global_cosine(q_hpcp, r_hpcp_aligned)
        
        # 6. Combine
        score = (
            self.cfg.w_chunk * chunk_res['score'] +
            self.cfg.w_dtw * g_dtw +
            self.cfg.w_cos * g_cos
        )
        
        # Boost if strong chunk match
        if chunk_res['plag_ratio'] > 0.5 and chunk_res['score'] > 0.6:
            score = max(score, chunk_res['score'] * 0.95)
        
        # 7. Decision
        is_plag = score >= self.cfg.threshold
        
        # What stems were used?
        stems_used = list(query_feat.get('stem_hpcp', {}).keys())
        if not stems_used:
            stems_used = ['full_mix']
        
        return ComparisonResult(
            score=score,
            is_plagiarism=is_plag,
            chunk_score=chunk_res['score'],
            n_plag_chunks=chunk_res['n_plag'],
            n_total_chunks=chunk_res['n_total'],
            plag_percentage=chunk_res['plag_ratio'] * 100,
            global_dtw=g_dtw,
            global_cosine=g_cos,
            oti=oti,
            stems_used=stems_used,
            chunk_matches=chunk_res['matches']
        )
    
    def compare_stems_separately(self, query_path: str, ref_path: str) -> Dict[str, ComparisonResult]:
        """
        Порівнює кожен stem окремо.
        Корисно для аналізу: який саме елемент сплагіачено.
        
        Returns:
            {
                'vocals': ComparisonResult,
                'other': ComparisonResult,
                'combined': ComparisonResult
            }
        """
        # Extract with stems
        query_feat = self.extractor.extract_features(query_path)
        ref_feat = self.extractor.extract_features(ref_path)
        
        results = {}
        
        # Compare each stem separately
        for stem_name in self.cfg.stems_to_compare:
            q_hpcp = query_feat['stem_hpcp'].get(stem_name)
            r_hpcp = ref_feat['stem_hpcp'].get(stem_name)
            
            if q_hpcp is not None and r_hpcp is not None:
                # OTI
                oti = compute_oti(q_hpcp, r_hpcp) if self.cfg.use_oti else 0
                r_hpcp_aligned = apply_oti(r_hpcp, oti)
                
                # Analysis
                chunk_res = chunk_analysis(q_hpcp, r_hpcp_aligned, self.cfg)
                g_dtw = global_dtw(q_hpcp, r_hpcp_aligned)
                g_cos = global_cosine(q_hpcp, r_hpcp_aligned)
                
                score = (self.cfg.w_chunk * chunk_res['score'] + 
                        self.cfg.w_dtw * g_dtw + 
                        self.cfg.w_cos * g_cos)
                
                results[stem_name] = ComparisonResult(
                    score=score,
                    is_plagiarism=score >= self.cfg.threshold,
                    chunk_score=chunk_res['score'],
                    n_plag_chunks=chunk_res['n_plag'],
                    n_total_chunks=chunk_res['n_total'],
                    plag_percentage=chunk_res['plag_ratio'] * 100,
                    global_dtw=g_dtw,
                    global_cosine=g_cos,
                    oti=oti,
                    stems_used=[stem_name],
                    chunk_matches=chunk_res['matches']
                )
        
        # Combined
        results['combined'] = self.compare(query_path, ref_path)
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_compare(query_path: str, ref_path: str, use_stems: bool = True) -> ComparisonResult:
    """Quick comparison"""
    config = Config(use_stems=use_stems)
    detector = StemPlagiarismDetector(config)
    return detector.compare(query_path, ref_path)


def compare_with_report(query_path: str, ref_path: str) -> None:
    """Compare and print detailed report"""
    config = Config(use_stems=True)
    detector = StemPlagiarismDetector(config)
    
    print("="*60)
    print("PLAGIARISM DETECTION REPORT")
    print("="*60)
    print(f"Query: {query_path}")
    print(f"Reference: {ref_path}")
    print("-"*60)
    
    # Compare each stem
    print("\nAnalyzing stems separately...")
    results = detector.compare_stems_separately(query_path, ref_path)
    
    print("\n" + "-"*60)
    print("STEM-BY-STEM ANALYSIS:")
    print("-"*60)
    
    for stem_name, res in results.items():
        if stem_name == 'combined':
            continue
        status = "🚨 PLAG" if res.is_plagiarism else "✅ OK"
        print(f"\n{stem_name.upper()}:")
        print(f"  Score: {res.score:.3f} {status}")
        print(f"  Chunks: {res.n_plag_chunks}/{res.n_total_chunks} ({res.plag_percentage:.1f}%)")
        print(f"  OTI: {res.oti} semitones")
    
    print("\n" + "-"*60)
    print("COMBINED RESULT:")
    print("-"*60)
    
    combined = results['combined']
    status = "🚨 PLAGIARISM DETECTED!" if combined.is_plagiarism else "✅ NO PLAGIARISM"
    
    print(f"\n{status}")
    print(f"\nFinal Score: {combined.score:.3f}")
    print(f"Plagiarized chunks: {combined.n_plag_chunks}/{combined.n_total_chunks} ({combined.plag_percentage:.1f}%)")
    print(f"Stems used: {', '.join(combined.stems_used)}")
    
    print("="*60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

        # Quick compare
        result = quick_compare('/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/comparison/_DANCE_ 싸이 _PSY_ - 챔피언.wav', '/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav')
        print(f"Score: {result.score}")
        
        # With detailed report
        compare_with_report('/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/comparison/_DANCE_ 싸이 _PSY_ - 챔피언.wav', '/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav')
        
        # Compare stems separately
        detector = StemPlagiarismDetector()
        results = detector.compare_stems_separately('/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/comparison/_DANCE_ 싸이 _PSY_ - 챔피언.wav', '/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/original/_Official Audio_ 이정현_Lee Jung-hyun_ - 와.wav')
        print(f"Vocals similarity: {results['vocals'].score}")
        print(f"Melody similarity: {results['other'].score}")



    # print("""
    # Music Plagiarism Detection with Stem Separation
    # ================================================
    
    # Usage:
    
    #     # # Quick compare
    #     # result = quick_compare('query.mp3', 'reference.mp3')
    #     # print(f"Score: {result.score}")
        
    #     # # With detailed report
    #     # compare_with_report('query.mp3', 'reference.mp3')
        
    #     # # Compare stems separately
    #     # detector = StemPlagiarismDetector()
    #     # results = detector.compare_stems_separately('query.mp3', 'ref.mp3')
    #     # print(f"Vocals similarity: {results['vocals'].score}")
    #     # print(f"Melody similarity: {results['other'].score}")
    
    # Installation:
    #     pip install demucs torch librosa soundfile
        
    #     # Or lighter alternative:
    #     pip install spleeter librosa
    # """)

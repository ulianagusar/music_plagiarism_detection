"""
Гібридний пошук схожих пісень через API аналізу структури.

API повертає:
- segments: частини пісні (verse, chorus, bridge...)
- logits: матриця (T, 128) — фічі по часу
- structure: текстова структура ("verse-chorus-chorus")

Пошук:
1. FAISS — швидкий відбір по усередненому вектору logits
2. DTW — точне порівняння послідовностей logits

Запуск:
    pip install requests numpy pandas faiss-cpu scipy tqdm
    python api_song_search.py
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd
import faiss
from scipy.spatial.distance import cdist
from tqdm import tqdm


# ============================================================
# НАЛАШТУВАННЯ
# ============================================================

@dataclass
class Config:
    # API
    api_url: str = "http://13.220.223.194:8000/analyze"
    api_timeout: int = 300
    max_workers: int = 4  # паралельні запити до API
    
    # Шляхи
    origin_dir: Path = Path("/content/original")
    comparison_dir: Path = Path("/content/comparison/comparison")
    pairs_csv: Path = Path("/content/song_pairs.csv")
    cache_dir: Path = Path("./cache_api")
    
    # Гібридний пошук
    faiss_candidates: int = 50
    
    # Кеш
    use_cache: bool = True


# ============================================================
# API КЛІЄНТ
# ============================================================

def analyze_file(filepath: Path, api_url: str, timeout: int) -> dict | None:
    """
    Відправляє файл на API і повертає результат.
    
    Повертає dict з ключами:
    - logits_matrix: np.ndarray (T, 128)
    - segments: list of dicts
    - structure: str
    """
    try:
        with open(filepath, "rb") as f:
            files = {"file": (filepath.name, f, "audio/wav")}
            response = requests.post(api_url, files=files, timeout=timeout)
        
        if response.status_code != 200:
            print(f"[API ERROR] {filepath.name}: status {response.status_code}")
            return None
        
        data = response.json()
        
        # Конвертуємо logits в матрицю (T, C)
        logits = data.get("logits", {})
        T = logits.get("T", 0)
        C = logits.get("C", 0)
        values = logits.get("values", [])
        
        if T == 0 or C == 0 or not values:
            print(f"[API ERROR] {filepath.name}: empty logits")
            return None
        
        # values — це плоский список, reshape в (T, C)
        matrix = np.array(values, dtype=np.float32).reshape(T, C)
        
        return {
            "logits_matrix": matrix,
            "segments": data.get("segments", []),
            "structure": data.get("structure", ""),
            "frame_rate": logits.get("frame_rate", 8.333),
        }
        
    except Exception as e:
        print(f"[API ERROR] {filepath.name}: {e}")
        return None


# ============================================================
# DTW (Dynamic Time Warping)
# ============================================================

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    DTW відстань між послідовностями.
    
    seq1: (T1, D)
    seq2: (T2, D)
    """
    n, m = len(seq1), len(seq2)
    
    # Косинусна відстань між кожною парою фреймів
    cost = cdist(seq1, seq2, metric="cosine")
    
    # DP таблиця
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i-1, j-1] + min(
                dp[i-1, j],
                dp[i, j-1],
                dp[i-1, j-1]
            )
    
    return dp[n, m] / (n + m)


# ============================================================
# ГІБРИДНИЙ ІНДЕКС
# ============================================================

class HybridIndex:
    """FAISS для швидкого відбору + DTW для точного порівняння."""
    
    def __init__(self, n_candidates: int = 50):
        self.n_candidates = n_candidates
        self.faiss_index = None
        self.sequences = []
        self.metadata = []
    
    def build(self, means: np.ndarray, sequences: list, metadata: list):
        """Будує індекс."""
        # Нормалізація для cosine similarity
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        means_norm = means / (norms + 1e-12)
        
        self.faiss_index = faiss.IndexFlatIP(means_norm.shape[1])
        self.faiss_index.add(means_norm.astype("float32"))
        
        self.sequences = sequences
        self.metadata = metadata
        
        print(f"Індекс побудовано: {len(metadata)} треків")
    
    def search(self, query_mean: np.ndarray, query_seq: np.ndarray) -> tuple:
        """
        Пошук найближчого треку.
        
        Повертає: (metadata_dict, similarity_score)
        """
        # Нормалізуємо запит
        query_mean = query_mean / (np.linalg.norm(query_mean) + 1e-12)
        query_mean = query_mean.reshape(1, -1).astype("float32")
        
        # FAISS: швидкий відбір кандидатів
        n_search = min(self.n_candidates, len(self.sequences))
        _, indices = self.faiss_index.search(query_mean, n_search)
        
        # DTW: точне порівняння
        best_idx = None
        best_dist = float("inf")
        
        for idx in indices[0]:
            dist = dtw_distance(query_seq, self.sequences[idx])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        similarity = 1.0 / (1.0 + best_dist)
        return self.metadata[best_idx], similarity


# ============================================================
# КЕШУВАННЯ
# ============================================================

def get_cache_path(cache_dir: Path, filename: str) -> Path:
    """Шлях до кешу для файлу."""
    return cache_dir / f"{filename}.json"


def save_to_cache(cache_dir: Path, filename: str, data: dict):
    """Зберігає результат API в кеш."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(cache_dir, filename)
    
    # Конвертуємо numpy в list для JSON
    cache_data = {
        "logits_matrix": data["logits_matrix"].tolist(),
        "segments": data["segments"],
        "structure": data["structure"],
        "frame_rate": data["frame_rate"],
    }
    cache_path.write_text(json.dumps(cache_data))


def load_from_cache(cache_dir: Path, filename: str) -> dict | None:
    """Завантажує з кешу."""
    cache_path = get_cache_path(cache_dir, filename)
    
    if not cache_path.exists():
        return None
    
    try:
        cache_data = json.loads(cache_path.read_text())
        cache_data["logits_matrix"] = np.array(cache_data["logits_matrix"], dtype=np.float32)
        return cache_data
    except:
        return None


# ============================================================
# ОБРОБКА ПАПКИ
# ============================================================

def process_folder(folder: Path, cfg: Config) -> tuple:
    """
    Обробляє всі .wav файли в папці через API.
    
    Повертає: (means, sequences, metadata)
    """
    files = sorted(folder.rglob("*.wav"))
    print(f"Знайдено {len(files)} файлів в {folder}")
    
    means = []
    sequences = []
    metadata = []
    
    # Функція для обробки одного файлу
    def process_one(filepath: Path) -> tuple:
        filename = filepath.name
        
        # Пробуємо кеш
        if cfg.use_cache:
            cached = load_from_cache(cfg.cache_dir, filename)
            if cached:
                return filepath, cached
        
        # Запит до API
        result = analyze_file(filepath, cfg.api_url, cfg.api_timeout)
        
        # Зберігаємо в кеш
        if result and cfg.use_cache:
            save_to_cache(cfg.cache_dir, filename, result)
        
        return filepath, result
    
    # Паралельна обробка
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(process_one, f): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Обробка"):
            filepath, result = future.result()
            
            if result is None:
                continue
            
            matrix = result["logits_matrix"]
            
            # Mean ембединг для FAISS
            mean = matrix.mean(axis=0)
            
            means.append(mean)
            sequences.append(matrix)
            metadata.append({
                "name": filepath.name,
                "path": str(filepath),
                "structure": result["structure"],
            })
    
    if not means:
        raise RuntimeError("Не вдалось обробити жодного файлу")
    
    means = np.stack(means)
    return means, sequences, metadata


# ============================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================

def find_wav_files(folder: Path) -> dict:
    """Мапа: ім'я файлу -> шлях."""
    return {p.name: p for p in folder.rglob("*.wav") if p.is_file()}


def ensure_wav(name: str) -> str:
    """Додає .wav якщо треба."""
    return name if name.lower().endswith(".wav") else name + ".wav"


def run(cfg: Config):
    """Запускає експеримент."""
    
    print("=" * 60)
    print("ГІБРИДНИЙ ПОШУК ЧЕРЕЗ API")
    print(f"API: {cfg.api_url}")
    print(f"Кандидатів FAISS: {cfg.faiss_candidates}")
    print("=" * 60)
    
    # 1. Обробляємо comparison папку
    print("\n[1] Обробка comparison...")
    means, sequences, metadata = process_folder(cfg.comparison_dir, cfg)
    
    # 2. Будуємо індекс
    print("\n[2] Будую індекс...")
    index = HybridIndex(n_candidates=cfg.faiss_candidates)
    index.build(means, sequences, metadata)
    
    # 3. Мапа файлів
    origin_map = find_wav_files(cfg.origin_dir)
    comp_map = {m["name"]: m for m in metadata}
    
    # 4. Читаємо пари з CSV
    print("\n[3] Читаю пари...")
    df = pd.read_csv(cfg.pairs_csv)
    
    # 5. Перевіряємо пари
    print("\n[4] Пошук...")
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ori_name = ensure_wav(row["ori_title"])
        comp_name = ensure_wav(row["comp_title"])
        
        ori_path = origin_map.get(ori_name)
        
        if ori_path is None:
            results.append({
                "ori": ori_name,
                "expected": comp_name,
                "predicted": None,
                "score": None,
                "match": False,
                "status": "ORI_NOT_FOUND"
            })
            continue
        
        # Отримуємо ембединг запиту
        if cfg.use_cache:
            cached = load_from_cache(cfg.cache_dir, ori_name)
            if cached:
                query_matrix = cached["logits_matrix"]
            else:
                result = analyze_file(ori_path, cfg.api_url, cfg.api_timeout)
                if result:
                    save_to_cache(cfg.cache_dir, ori_name, result)
                    query_matrix = result["logits_matrix"]
                else:
                    results.append({
                        "ori": ori_name,
                        "expected": comp_name,
                        "predicted": None,
                        "score": None,
                        "match": False,
                        "status": "API_ERROR"
                    })
                    continue
        else:
            result = analyze_file(ori_path, cfg.api_url, cfg.api_timeout)
            if not result:
                results.append({
                    "ori": ori_name,
                    "expected": comp_name,
                    "predicted": None,
                    "score": None,
                    "match": False,
                    "status": "API_ERROR"
                })
                continue
            query_matrix = result["logits_matrix"]
        
        query_mean = query_matrix.mean(axis=0)
        
        # Пошук
        match, score = index.search(query_mean, query_matrix)
        
        results.append({
            "ori": ori_name,
            "expected": comp_name,
            "predicted": match["name"],
            "score": round(score, 4),
            "match": match["name"] == comp_name,
            "status": "OK"
        })
    
    # 6. Результати
    results_df = pd.DataFrame(results)
    
    valid = results_df[results_df["status"] == "OK"]
    accuracy = valid["match"].mean() if len(valid) > 0 else 0
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТИ")
    print("=" * 60)
    print(f"Всього пар: {len(results_df)}")
    print(f"Перевірено: {len(valid)}")
    print(f"Точність: {accuracy:.2%}")
    
    # Статистика по статусах
    print(f"\nСтатуси:")
    print(results_df["status"].value_counts().to_string())
    
    # Помилки
    errors = valid[~valid["match"]].sort_values("score", ascending=False)
    if len(errors) > 0:
        print(f"\nПомилки ({len(errors)}):")
        print(errors[["ori", "expected", "predicted", "score"]].head(10).to_string(index=False))
    
    # Зберігаємо
    output = Path("./api_search_results.csv")
    results_df.to_csv(output, index=False)
    print(f"\nЗбережено: {output}")
    
    return results_df


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    cfg = Config()
    
    # Налаштування:
    # cfg.faiss_candidates = 100  # більше кандидатів = точніше, але повільніше
    # cfg.max_workers = 8         # більше паралельних запитів
    # cfg.use_cache = False       # вимкнути кеш
    
    run(cfg)

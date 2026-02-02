# """
# MIDI Similarity Pipeline

# Конвертує WAV файли в MIDI через API, порівнює пари пісень,
# знаходить найбільш подібні та рахує точність.

# Використання:
#     python pipeline.py --api http://your-server:8000
# """

# import os
# import csv
# import requests
# from pathlib import Path
# from collections import Counter
# from typing import List, Tuple, Dict
# import mido

# # ============== CONFIG ==============

# API_URL = "http://13.220.223.194:9000"  # Зміни на свій сервер
# ORIGINAL_DIR = "comparison"
# COMPARISON_DIR = "original"
# MIDI_DIR = "midi_cache"
# PAIRS_CSV = "song_pairs.csv"

# # ============== MIDI EXTRACTION ==============

# def extract_notes(midi_path: str) -> List[int]:
#     """Витягує послідовність нот з MIDI файлу"""
#     try:
#         mid = mido.MidiFile(midi_path)
#         notes = []
#         for track in mid.tracks:
#             for msg in track:
#                 if msg.type == 'note_on' and msg.velocity > 0:
#                     notes.append(msg.note)
#         return notes
#     except Exception as e:
#         print(f"  Помилка читання {midi_path}: {e}")
#         return []


# def extract_intervals(notes: List[int]) -> List[int]:
#     """Конвертує ноти в інтервали"""
#     if len(notes) < 2:
#         return []
#     return [notes[i+1] - notes[i] for i in range(len(notes)-1)]


# def get_ngrams(sequence: List[int], n: int) -> List[Tuple]:
#     """Створює n-грами"""
#     if len(sequence) < n:
#         return []
#     return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]


# def cosine_similarity(counter1: Counter, counter2: Counter) -> float:
#     """Косинусна подібність"""
#     all_keys = set(counter1.keys()) | set(counter2.keys())
#     dot_product = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in all_keys)
#     norm1 = sum(v**2 for v in counter1.values()) ** 0.5
#     norm2 = sum(v**2 for v in counter2.values()) ** 0.5
#     if norm1 == 0 or norm2 == 0:
#         return 0.0
#     return dot_product / (norm1 * norm2)


# def calculate_similarity(midi1: str, midi2: str, n_values: List[int] = [3, 4, 5]) -> float:
#     """Рахує середню подібність між двома MIDI файлами"""
#     notes1 = extract_notes(midi1)
#     notes2 = extract_notes(midi2)
    
#     if not notes1 or not notes2:
#         return 0.0
    
#     # Використовуємо інтервали для інваріантності до тональності
#     seq1 = extract_intervals(notes1)
#     seq2 = extract_intervals(notes2)
    
#     if not seq1 or not seq2:
#         return 0.0
    
#     similarities = []
#     for n in n_values:
#         ngrams1 = get_ngrams(seq1, n)
#         ngrams2 = get_ngrams(seq2, n)
        
#         if ngrams1 and ngrams2:
#             counter1 = Counter(ngrams1)
#             counter2 = Counter(ngrams2)
#             sim = cosine_similarity(counter1, counter2)
#             similarities.append(sim)
    
#     return sum(similarities) / len(similarities) if similarities else 0.0


# # ============== WAV TO MIDI CONVERSION ==============

# def convert_wav_to_midi(wav_path: str, output_midi: str, api_url: str) -> bool:
#     """Конвертує WAV в MIDI через API"""
    
#     if os.path.exists(output_midi):
#         print(f"  [CACHE] {output_midi}")
#         return True
    
#     print(f"  Конвертую: {wav_path}")
    
#     try:
#         with open(wav_path, "rb") as f:
#             response = requests.post(
#                 f"{api_url}/process",
#                 files={"file": (os.path.basename(wav_path), f)},
#                 data={
#                     "sound_type": "melody",
#                     "extract_stem": "false",
#                     "convert_midi": "true",
#                 },
#                 timeout=600
#             )
        
#         result = response.json()
        
#         if "detail" in result:
#             print(f"  Помилка API: {result['detail'][:100]}...")
#             return False
        
#         # Завантажуємо MIDI
#         for url in result.get("download_urls", []):
#             if url.endswith(".mid"):
#                 r = requests.get(f"{api_url}{url}")
#                 os.makedirs(os.path.dirname(output_midi), exist_ok=True)
#                 with open(output_midi, "wb") as f:
#                     f.write(r.content)
#                 print(f"  Збережено: {output_midi}")
#                 return True
        
#         print("  MIDI файл не знайдено у відповіді")
#         return False
        
#     except Exception as e:
#         print(f"  Помилка: {e}")
#         return False


# def find_wav_file(directory: str, title: str) -> str:
#     """Шукає WAV файл за назвою (часткове співпадіння)"""
#     dir_path = Path(directory)
    
#     if not dir_path.exists():
#         return None
    
#     # Очищаємо назву для пошуку
#     clean_title = title.lower().replace("_", " ").replace("-", " ")
    
#     for f in dir_path.iterdir():
#         if f.suffix.lower() in ['.wav', '.mp3', '.flac']:
#             clean_name = f.stem.lower().replace("_", " ").replace("-", " ")
#             # Перевіряємо чи назва файлу містить частину title або навпаки
#             if clean_title[:20] in clean_name or clean_name[:20] in clean_title:
#                 return str(f)
    
#     # Якщо не знайшли, шукаємо перші слова
#     title_words = clean_title.split()[:3]
#     for f in dir_path.iterdir():
#         if f.suffix.lower() in ['.wav', '.mp3', '.flac']:
#             clean_name = f.stem.lower()
#             if all(word in clean_name for word in title_words if len(word) > 2):
#                 return str(f)
    
#     return None


# # ============== MAIN PIPELINE ==============

# def load_pairs(csv_path: str) -> List[Dict]:
#     """Завантажує пари з CSV"""
#     pairs = []
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             pairs.append({
#                 'id': row.get('', row.get('id', '')),
#                 'original': row['ori_title'],
#                 'comparison': row['comp_title'],
#                 'relation': row['relation'],
#             })
#     return pairs


# def run_pipeline(api_url: str, original_dir: str, comparison_dir: str, 
#                  midi_dir: str, pairs_csv: str):
#     """Основний пайплайн"""
    
#     print("=" * 70)
#     print("MIDI SIMILARITY PIPELINE")
#     print("=" * 70)
    
#     # Завантажуємо пари
#     pairs = load_pairs(pairs_csv)
#     print(f"\nЗавантажено {len(pairs)} пар з {pairs_csv}")
    
#     # Створюємо папки для MIDI
#     os.makedirs(f"{midi_dir}/original", exist_ok=True)
#     os.makedirs(f"{midi_dir}/comparison", exist_ok=True)
    
#     # Конвертуємо всі файли в MIDI
#     print("\n" + "=" * 70)
#     print("КРОК 1: Конвертація WAV -> MIDI")
#     print("=" * 70)
    
#     original_midis = {}  # title -> midi_path
#     comparison_midis = {}
    
#     # Збираємо всі оригінальні файли
#     print("\n[ORIGINAL FILES]")
#     for pair in pairs:
#         title = pair['original']
#         if title in original_midis:
#             continue
            
#         wav_path = find_wav_file(original_dir, title)
#         if wav_path:
#             midi_path = f"{midi_dir}/original/{Path(wav_path).stem}.mid"
#             if convert_wav_to_midi(wav_path, midi_path, api_url):
#                 original_midis[title] = midi_path
#         else:
#             print(f"  [NOT FOUND] {title[:50]}...")
    
#     # Збираємо всі comparison файли
#     print("\n[COMPARISON FILES]")
#     for pair in pairs:
#         title = pair['comparison']
#         if title in comparison_midis:
#             continue
            
#         wav_path = find_wav_file(comparison_dir, title)
#         if wav_path:
#             midi_path = f"{midi_dir}/comparison/{Path(wav_path).stem}.mid"
#             if convert_wav_to_midi(wav_path, midi_path, api_url):
#                 comparison_midis[title] = midi_path
#         else:
#             print(f"  [NOT FOUND] {title[:50]}...")
    
#     print(f"\nКонвертовано: {len(original_midis)} original, {len(comparison_midis)} comparison")
    
#     # Порівнюємо кожен comparison з усіма original
#     print("\n" + "=" * 70)
#     print("КРОК 2: Пошук подібних пісень")
#     print("=" * 70)
    
#     results = []
#     correct = 0
#     total = 0
    
#     for pair in pairs:
#         comp_title = pair['comparison']
#         expected_ori = pair['original']
        
#         if comp_title not in comparison_midis:
#             continue
#         if expected_ori not in original_midis:
#             continue
            
#         total += 1
#         comp_midi = comparison_midis[comp_title]
        
#         print(f"\n[{total}] Шукаю подібність для: {comp_title[:50]}...")
#         print(f"    Очікуваний оригінал: {expected_ori[:50]}...")
        
#         # Рахуємо подібність з усіма оригіналами
#         similarities = []
#         for ori_title, ori_midi in original_midis.items():
#             sim = calculate_similarity(comp_midi, ori_midi)
#             similarities.append((ori_title, sim))
        
#         # Сортуємо за подібністю
#         similarities.sort(key=lambda x: x[1], reverse=True)
        
#         # Топ-3 результати
#         print(f"    Топ-3 подібні:")
#         for i, (title, sim) in enumerate(similarities[:3]):
#             marker = "✓" if title == expected_ori else " "
#             print(f"      {i+1}. [{marker}] {sim:.4f} - {title[:40]}...")
        
#         # Перевіряємо чи правильно знайдено
#         best_match = similarities[0][0] if similarities else None
#         is_correct = best_match == expected_ori
        
#         if is_correct:
#             correct += 1
#             print(f"    ✅ ПРАВИЛЬНО!")
#         else:
#             print(f"    ❌ Неправильно (знайдено: {best_match[:40] if best_match else 'N/A'}...)")
        
#         results.append({
#             'comparison': comp_title,
#             'expected': expected_ori,
#             'found': best_match,
#             'correct': is_correct,
#             'similarity': similarities[0][1] if similarities else 0,
#             'top3': similarities[:3],
#         })
    
#     # Підсумок
#     print("\n" + "=" * 70)
#     print("РЕЗУЛЬТАТИ")
#     print("=" * 70)
    
#     accuracy = correct / total if total > 0 else 0
#     print(f"\nТочність (Top-1): {correct}/{total} = {accuracy:.2%}")
    
#     # Top-3 accuracy
#     top3_correct = sum(1 for r in results if r['expected'] in [t[0] for t in r['top3']])
#     top3_accuracy = top3_correct / total if total > 0 else 0
#     print(f"Точність (Top-3): {top3_correct}/{total} = {top3_accuracy:.2%}")
    
#     # Деталі
#     print("\n" + "-" * 70)
#     print("Детальні результати:")
#     print("-" * 70)
    
#     for r in results:
#         status = "✅" if r['correct'] else "❌"
#         print(f"{status} {r['comparison'][:35]:35} -> {r['found'][:30] if r['found'] else 'N/A':30} ({r['similarity']:.3f})")
    
#     return results, accuracy


# # ============== RUN ==============

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='MIDI Similarity Pipeline')
#     parser.add_argument('--api', default=API_URL, help='API URL')
#     parser.add_argument('--original', default=ORIGINAL_DIR, help='Original files directory')
#     parser.add_argument('--comparison', default=COMPARISON_DIR, help='Comparison files directory')
#     parser.add_argument('--midi', default=MIDI_DIR, help='MIDI cache directory')
#     parser.add_argument('--pairs', default=PAIRS_CSV, help='Pairs CSV file')
    
#     args = parser.parse_args()
    
#     results, accuracy = run_pipeline(
#         api_url=args.api,
#         original_dir=args.original,
#         comparison_dir=args.comparison,
#         midi_dir=args.midi,
#         pairs_csv=args.pairs,
#     )

"""
MIDI Similarity Pipeline v2

Конвертує WAV файли в MIDI через API, порівнює пари пісень,
знаходить найбільш подібні та рахує точність.

Оновлення:
- Sliding window для різної довжини пісень
- Overlap coefficient замість Jaccard
- Кращі метрики

Використання:
    python pipeline.py --api http://your-server:8000
"""

import os
import csv
import requests
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict
import mido

# ============== CONFIG ==============

API_URL = "http://13.220.223.194:9000"
ORIGINAL_DIR = "original"
COMPARISON_DIR = "comparison"
MIDI_DIR = "midi_cache"
PAIRS_CSV = "song_pairs.csv"

# ============== MIDI EXTRACTION ==============

def extract_notes(midi_path: str) -> List[int]:
    """Витягує послідовність нот з MIDI файлу"""
    try:
        mid = mido.MidiFile(midi_path)
        notes = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
        return notes
    except Exception as e:
        print(f"  Помилка читання {midi_path}: {e}")
        return []


def extract_intervals(notes: List[int]) -> List[int]:
    """Конвертує ноти в інтервали"""
    if len(notes) < 2:
        return []
    return [notes[i+1] - notes[i] for i in range(len(notes)-1)]


def get_ngrams(sequence: List[int], n: int) -> List[Tuple]:
    """Створює n-грами"""
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]


# ============== SIMILARITY METRICS ==============

def cosine_similarity(counter1: Counter, counter2: Counter) -> float:
    """Косинусна подібність"""
    all_keys = set(counter1.keys()) | set(counter2.keys())
    dot_product = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in all_keys)
    norm1 = sum(v**2 for v in counter1.values()) ** 0.5
    norm2 = sum(v**2 for v in counter2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def overlap_coefficient(set1: set, set2: set) -> float:
    """Overlap coefficient - нормалізується на меншу множину"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    return intersection / min(len(set1), len(set2))


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity"""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# ============== ROBUST SIMILARITY ==============

def calculate_similarity_simple(seq1: List[int], seq2: List[int], n_values: List[int]) -> dict:
    """Проста подібність без sliding window"""
    results = {}
    
    for n in n_values:
        ng1 = get_ngrams(seq1, n)
        ng2 = get_ngrams(seq2, n)
        
        if not ng1 or not ng2:
            continue
        
        set1 = set(ng1)
        set2 = set(ng2)
        counter1 = Counter(ng1)
        counter2 = Counter(ng2)
        
        results[n] = {
            'cosine': cosine_similarity(counter1, counter2),
            'overlap': overlap_coefficient(set1, set2),
            'jaccard': jaccard_similarity(set1, set2),
        }
    
    return results


def calculate_similarity_sliding(short_seq: List[int], long_seq: List[int], 
                                  n_values: List[int], step_ratio: float = 0.25) -> dict:
    """Sliding window подібність - шукає найкращий збіг"""
    
    window_size = len(short_seq)
    step = max(1, int(window_size * step_ratio))
    
    best_results = {n: {'cosine': 0, 'overlap': 0, 'jaccard': 0} for n in n_values}
    
    # Попередньо рахуємо n-грами для короткої послідовності
    short_ngrams = {n: get_ngrams(short_seq, n) for n in n_values}
    short_sets = {n: set(ng) for n, ng in short_ngrams.items()}
    short_counters = {n: Counter(ng) for n, ng in short_ngrams.items()}
    
    for start in range(0, len(long_seq) - window_size + 1, step):
        window = long_seq[start:start + window_size]
        
        for n in n_values:
            if not short_ngrams[n]:
                continue
                
            window_ngrams = get_ngrams(window, n)
            if not window_ngrams:
                continue
            
            window_set = set(window_ngrams)
            window_counter = Counter(window_ngrams)
            
            cosine = cosine_similarity(short_counters[n], window_counter)
            overlap = overlap_coefficient(short_sets[n], window_set)
            jaccard = jaccard_similarity(short_sets[n], window_set)
            
            best_results[n]['cosine'] = max(best_results[n]['cosine'], cosine)
            best_results[n]['overlap'] = max(best_results[n]['overlap'], overlap)
            best_results[n]['jaccard'] = max(best_results[n]['jaccard'], jaccard)
    
    return best_results


def robust_similarity(midi1: str, midi2: str, 
                      n_values: List[int] = [3, 4, 5],
                      use_intervals: bool = True,
                      use_sliding: bool = True) -> dict:
    """
    Robust подібність між двома MIDI файлами.
    
    Повертає dict з різними метриками.
    """
    notes1 = extract_notes(midi1)
    notes2 = extract_notes(midi2)
    
    if not notes1 or not notes2:
        return {'score': 0, 'details': {}}
    
    # Інтервали або ноти
    if use_intervals:
        seq1 = extract_intervals(notes1)
        seq2 = extract_intervals(notes2)
    else:
        seq1 = notes1
        seq2 = notes2
    
    if not seq1 or not seq2:
        return {'score': 0, 'details': {}}
    
    # Визначаємо коротшу і довшу послідовність
    if len(seq1) <= len(seq2):
        short_seq, long_seq = seq1, seq2
    else:
        short_seq, long_seq = seq2, seq1
    
    length_ratio = len(short_seq) / len(long_seq)
    
    # Вибираємо метод в залежності від різниці довжин
    if use_sliding and length_ratio < 0.8:
        # Велика різниця - використовуємо sliding window
        results = calculate_similarity_sliding(short_seq, long_seq, n_values)
    else:
        # Схожа довжина - проста подібність
        results = calculate_similarity_simple(seq1, seq2, n_values)
    
    # Агрегуємо результати
    all_cosine = [r['cosine'] for r in results.values() if 'cosine' in r]
    all_overlap = [r['overlap'] for r in results.values() if 'overlap' in r]
    all_jaccard = [r['jaccard'] for r in results.values() if 'jaccard' in r]
    
    avg_cosine = sum(all_cosine) / len(all_cosine) if all_cosine else 0
    avg_overlap = sum(all_overlap) / len(all_overlap) if all_overlap else 0
    avg_jaccard = sum(all_jaccard) / len(all_jaccard) if all_jaccard else 0
    
    max_cosine = max(all_cosine) if all_cosine else 0
    max_overlap = max(all_overlap) if all_overlap else 0
    max_jaccard = max(all_jaccard) if all_jaccard else 0
    
    # Комбінований score (можна налаштувати ваги)
    combined_score = (max_cosine + max_overlap + avg_cosine) / 3
    
    return {
        'score': combined_score,
        'avg_cosine': avg_cosine,
        'avg_overlap': avg_overlap,
        'avg_jaccard': avg_jaccard,
        'max_cosine': max_cosine,
        'max_overlap': max_overlap,
        'max_jaccard': max_jaccard,
        'length_ratio': length_ratio,
        'notes_count': (len(notes1), len(notes2)),
        'details': results,
    }


# ============== WAV TO MIDI CONVERSION ==============

def convert_wav_to_midi(wav_path: str, output_midi: str, api_url: str) -> bool:
    """Конвертує WAV в MIDI через API"""
    
    if os.path.exists(output_midi):
        print(f"  [CACHE] {output_midi}")
        return True
    
    print(f"  Конвертую: {wav_path}")
    
    try:
        with open(wav_path, "rb") as f:
            response = requests.post(
                f"{api_url}/process",
                files={"file": (os.path.basename(wav_path), f)},
                data={
            "sound_type": "vocals",
            "extract_stem": "true",   
            "convert_midi": "true",
                },
                # data={
                #     "sound_type": "melody",
                #     "extract_stem": "true",
                #     "convert_midi": "true",
                # },
                timeout=600
            )
        
        result = response.json()
        
        if "detail" in result:
            print(f"  Помилка API: {result['detail'][:100]}...")
            return False
        
        for url in result.get("download_urls", []):
            if url.endswith(".mid"):
                r = requests.get(f"{api_url}{url}")
                os.makedirs(os.path.dirname(output_midi), exist_ok=True)
                with open(output_midi, "wb") as f:
                    f.write(r.content)
                print(f"  Збережено: {output_midi}")
                return True
        
        print("  MIDI файл не знайдено у відповіді")
        return False
        
    except Exception as e:
        print(f"  Помилка: {e}")
        return False


def find_wav_file(directory: str, title: str) -> str:
    """Шукає WAV файл за назвою"""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return None
    
    clean_title = title.lower().replace("_", " ").replace("-", " ")
    
    for f in dir_path.iterdir():
        if f.suffix.lower() in ['.wav', '.mp3', '.flac']:
            clean_name = f.stem.lower().replace("_", " ").replace("-", " ")
            if clean_title[:20] in clean_name or clean_name[:20] in clean_title:
                return str(f)
    
    title_words = clean_title.split()[:3]
    for f in dir_path.iterdir():
        if f.suffix.lower() in ['.wav', '.mp3', '.flac']:
            clean_name = f.stem.lower()
            if all(word in clean_name for word in title_words if len(word) > 2):
                return str(f)
    
    return None


# ============== MAIN PIPELINE ==============

def load_pairs(csv_path: str) -> List[Dict]:
    """Завантажує пари з CSV"""
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                'id': row.get('', row.get('id', '')),
                'original': row['ori_title'],
                'comparison': row['comp_title'],
                'relation': row['relation'],
            })
    return pairs


def run_pipeline(api_url: str, original_dir: str, comparison_dir: str, 
                 midi_dir: str, pairs_csv: str,
                 n_values: List[int] = [3, 4, 5],
                 use_sliding: bool = True):
    """Основний пайплайн"""
    
    print("=" * 70)
    print("MIDI SIMILARITY PIPELINE v2")
    print("=" * 70)
    print(f"N-grams: {n_values}")
    print(f"Sliding window: {use_sliding}")
    
    pairs = load_pairs(pairs_csv)
    print(f"\nЗавантажено {len(pairs)} пар з {pairs_csv}")
    
    os.makedirs(f"{midi_dir}/original", exist_ok=True)
    os.makedirs(f"{midi_dir}/comparison", exist_ok=True)
    
    # КРОК 1: Конвертація
    print("\n" + "=" * 70)
    print("КРОК 1: Конвертація WAV -> MIDI")
    print("=" * 70)
    
    original_midis = {}
    comparison_midis = {}
    
    print("\n[ORIGINAL FILES]")
    for pair in pairs:
        title = pair['original']
        if title in original_midis:
            continue
        wav_path = find_wav_file(original_dir, title)
        if wav_path:
            midi_path = f"{midi_dir}/original/{Path(wav_path).stem}.mid"
            if convert_wav_to_midi(wav_path, midi_path, api_url):
                original_midis[title] = midi_path
        else:
            print(f"  [NOT FOUND] {title[:50]}...")
    
    print("\n[COMPARISON FILES]")
    for pair in pairs:
        title = pair['comparison']
        if title in comparison_midis:
            continue
        wav_path = find_wav_file(comparison_dir, title)
        if wav_path:
            midi_path = f"{midi_dir}/comparison/{Path(wav_path).stem}.mid"
            if convert_wav_to_midi(wav_path, midi_path, api_url):
                comparison_midis[title] = midi_path
        else:
            print(f"  [NOT FOUND] {title[:50]}...")
    
    print(f"\nКонвертовано: {len(original_midis)} original, {len(comparison_midis)} comparison")
    
    # КРОК 2: Порівняння
    print("\n" + "=" * 70)
    print("КРОК 2: Пошук подібних пісень")
    print("=" * 70)
    
    results = []
    correct = 0
    total = 0
    
    for pair in pairs:
        comp_title = pair['comparison']
        expected_ori = pair['original']
        
        if comp_title not in comparison_midis:
            continue
        if expected_ori not in original_midis:
            continue
            
        total += 1
        comp_midi = comparison_midis[comp_title]
        
        print(f"\n[{total}] {comp_title[:50]}...")
        print(f"    Очікується: {expected_ori[:50]}...")
        
        # Порівняння з усіма оригіналами
        similarities = []
        for ori_title, ori_midi in original_midis.items():
            result = robust_similarity(
                comp_midi, ori_midi, 
                n_values=n_values, 
                use_sliding=use_sliding
            )
            similarities.append((ori_title, result['score'], result))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Топ-3
        print(f"    Топ-3:")
        for i, (title, score, details) in enumerate(similarities[:3]):
            marker = "✓" if title == expected_ori else " "
            ratio = details['length_ratio']
            print(f"      {i+1}. [{marker}] {score:.4f} (ratio={ratio:.2f}) - {title[:35]}...")
        
        best_match = similarities[0][0] if similarities else None
        is_correct = best_match == expected_ori
        
        if is_correct:
            correct += 1
            print(f"    ✅ ПРАВИЛЬНО!")
        else:
            # Знайти позицію правильної відповіді
            correct_pos = next((i for i, (t, _, _) in enumerate(similarities) if t == expected_ori), -1)
            print(f"    ❌ Неправильно (правильна на позиції {correct_pos + 1})")
        
        results.append({
            'comparison': comp_title,
            'expected': expected_ori,
            'found': best_match,
            'correct': is_correct,
            'score': similarities[0][1] if similarities else 0,
            'details': similarities[0][2] if similarities else {},
            'top3': [(t, s) for t, s, _ in similarities[:3]],
            'top5': [(t, s) for t, s, _ in similarities[:5]],
            'all_rankings': [(t, s) for t, s, _ in similarities],
        })
    
    # РЕЗУЛЬТАТИ
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ")
    print("=" * 70)
    
    accuracy_top1 = correct / total if total > 0 else 0
    print(f"\nТочність Top-1: {correct}/{total} = {accuracy_top1:.2%}")
    
    top3_correct = sum(1 for r in results if r['expected'] in [t for t, _ in r['top3']])
    accuracy_top3 = top3_correct / total if total > 0 else 0
    print(f"Точність Top-3: {top3_correct}/{total} = {accuracy_top3:.2%}")
    
    top5_correct = sum(1 for r in results if r['expected'] in [t for t, _ in r['top5']])
    accuracy_top5 = top5_correct / total if total > 0 else 0
    print(f"Точність Top-5: {top5_correct}/{total} = {accuracy_top5:.2%}")
    
    # MRR (Mean Reciprocal Rank)
    mrr_sum = 0
    for r in results:
        for i, (t, _) in enumerate(r['all_rankings']):
            if t == r['expected']:
                mrr_sum += 1 / (i + 1)
                break
    mrr = mrr_sum / total if total > 0 else 0
    print(f"MRR: {mrr:.4f}")
    
    # Деталі
    print("\n" + "-" * 70)
    print("Детальні результати:")
    print("-" * 70)
    
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"{status} {r['comparison'][:30]:30} -> {r['found'][:25] if r['found'] else 'N/A':25} ({r['score']:.3f})")
    
    # Зберегти результати
    with open('results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['comparison', 'expected', 'found', 'correct', 'score', 'top3'])
        for r in results:
            writer.writerow([
                r['comparison'], 
                r['expected'], 
                r['found'], 
                r['correct'], 
                f"{r['score']:.4f}",
                '; '.join([f"{t}: {s:.3f}" for t, s in r['top3']])
            ])
    print(f"\nРезультати збережено в results.csv")
    
    return results, {
        'top1': accuracy_top1,
        'top3': accuracy_top3,
        'top5': accuracy_top5,
        'mrr': mrr,
    }


# ============== RUN ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MIDI Similarity Pipeline v2')
    parser.add_argument('--api', default=API_URL, help='API URL')
    parser.add_argument('--original', default=ORIGINAL_DIR, help='Original files directory')
    parser.add_argument('--comparison', default=COMPARISON_DIR, help='Comparison files directory')
    parser.add_argument('--midi', default=MIDI_DIR, help='MIDI cache directory')
    parser.add_argument('--pairs', default=PAIRS_CSV, help='Pairs CSV file')
    parser.add_argument('--ngrams', default='3,4,5', help='N-gram sizes (comma-separated)')
    parser.add_argument('--no-sliding', action='store_true', help='Disable sliding window')
    
    args = parser.parse_args()
    
    n_values = [int(n) for n in args.ngrams.split(',')]
    
    results, metrics = run_pipeline(
        api_url=args.api,
        original_dir=args.original,
        comparison_dir=args.comparison,
        midi_dir=args.midi,
        pairs_csv=args.pairs,
        n_values=n_values,
        use_sliding=not args.no_sliding,
    )



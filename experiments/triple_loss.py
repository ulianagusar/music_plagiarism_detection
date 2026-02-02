"""
Пошук схожих пісень через PlagiarismDetector.
Розбито на клітинки для Jupyter — копіюй кожну клітинку окремо.
"""

# ============================================================
# КЛІТИНКА 1: Імпорти та налаштування
# ============================================================

import gc
import numpy as np
import pandas as pd
from pathlib import Path

# Шляхи — ЗМІНИ НА СВОЇ
ORIGINAL_DIR = Path("/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/midi_cache/original")
COMPARISON_DIR = Path("/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/midi_cache/comparison")
PAIRS_CSV = Path("/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/song_pairs.csv")
MODEL_PATH = "/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset/best_model.pth"

print("✓ Імпорти OK")
print(f"Original dir exists: {ORIGINAL_DIR.exists()}")
print(f"Comparison dir exists: {COMPARISON_DIR.exists()}")
print(f"CSV exists: {Path(PAIRS_CSV).exists()}")


# ============================================================
# КЛІТИНКА 2: Завантаження моделі
# ============================================================

from standalone_detector import PlagiarismDetector

detector = PlagiarismDetector(MODEL_PATH)
print("✓ Модель завантажена")


# ============================================================
# КЛІТИНКА 3: Тест одного ембедингу
# ============================================================

test_files = list(ORIGINAL_DIR.glob("*.mid")) + list(ORIGINAL_DIR.glob("*.midi"))
print(f"Знайдено файлів: {len(test_files)}")

if test_files:
    test_file = test_files[0]
    print(f"Тестую: {test_file.name}")
    
    emb = detector.get_embedding(str(test_file))
    emb = np.array(emb)
    
    print(f"Shape: {emb.shape}")
    print(f"Dtype: {emb.dtype}")
    print(f"Sample: {emb[:5]}")
    print("✓ Ембединг працює!")


# ============================================================
# КЛІТИНКА 4: Ембединг всіх original
# ============================================================

def get_all_midi(folder):
    """Знаходить всі MIDI файли."""
    folder = Path(folder)
    files = list(folder.glob("*.mid")) + list(folder.glob("*.midi"))
    return sorted(files)

def safe_embedding(detector, filepath):
    """Безпечно отримує ембединг."""
    try:
        emb = detector.get_embedding(str(filepath))
        emb = np.array(emb).flatten().astype("float32")
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        gc.collect()
        return emb
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

# Отримуємо файли
original_files = get_all_midi(ORIGINAL_DIR)
print(f"Original файлів: {len(original_files)}")

# Ембединг по одному
original_embeddings = {}
original_names = []

for i, filepath in enumerate(original_files):
    name = filepath.stem
    print(f"[{i+1}/{len(original_files)}] {name[:50]}", end=" ... ")
    
    emb = safe_embedding(detector, filepath)
    
    if emb is not None:
        original_embeddings[name] = emb
        original_names.append(name)
        print("✓")
    else:
        print("✗")

print(f"\n✓ Готово: {len(original_embeddings)} ембедингів")


# ============================================================
# КЛІТИНКА 5: Читаємо CSV з парами
# ============================================================

df = pd.read_csv(PAIRS_CSV)
print(f"Пар у CSV: {len(df)}")
print(f"Колонки: {list(df.columns)}")
print("\nПерші 3 рядки:")
print(df[["ori_title", "comp_title"]].head(3))


# ============================================================
# КЛІТИНКА 6: Функції для пошуку
# ============================================================

def normalize_name(name):
    """Нормалізує ім'я файлу."""
    name = str(name)
    for ext in [".mid", ".midi", ".wav", ".mp3"]:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
    return name

def find_file_by_name(name, files):
    """Шукає файл за частковим збігом імені."""
    name = normalize_name(name)
    
    # Точний збіг
    for f in files:
        if f.stem == name:
            return f
    
    # Частковий збіг
    for f in files:
        if name in f.stem or f.stem in name:
            return f
    
    return None

def find_nearest(query_emb, embeddings_dict):
    """Знаходить найближчий ембединг."""
    best_name = None
    best_score = -999
    
    for name, emb in embeddings_dict.items():
        score = float(np.dot(query_emb, emb))
        if score > best_score:
            best_score = score
            best_name = name
    
    return best_name, best_score

def check_match(expected, predicted):
    """Перевіряє чи імена збігаються."""
    expected = normalize_name(expected).lower()
    predicted = normalize_name(predicted).lower()
    
    # Точний збіг
    if expected == predicted:
        return True
    
    # Один містить інший
    if expected in predicted or predicted in expected:
        return True
    
    # Перші 20 символів збігаються
    if expected[:20] == predicted[:20]:
        return True
    
    return False

print("✓ Функції готові")


# ============================================================
# КЛІТИНКА 7: Головний пошук
# ============================================================

comparison_files = get_all_midi(COMPARISON_DIR)
print(f"Comparison файлів: {len(comparison_files)}")

results = []

for idx, row in df.iterrows():
    ori_title = str(row["ori_title"])
    comp_title = str(row["comp_title"])
    
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(df)}]")
    print(f"Comparison: {comp_title[:50]}")
    print(f"Expected:   {ori_title[:50]}")
    
    # Шукаємо файл comparison
    comp_file = find_file_by_name(comp_title, comparison_files)
    
    if comp_file is None:
        print("❌ Comparison файл не знайдено!")
        results.append({
            "idx": idx,
            "ori_title": ori_title,
            "comp_title": comp_title,
            "predicted": None,
            "score": None,
            "match": False,
            "status": "COMP_NOT_FOUND"
        })
        continue
    
    print(f"Файл: {comp_file.name}")
    
    # Ембединг comparison
    comp_emb = safe_embedding(detector, comp_file)
    
    if comp_emb is None:
        print("❌ Помилка ембедингу!")
        results.append({
            "idx": idx,
            "ori_title": ori_title,
            "comp_title": comp_title,
            "predicted": None,
            "score": None,
            "match": False,
            "status": "EMB_ERROR"
        })
        continue
    
    # Шукаємо найближчий original
    predicted_name, score = find_nearest(comp_emb, original_embeddings)
    
    # Перевіряємо
    match = check_match(ori_title, predicted_name)
    
    print(f"Predicted:  {predicted_name[:50]}")
    print(f"Score:      {score:.4f}")
    print(f"Result:     {'✅ CORRECT' if match else '❌ WRONG'}")
    
    results.append({
        "idx": idx,
        "ori_title": ori_title,
        "comp_title": comp_title,
        "predicted": predicted_name,
        "score": round(score, 4),
        "match": match,
        "status": "OK"
    })

print("\n" + "="*60)
print("ПОШУК ЗАВЕРШЕНО")
print("="*60)


# ============================================================
# КЛІТИНКА 8: Результати
# ============================================================

results_df = pd.DataFrame(results)

# Статистика
total = len(results_df)
valid = results_df[results_df["status"] == "OK"]
n_valid = len(valid)
n_correct = valid["match"].sum() if n_valid > 0 else 0
accuracy = valid["match"].mean() if n_valid > 0 else 0

print("="*60)
print("ФІНАЛЬНІ РЕЗУЛЬТАТИ")
print("="*60)
print(f"Всього пар:     {total}")
print(f"Перевірено:     {n_valid}")
print(f"Правильних:     {n_correct}")
print(f"Помилок:        {n_valid - n_correct}")
print(f"Точність:       {accuracy:.2%}")

print("\nСтатуси:")
print(results_df["status"].value_counts())

# Таблиця результатів
print("\n" + "="*60)
print("ДЕТАЛЬНА ТАБЛИЦЯ")
print("="*60)
print(results_df[["comp_title", "ori_title", "predicted", "score", "match"]].to_string())


# ============================================================
# КЛІТИНКА 9: Аналіз помилок
# ============================================================

errors = valid[valid["match"] == False]

if len(errors) > 0:
    print("="*60)
    print(f"ПОМИЛКИ ({len(errors)})")
    print("="*60)
    
    for _, row in errors.iterrows():
        print(f"\nComparison: {row['comp_title'][:50]}")
        print(f"  Очікувався: {row['ori_title'][:50]}")
        print(f"  Знайдено:   {row['predicted'][:50] if row['predicted'] else 'None'}")
        print(f"  Score:      {row['score']}")
else:
    print("✓ Всі відповіді правильні!")


# ============================================================
# КЛІТИНКА 10: Зберегти результати
# ============================================================

output_path = "detector_results.csv"
results_df.to_csv(output_path, index=False)
print(f"✓ Збережено: {output_path}")

# Показати файл
print("\nЗбережені дані:")
print(pd.read_csv(output_path).head())

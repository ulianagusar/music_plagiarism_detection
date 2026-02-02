"""
Оцінка точності моделі: для кожного файлу з comparison
знаходимо найподібніший в original та перевіряємо чи це правильна пара.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from standalone_detector import PlagiarismDetector

# Шляхи
BASE_DIR = Path("/Users/ulanagusar/Desktop/ML_week-2026/smp_dataset")
COMPARISON_DIR = BASE_DIR / "midi_cache" / "comparison"
ORIGINAL_DIR = BASE_DIR / "midi_cache" / "original"
CSV_PATH = BASE_DIR / "Final_dataset_pairs.csv"
MODEL_PATH = BASE_DIR / "best_model.pth"

def normalize_name(name: str) -> str:
    """Нормалізує назву для порівняння"""
    return name.lower().replace(".mid", "").replace(".midi", "").strip()

def find_matching_file(title: str, files: list) -> Path:
    """Знаходить файл за назвою (часткове співпадіння)"""
    title_norm = normalize_name(title)

    for f in files:
        if normalize_name(f.stem) in title_norm or title_norm in normalize_name(f.stem):
            return f

    # Спробуємо знайти за першими словами
    title_words = title_norm.split()[:3]
    for f in files:
        file_norm = normalize_name(f.stem)
        if all(word in file_norm for word in title_words if len(word) > 2):
            return f

    return None

def main():
    print("=" * 60)
    print("🎵 ОЦІНКА ТОЧНОСТІ МОДЕЛІ")
    print("=" * 60)

    # Завантажуємо модель
    detector = PlagiarismDetector(str(MODEL_PATH))

    # Отримуємо списки файлів
    comparison_files = list(COMPARISON_DIR.glob("*.mid"))
    original_files = list(ORIGINAL_DIR.glob("*.mid"))

    print(f"\n📁 Comparison файлів: {len(comparison_files)}")
    print(f"📁 Original файлів: {len(original_files)}")

    # Завантажуємо ground truth
    df = pd.read_csv(CSV_PATH)
    # Залишаємо унікальні пари (ori_title, comp_title)
    df_unique = df.drop_duplicates(subset=['ori_title', 'comp_title'])
    print(f"📊 Унікальних пар у CSV: {len(df_unique)}")

    # Отримуємо ембедінги для всіх файлів
    print("\n🔄 Обчислюємо ембедінги для original файлів...")
    original_embeddings = {}
    for f in original_files:
        emb = detector.get_embedding(str(f))
        if emb is not None:
            original_embeddings[f.name] = emb
    print(f"✓ Отримано {len(original_embeddings)} ембедінгів")

    print("\n🔄 Обчислюємо ембедінги для comparison файлів...")
    comparison_embeddings = {}
    for f in comparison_files:
        emb = detector.get_embedding(str(f))
        if emb is not None:
            comparison_embeddings[f.name] = emb
    print(f"✓ Отримано {len(comparison_embeddings)} ембедінгів")

    # Створюємо ground truth mapping з CSV
    print("\n🔍 Створюємо ground truth mapping...")
    ground_truth = {}  # comp_file -> ori_file

    for _, row in df_unique.iterrows():
        ori_title = str(row['ori_title'])
        comp_title = str(row['comp_title'])

        ori_file = find_matching_file(ori_title, original_files)
        comp_file = find_matching_file(comp_title, comparison_files)

        if ori_file and comp_file:
            ground_truth[comp_file.name] = ori_file.name

    print(f"✓ Знайдено {len(ground_truth)} пар у ground truth")

    # Оцінка: для кожного comparison файлу знаходимо найближчий original
    print("\n" + "=" * 60)
    print("📊 ОЦІНКА ТОЧНОСТІ")
    print("=" * 60)

    results = []
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0

    for comp_name, comp_emb in comparison_embeddings.items():
        # Обчислюємо відстані до всіх original
        distances = []
        for ori_name, ori_emb in original_embeddings.items():
            dist = detector.distance(comp_emb, ori_emb)
            sim = detector.similarity(comp_emb, ori_emb)
            distances.append({
                'original': ori_name,
                'distance': dist,
                'similarity': sim
            })

        # Сортуємо за відстанню (менше = більш схожі)
        distances.sort(key=lambda x: x['distance'])

        # Знаходимо ground truth для цього файлу
        expected_original = ground_truth.get(comp_name)

        if expected_original is None:
            # Немає ground truth для цього файлу
            continue

        total += 1
        top1 = distances[0]['original']
        top3 = [d['original'] for d in distances[:3]]
        top5 = [d['original'] for d in distances[:5]]

        is_correct_top1 = top1 == expected_original
        is_correct_top3 = expected_original in top3
        is_correct_top5 = expected_original in top5

        if is_correct_top1:
            correct_top1 += 1
        if is_correct_top3:
            correct_top3 += 1
        if is_correct_top5:
            correct_top5 += 1

        result = {
            'comparison': comp_name,
            'expected': expected_original,
            'predicted_top1': top1,
            'distance': distances[0]['distance'],
            'similarity': distances[0]['similarity'],
            'correct_top1': is_correct_top1,
            'correct_top3': is_correct_top3,
            'correct_top5': is_correct_top5
        }
        results.append(result)

        # Виводимо результат
        status = "✅" if is_correct_top1 else ("🟡" if is_correct_top3 else "❌")
        print(f"\n{status} {comp_name[:50]}")
        print(f"   Expected:  {expected_original[:50]}")
        print(f"   Predicted: {top1[:50]} (dist: {distances[0]['distance']:.4f})")
        if not is_correct_top1 and is_correct_top3:
            # Показуємо де знаходиться правильна відповідь
            for i, d in enumerate(distances[:5]):
                if d['original'] == expected_original:
                    print(f"   Correct at position {i+1}: dist={d['distance']:.4f}")

    # Підсумкова статистика
    print("\n" + "=" * 60)
    print("📈 ПІДСУМОК")
    print("=" * 60)

    if total > 0:
        acc_top1 = correct_top1 / total * 100
        acc_top3 = correct_top3 / total * 100
        acc_top5 = correct_top5 / total * 100

        print(f"\nВсього оцінено пар: {total}")
        print(f"\n🎯 Top-1 Accuracy: {correct_top1}/{total} = {acc_top1:.1f}%")
        print(f"🎯 Top-3 Accuracy: {correct_top3}/{total} = {acc_top3:.1f}%")
        print(f"🎯 Top-5 Accuracy: {correct_top5}/{total} = {acc_top5:.1f}%")
    else:
        print("❌ Не вдалося оцінити жодної пари")

    # Зберігаємо результати
    results_df = pd.DataFrame(results)
    results_path = BASE_DIR / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Результати збережено: {results_path}")

    return results_df

if __name__ == "__main__":
    main()

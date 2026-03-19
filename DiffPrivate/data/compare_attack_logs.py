import json

import os

from statistics import mean



BASELINE_PATH = "output_baseline_100/attack_log.json"

FINETUNED_PATH = "output_finetuned_100//attack_log.json"





def load_attack_log(path):

    """

    Supports:

    1. JSON array: [ {...}, {...} ]

    2. JSON lines: one JSON object per line

    """

    with open(path, "r", encoding="utf-8") as f:

        content = f.read().strip()



    if not content:

        return []



    try:

        data = json.loads(content)

        if isinstance(data, list):

            return data

        elif isinstance(data, dict):

            return [data]

    except json.JSONDecodeError:

        pass



    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(json.loads(line))

    return rows





def safe_mean(values):

    values = [v for v in values if isinstance(v, (int, float))]

    return mean(values) if values else None





def fmt(x):

    if x is None:

        return "N/A"

    if isinstance(x, float):

        return f"{x:.6f}"

    return str(x)





def summarize_run(name, rows):

    print(f"\n=== {name} SUMMARY ===")

    print(f"Images: {len(rows)}")



    id_loss_vals = [r.get("id_loss") for r in rows]

    victim_vals = [r.get("ID_victim_distance") for r in rows]

    target_vals = [r.get("ID_dist_to_target") for r in rows]



    print(f"Avg id_loss: {fmt(safe_mean(id_loss_vals))}")

    print(f"Avg ID_victim_distance: {fmt(safe_mean(victim_vals))}")

    print(f"Avg ID_dist_to_target: {fmt(safe_mean(target_vals))}")



    # Collect distance keys

    distance_keys = set()

    protected_keys = set()

    for r in rows:

        distance_keys.update(r.get("distance", {}).keys())

        protected_keys.update(r.get("protected", {}).keys())



    print("\nPer-model average distances:")

    for k in sorted(distance_keys):

        vals = [r.get("distance", {}).get(k) for r in rows]

        print(f"  {k}: {fmt(safe_mean(vals))}")



    print("\nPer-model protection rates:")

    for k in sorted(protected_keys):

        vals = [r.get("protected", {}).get(k) for r in rows]

        vals = [v for v in vals if isinstance(v, (int, float))]

        rate = (sum(vals) / len(vals)) if vals else None

        print(f"  {k}: {fmt(rate)}")





def compare_runs(baseline_rows, finetuned_rows):

    base_by_image = {r["image_name"]: r for r in baseline_rows if "image_name" in r}

    fine_by_image = {r["image_name"]: r for r in finetuned_rows if "image_name" in r}



    shared = sorted(set(base_by_image.keys()) & set(fine_by_image.keys()))

    only_base = sorted(set(base_by_image.keys()) - set(fine_by_image.keys()))

    only_fine = sorted(set(fine_by_image.keys()) - set(base_by_image.keys()))



    print("\n=== COMPARISON ===")

    print(f"Shared images: {len(shared)}")

    print(f"Only in baseline: {len(only_base)}")

    print(f"Only in finetuned: {len(only_fine)}")



    victim_deltas = []

    id_loss_deltas = []



    all_distance_keys = set()

    all_protected_keys = set()



    for img in shared:

        b = base_by_image[img]

        f = fine_by_image[img]

        all_distance_keys.update(b.get("distance", {}).keys())

        all_distance_keys.update(f.get("distance", {}).keys())

        all_protected_keys.update(b.get("protected", {}).keys())

        all_protected_keys.update(f.get("protected", {}).keys())



        b_v = b.get("ID_victim_distance")

        f_v = f.get("ID_victim_distance")

        if isinstance(b_v, (int, float)) and isinstance(f_v, (int, float)):

            victim_deltas.append(f_v - b_v)



        b_id = b.get("id_loss")

        f_id = f.get("id_loss")

        if isinstance(b_id, (int, float)) and isinstance(f_id, (int, float)):

            id_loss_deltas.append(f_id - b_id)



    print(f"\nAvg delta ID_victim_distance (finetuned - baseline): {fmt(safe_mean(victim_deltas))}")

    print(f"Avg delta id_loss (finetuned - baseline): {fmt(safe_mean(id_loss_deltas))}")



    improved_victim = sum(1 for d in victim_deltas if d > 0)

    worsened_victim = sum(1 for d in victim_deltas if d < 0)

    same_victim = sum(1 for d in victim_deltas if d == 0)



    print("\nID_victim_distance change counts:")

    print(f"  Improved: {improved_victim}")

    print(f"  Worsened: {worsened_victim}")

    print(f"  Same: {same_victim}")



    print("\nPer-model distance delta averages (finetuned - baseline):")

    for k in sorted(all_distance_keys):

        deltas = []

        for img in shared:

            b_val = base_by_image[img].get("distance", {}).get(k)

            f_val = fine_by_image[img].get("distance", {}).get(k)

            if isinstance(b_val, (int, float)) and isinstance(f_val, (int, float)):

                deltas.append(f_val - b_val)

        print(f"  {k}: {fmt(safe_mean(deltas))}")



    print("\nPer-model protection rate comparison:")

    for k in sorted(all_protected_keys):

        base_vals = [

            base_by_image[img].get("protected", {}).get(k)

            for img in shared

            if isinstance(base_by_image[img].get("protected", {}).get(k), (int, float))

        ]

        fine_vals = [

            fine_by_image[img].get("protected", {}).get(k)

            for img in shared

            if isinstance(fine_by_image[img].get("protected", {}).get(k), (int, float))

        ]



        base_rate = (sum(base_vals) / len(base_vals)) if base_vals else None

        fine_rate = (sum(fine_vals) / len(fine_vals)) if fine_vals else None

        delta = None

        if base_rate is not None and fine_rate is not None:

            delta = fine_rate - base_rate



        print(f"  {k}: baseline={fmt(base_rate)}  finetuned={fmt(fine_rate)}  delta={fmt(delta)}")



    print("\nTop 10 images with biggest improvement in ID_victim_distance:")

    image_deltas = []

    for img in shared:

        b_v = base_by_image[img].get("ID_victim_distance")

        f_v = fine_by_image[img].get("ID_victim_distance")

        if isinstance(b_v, (int, float)) and isinstance(f_v, (int, float)):

            image_deltas.append((img, f_v - b_v, b_v, f_v))



    image_deltas.sort(key=lambda x: x[1], reverse=True)

    for img, delta, b_v, f_v in image_deltas[:10]:

        print(f"  {img}: delta={delta:.6f}  baseline={b_v:.6f}  finetuned={f_v:.6f}")



    print("\nTop 10 images with biggest drop in ID_victim_distance:")

    image_deltas.sort(key=lambda x: x[1])

    for img, delta, b_v, f_v in image_deltas[:10]:

        print(f"  {img}: delta={delta:.6f}  baseline={b_v:.6f}  finetuned={f_v:.6f}")





def main():

    if not os.path.exists(BASELINE_PATH):

        raise FileNotFoundError(f"Missing baseline file: {BASELINE_PATH}")

    if not os.path.exists(FINETUNED_PATH):

        raise FileNotFoundError(f"Missing finetuned file: {FINETUNED_PATH}")



    baseline_rows = load_attack_log(BASELINE_PATH)

    finetuned_rows = load_attack_log(FINETUNED_PATH)



    summarize_run("BASELINE", baseline_rows)

    summarize_run("FINETUNED", finetuned_rows)

    compare_runs(baseline_rows, finetuned_rows)





if __name__ == "__main__":

    main()

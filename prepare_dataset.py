import os
import shutil
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else "../Dataset"
out = "dataset"

classes = ["with_mask", "without_mask", "mask_weared_incorrect"]
counts = {}

for label in classes:
    source = os.path.join(raw, label)
    destination = os.path.join(out, label)

    os.makedirs(destination, exist_ok=True)

    if not os.path.exists(source):
        print("Missing folder:", source)
        continue

    count = 0

    for filename in os.listdir(source):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            source_file = os.path.join(source, filename)
            destination_file = os.path.join(destination, filename)

            shutil.copy2(source_file, destination_file)
            count += 1

    counts[label] = count

print("Dataset preparation completed.")
print("Images per class:", counts)
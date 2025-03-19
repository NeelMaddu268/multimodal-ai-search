# utils/parse_flickr_all_captions.py

from collections import defaultdict

def load_all_captions(filepath):
    captions_dict = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image, caption = line.split(",", 1)
            captions_dict[image].append(caption.strip())

    return captions_dict

def save_all_captions(captions_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as out:
        for image, captions in captions_dict.items():
            for cap in captions:
                out.write(f"{image}|{cap}\n")

if __name__ == "__main__":
    input_file = "captions/captions.txt"
    output_file = "captions/formatted_all_captions.txt"

    captions = load_all_captions(input_file)
    save_all_captions(captions, output_file)

    print(f"Saved all 5 captions per image to: {output_file}")

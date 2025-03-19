# utils/parse_flickr_captions.py

from collections import defaultdict

def load_captions(filepath):
    captions_dict = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image, caption = line.split(",", 1)
            captions_dict[image].append(caption.strip())

    return captions_dict

def save_one_caption_per_image(captions_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as out:
        for image, captions in captions_dict.items():
            # Pick the first caption (or random, or avg length, etc.)
            first_caption = captions[0]
            out.write(f"{image}|{first_caption}\n")

if __name__ == "__main__":
    input_file = "captions/captions.txt"  # replace with your actual file path
    output_file = "captions/formatted_captions.txt"

    captions = load_captions(input_file)
    save_one_caption_per_image(captions, output_file)

    print(f"Saved simplified captions to: {output_file}")

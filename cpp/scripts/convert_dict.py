import os

# Tone mapping dictionary
TONE_MAP = {
    'ā': ('a', 1), 'á': ('a', 2), 'ǎ': ('a', 3), 'à': ('a', 4),
    'ē': ('e', 1), 'é': ('e', 2), 'ě': ('e', 3), 'è': ('e', 4),
    'ī': ('i', 1), 'í': ('i', 2), 'ǐ': ('i', 3), 'ì': ('i', 4),
    'ō': ('o', 1), 'ó': ('o', 2), 'ǒ': ('o', 3), 'ò': ('o', 4),
    'ū': ('u', 1), 'ú': ('u', 2), 'ǔ': ('u', 3), 'ù': ('u', 4),
    'ü': ('v', 0), 'ǖ': ('v', 1), 'ǘ': ('v', 2), 'ǚ': ('v', 3), 'ǜ': ('v', 4),
    'ń': ('n', 2), 'ň': ('n', 3), 'ǹ': ('n', 4), 'ḿ': ('m', 2),
    'm̀': ('m', 4)
}

def convert_tone(pinyin):
    if not pinyin: return ""
    tone = 5  # Default neutral tone
    base = ""
    
    for char in pinyin:
        if char in TONE_MAP:
            base_char, t = TONE_MAP[char]
            base += base_char
            tone = t
        else:
            base += char
            
    if tone != 5:
        return f"{base}{tone}"
    else:
        return f"{base}5"

def process_char_dict(input_file, out_file):
    print(f"Processing char dict: {input_file} -> {out_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            new_lines.append(line)
            continue
        
        # Expected format: U+XXXX: pinyin,pinyin # char
        if ':' not in line:
            new_lines.append(line)
            continue
            
        part1, part2 = line.split(':', 1) # U+XXXX, " pinyin,pinyin # char"
        
        char_comment = ""
        if '#' in part2:
            pinyins_part, char_comment = part2.split('#', 1)
            char_comment = " #" + char_comment
        else:
            pinyins_part = part2
            
        # Split by comma for polyphones
        pinyins = [p.strip() for p in pinyins_part.split(',')]
        numeric_pinyins = [convert_tone(p) for p in pinyins if p]
        
        new_pinyins_str = ",".join(numeric_pinyins)
        
        # Reconstruct line
        new_line = f"{part1}: {new_pinyins_str}{char_comment}"
        new_lines.append(new_line)

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines))
    print(f"Done. Wrote {len(new_lines)} lines.")

def process_phrase_dict(input_file, out_file):
    print(f"Processing phrase dict: {input_file} -> {out_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    phrase_entries = []

    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        word, pinyins_str = line.split(':', 1)
        word = word.strip()
        pinyins = [p.strip() for p in pinyins_str.split()]
        
        # Skip single char entries in phrase dict if any
        if len(word) == 1:
            continue

        numeric_pinyins = [convert_tone(p) for p in pinyins]
        pinyins_joined = " ".join(numeric_pinyins)

        phrase_entries.append(f"{word}: {pinyins_joined}")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(phrase_entries))
    print(f"Done. Wrote {len(phrase_entries)} phrases.")

if __name__ == "__main__":
    # Input files
    input_char_path = r"g:\work\misaki\cpp\dict2\pinyin.txt"
    input_phrase_path = r"g:\work\misaki\cpp\dict2\large_pinyin.txt"
    
    # Output directory
    output_dir = r"g:\work\misaki\cpp\dict" 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process both
    if os.path.exists(input_char_path):
        process_char_dict(input_char_path, os.path.join(output_dir, "pinyin.txt"))
    else:
        print(f"Warning: {input_char_path} not found.")
        
    if os.path.exists(input_phrase_path):
        process_phrase_dict(input_phrase_path, os.path.join(output_dir, "pinyin_phrase.txt"))
    else:
        print(f"Warning: {input_phrase_path} not found.")

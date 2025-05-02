import os
import subprocess
import re

# Ścieżki
input_video = r'I:\[ACESSE...HD-ELITE.ME] Action Point 2018 BD25 THYRSO10\BDMV\STREAM\00351.m2ts'
frags_file = r'd:\AI\PPL_DETECTION\frags.txt'
output_dir = r'd:\AI\PPL_DETECTION'

# Funkcja konwertująca czas do sekund
def time_to_seconds(t):
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s

# Wczytaj fragmenty
with open(frags_file, encoding='utf-8') as f:
    lines = f.readlines()

frag_idx = 1
for line in lines:
    match = re.match(r'(\d+:\d+:\d+) - (\d+:\d+:\d+)', line.strip())
    if match:
        start, end = match.groups()
        start_sec = time_to_seconds(start)
        end_sec = time_to_seconds(end)
        duration = end_sec - start_sec
        output_path = os.path.join(output_dir, f'fragment_{frag_idx:02d}.mp4')
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_sec),
            '-i', input_video,
            '-t', str(duration),
            '-c', 'copy',
            output_path
        ]
        print(f'Wycinam: {start} - {end} -> {output_path}')
        subprocess.run(cmd, check=True)
        frag_idx += 1

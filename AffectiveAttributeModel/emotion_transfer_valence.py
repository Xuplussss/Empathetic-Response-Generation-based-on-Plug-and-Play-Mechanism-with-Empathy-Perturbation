
emo_file = "valid_situation_emotion.txt"
valence_file = "valid_situation_valence.txt"
transfer_table_file = "emotion_transfer_valence_table.txt"

emotion = []
valence = []
with open(transfer_table_file, 'r', encoding='utf-8') as table_file:
    for line in table_file:
        line = line.strip()
        line = line.split()
        emotion.append(line[0])
        valence.append(line[1])

with open(emo_file, 'r', encoding='utf-8') as f_text:
    with open(valence_file, 'w', encoding='utf-8') as w_text:
        for line in f_text:
            line = line.strip()
            w_text.write("{}\n".format(valence[emotion.index(line)]))

FILE_PATH="test/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/squad"
LEN=1024

python calculate_f1.py \
    --input $FILE_PATH/squad_$LEN.json \
    --output $FILE_PATH/squad_$LEN_f1_score.txt

python calculate_f1.py \
    --input $FILE_PATH/squad_$LEN_no_metanet.json \
    --output $FILE_PATH/squad_$LEN_no_metanet_f1_score.txt

python calculate_f1.py \
    --input $FILE_PATH/squad_$LEN_only_question.json \
    --output $FILE_PATH/squad_$LEN_only_question_f1_score.txt
FILE_PATH="test/squad"

python calculate_f1.py \
    --input $FILE_PATH/squad.json \
    --output $FILE_PATH/squad_f1_score.txt

python calculate_f1.py \
    --input $FILE_PATH/squad_no_metanet.json \
    --output $FILE_PATH/squad_no_metanet_f1_score.txt

python calculate_f1.py \
    --input $FILE_PATH/squad_only_question.json \
    --output $FILE_PATH/squad_only_question_f1_score.txt
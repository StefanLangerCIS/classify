# Convenience script (powershell) to run some classifier in a row

# Dense parameter can be "" or "--dense"
$dense = ""

# Classifier to use. You can use any scikit learn classifier listed in the python code
# "DecisionTreeClassifier", "RandomForestClassifier", "LogisticRegression", "MLPClassifier",
# "GaussianNB", "MultinomialNB", "KNeighborsClassifier", "LinearSVC", "Perceptron"
$classifier = "LogisticRegression"

#  Additional label for the output file (labelled by default with classifier name and dense info)
$label = "default"

$DATA_DIR = "C:\ProjectData\Uni\classif_srch\data\letters\classification"
python .\run_classifier.py  $dense --classifier $classifier --training $DATA_DIR\classifier_data_train.jsonl --input $DATA_DIR\classifier_data_eval.jsonl --output $DATA_DIR\results --label author --text_label text --file_label $label
python .\run_classifier.py  $dense --classifier $classifier --training $DATA_DIR\classifier_data_train.jsonl --input $DATA_DIR\classifier_data_eval.jsonl --output $DATA_DIR\results --label lang --text_label text --file_label $label

$DATA_DIR = "C:\ProjectData\Uni\classif_srch\data\news\classification"
python .\run_classifier.py  $dense --classifier $classifier --training $DATA_DIR\classifier_data_train.jsonl --input $DATA_DIR\classifier_data_eval.jsonl --output $DATA_DIR\results --label category --text_label headline,short_description --file_label $label

$DATA_DIR = "C:\ProjectData\Uni\classif_srch\data\sentiment\classification"
python .\run_classifier.py  $dense --classifier $classifier --training $DATA_DIR\classifier_data_train.jsonl --input $DATA_DIR\classifier_data_eval.jsonl --output $DATA_DIR\results --label sentiment  --text_label text --file_label $label


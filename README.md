# CS412_Project


---

# Machine Learning Homework Grading System

## Overview of the Repository
Our project consists of several scripts and modules that collectively build a machine learning system to grade homework based on students' interactions with ChatGPT. Key components include:

- ## **HTML Parsing**: Extracting conversation data from HTML files using BeautifulSoup.
```python
data_path = "data/html/*.html"

code2convos = dict()

pbar = tqdm.tqdm(sorted(list(glob(data_path))))
for path in pbar:
    # print(Path.cwd() / path)
    file_code = os.path.basename(path).split(".")[0]
    with open(path, "r", encoding="latin1") as fh:

        # get the file id to use it as key later on
        fid = os.path.basename(path).split(".")[0]

        # read the html file
        html_page = fh.read()

        # parse the html file with bs4 so we can extract needed stuff
        soup = BeautifulSoup(html_page, "html.parser")

        # grab the conversations with the data-testid pattern
        data_test_id_pattern = re.compile(r"conversation-turn-[0-9]+")
        conversations = soup.find_all("div", attrs={"data-testid": data_test_id_pattern})

        convo_texts = []

        for i, convo in enumerate(conversations):
            convo = convo.find_all("div", attrs={"data-message-author-role":re.compile( r"[user|assistant]") })
            if len(convo) > 0:
                role = convo[0].get("data-message-author-role")
                convo_texts.append({
                        "role" : role,
                        "text" : convo[0].text
                    }
                )

        code2convos[file_code] = convo_texts
```
- ## **Data Preprocessing**: Utilizing `pandas`, `numpy`, and `nltk` for data cleaning, manipulation, and natural language processing.
- ## **Feature Engineering**: For feature engineering, we used nltk library to tokenize sentences then we trained a word2vec model.
```python
nltk.download('punkt')

# Tokenize sentences and create a flat list of sentences
sentence_load = []
test_list = list(code2prompts.values())
test_list.append(questions)
for sentences in test_list:
    for sentence in sentences:
        # Tokenize each sentence
        tokenized_sentence = nltk.word_tokenize(sentence)
        sentence_load.append(tokenized_sentence)

vector_size = 600
window = 5
min_count = 2

hw_grading_word2vec_model = Word2Vec(
    sentences=sentence_load,
    vector_size=vector_size,
    window=window,
    min_count= min_count,
    workers=10
)
hw_grading_word2vec_model
```
This code is designed to vectorize textual data (user prompts and questions) using a word2vec model. Each sentence is converted into a vector by averaging the vectors of the words it contains. 
```python
def vectorize_sentence(prompt): ## Each sentence of prompt being vectorized
  user_vector_list = list([])
  for sentence in prompt:
    tokens = sentence.split()
    word_vectors = [hw_grading_word2vec_model.wv[word] for word in tokens if word in hw_grading_word2vec_model.wv]
    if not word_vectors:
      user_vector_list.append(np.zeros(hw_grading_word2vec_model.vector_size))
      continue
    user_vector_list.append(np.mean(word_vectors, axis=0))
  return user_vector_list

code2prompts_word2vec = dict()
for code, user_prompts in code2prompts.items():
  if len(user_prompts) == 0:
      print(code+".html")
      continue
  prompts_word2Vec = pd.DataFrame(vectorize_sentence(user_prompts))
  code2prompts_word2vec[code] = prompts_word2Vec

questions_word2Vec = pd.DataFrame(vectorize_sentence(questions)) #Questions vectorized

```
After this process we vectorize user prompts, questions and map them to different dictionaries which are present in the trained word2vec model.
```python
code2questionmapping_word2vec = dict()
for code, cosine_scores in code2cosine_word2vec.items():
    code2questionmapping_word2vec[code] = code2cosine_word2vec[code].max(axis=1).tolist()


question_mapping_scores_word2vec = pd.DataFrame(code2questionmapping_word2vec).T
question_mapping_scores_word2vec.reset_index(inplace=True)
question_mapping_scores_word2vec.rename(columns={i: f"Q_{i}" for i in range(len(questions))}, inplace=True)
question_mapping_scores_word2vec.rename(columns={"index" : "code"}, inplace=True)

question_mapping_scores_word2vec #Similarity matrix between questions and prompts of user
```
Then, we look at the cosine similarity between the user prompts and questions using the dictionaries.

```python
code2features = defaultdict(lambda : defaultdict(int))

keywords2search = ["error", "no", "thank", "next", "Entropy","how"]
keywords2search = [k.lower() for k in keywords2search]

for code, convs in code2convos.items():
  if len(convs) == 0:
      print(code)
      continue
  for c in convs:
    text = c["text"].lower()
    if c["role"] == "user":
        # User Prompts

        # count the user prompts
        code2features[code]["#user_prompts"] += 1

        # count the keywords
        for kw in keywords2search:
            code2features[code][f"#{kw}"] +=  len(re.findall(rf"\b{kw}\b", text))

        code2features[code]["prompt_avg_chars"] += len(text)

        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        blob = TextBlob(text)
        code2features[code]["unique_avg_chars"] += len(words)
        code2features[code]["sentiment_point"] += blob.sentiment.polarity

    else:
        # ChatGPT Responses
        code2features[code]["response_avg_chars"] += len(text)
        code2features[code]["response_unique_avg_chars"] += len(words)

    code2features[code]["prompt_avg_chars"] /= code2features[code]["#user_prompts"]
    code2features[code]["response_avg_chars"] /= code2features[code]["#user_prompts"]
    code2features[code]["unique_avg_chars"] /= code2features[code]["#user_prompts"]
    code2features[code]["sentiment_point"] /= code2features[code]["#user_prompts"]
    code2features[code]["response_unique_avg_chars"] /= code2features[code]["#user_prompts"]
```
This script is a text analysis tool, which we used for extracting features like keyword frequency, character count, word count, and sentiment from conversations. These features are calculated separately for user prompts and ChatGPT responses.

In order to compare the scores of the students with the other features of the word2vec model, we extracted the code and grade from the scores table, later which we merged the table with our question mapping table that was made from the word2vec model.

```python
# reading the scores
scores = pd.read_csv("data/scores.csv", sep=",")
scores["code"] = scores["code"].apply(lambda x: x.strip())

# selecting the columns we need and we care
scores = scores[["code", "grade"]]

# show examples
scores.head()
```
This part is where we extracted the scores and code
```python
df_word2vec.reset_index(inplace=True, drop=False)
df_word2vec.rename(columns={"index": "code"}, inplace=True)

df_word2vec.head()

df_word2vec = pd.merge(df_word2vec, question_mapping_scores_word2vec, on="code", how="left")
df_word2vec.head()

temp_df_word2vec = pd.merge(df_word2vec, scores, on='code', how="left")
temp_df_word2vec.dropna(inplace=True)
temp_df_word2vec.drop_duplicates("code",inplace=True, keep="first")

temp_df_word2vec.head()
```
And these are the parts were we merged them with the question mapping table

- ## **Machine Learning Models**: Several models like Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, and CatBoost Regressor are developed and evaluated.
```python
#Initial word2vec Decision Tree Regressor Model
model_word2vec_init = DecisionTreeRegressor(criterion='squared_error', random_state=42)
model_word2vec_init.fit(X_train_word2vec, y_train_word2vec)
# word2vec Decision Tree Regressor Model evaluation
y_pred_word2vec_init = model_word2vec_init.predict(X_test_word2vec)

mae_word2vec_init = mean_absolute_error(y_test_word2vec, y_pred_word2vec_init)
mse_word2vec_init = mean_squared_error(y_test_word2vec, y_pred_word2vec_init)
rmse_word2vec_init = np.sqrt(mse_word2vec_init)
r2_word2vec_init = r2_score(y_test_word2vec, y_pred_word2vec_init)

print(f"Mean Absolute Error (MAE): {mae_word2vec_init}")
print(f"Mean Squared Error (MSE): {mse_word2vec_init}")
print(f"Root Mean Squared Error (RMSE): {rmse_word2vec_init}")
print(f"R-squared: {r2_word2vec_init}")
```
This code is for the initial training of the Decision Tree Regression model, in which we make predictions according to the model.
```python
#Cross validation check for all min_samples_split values
min_samples_split_arr = np.arange(2, 100, 3)

train_error_arr_min_samples_split = []
val_error_arr_min_samples_split = []
for min_samples_split in min_samples_split_arr:

  # Conducting cross validation
  skf = KFold(n_splits=5)

  # Arrays to save errors for each fold split
  fold_train_error_arr = []
  fold_val_error_arr = []
  for i, (train_idx, val_idx) in enumerate(skf.split(X_train_word2vec, y_train_word2vec)):
    xt = X_train_word2vec[train_idx]
    yt = y_train_word2vec[train_idx]

    xv = X_train_word2vec[val_idx]
    yv = y_train_word2vec[val_idx]

    model = DecisionTreeRegressor(
        criterion='squared_error',
        random_state=42,
        min_samples_split=min_samples_split)

    # Fitting the model to be cross validated
    model.fit(xt, yt)

    # Getting predictions
    y_pred_train = model.predict(xt)
    y_pred_val = model.predict(xv)

    # Computing error

    # Train
    train_error = mean_squared_error(y_pred_train, yt)
    fold_train_error_arr.append(train_error)

    # Validation
    valid_error = mean_squared_error(y_pred_val, yv)
    fold_val_error_arr.append(valid_error)

  # After running all splits, we compute the avgof errors in
  # the cross-validation run
  train_score_mean = np.mean(fold_train_error_arr)
  val_score_mean = np.mean(fold_val_error_arr)


  train_error_arr_min_samples_split.append(train_score_mean)
  val_error_arr_min_samples_split.append(val_score_mean)
```
Then, we do a hyperparameter tuning via cross-validation check for all minimum sample split values to get the validation error. This code is essential for optimizing model and ensuring their generalizability to unseen data.
- ## **Evaluation and Tuning**: Model evaluation using metrics like Mean Squared Error (MSE) and R-squared, and hyperparameter tuning using GridSearchCV.
```python
#Hyper parameter search space
param_grid = {
    'max_depth': [i for i in range(6,15)],
    'min_samples_split': [j for j in range(60,75)]
}

estimator = DecisionTreeRegressor(criterion='squared_error', random_state=42)
scoring='neg_mean_squared_error'
cv = 5

grid_search_decision_tree_tune = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    scoring=scoring,
    cv=cv
)
grid_search_decision_tree_tune.fit(X_train_word2vec, y_train_word2vec)
```
```python
#Use parameters that were the best from previous part
model_word2vec_tuned = DecisionTreeRegressor(
    criterion='squared_error',
    random_state=42,
    max_depth=6,
    min_samples_split=60
)

model_word2vec_tuned.fit(X_train_word2vec, y_train_word2vec)
y_pred_word2vec_tuned = model_word2vec_tuned.predict(X_test_word2vec)

mae_word2vec_tuned = mean_absolute_error(y_test_word2vec, y_pred_word2vec_tuned)
mse_word2vec_tuned = mean_squared_error(y_test_word2vec, y_pred_word2vec_tuned)
rmse_word2vec_tuned = np.sqrt(mse_word2vec_tuned)
r2_word2vec_tuned = r2_score(y_test_word2vec, y_pred_word2vec_tuned)

print(f"Mean Absolute Error (MAE): {mae_word2vec_tuned}")
print(f"Mean Squared Error (MSE): {mse_word2vec_tuned}")
print(f"Root Mean Squared Error (RMSE): {rmse_word2vec_tuned}")
print(f"R-squared: {r2_word2vec_tuned}")
```

## Methodology
Our project adopts a structured approach to grade homework. It involves:

1. Parsing HTML files to extract conversation texts.
2. Pre-processing the data.
3. Performing prompt matching using and Word2Vec methods.
4. Engineering features like the number of user prompts, average prompt length, etc.
5. Splitting data into training and testing sets.
6. Developing and tuning various machine learning models to predict grades.
7. Evaluating model performance through metrics like MSE and R-squared.

## Results
The project's effectiveness is evaluated by comparing the predicted grades against actual grades. Key observations include:

**Model Comparison**
- In our inital word2vec Decision Tree Regression Model evaluation:
- Mean Absolute Error (MAE): 6.12
- Mean Squared Error (MSE): 121.16
- Root Mean Squared Error (RMSE): 11.00727032465361
- R-squared: -0.07922640595160058
- After doing hyperparameter tuning with cross-validation for minimum sample split value
- ![image](https://github.com/AtaErnam/CS412_Project/assets/67603284/87c50d63-f370-40ba-86d0-ced466549341)
- After doing hyperparameter tuning with cross-validation for maximum depth of the tree
- ![image](https://github.com/AtaErnam/CS412_Project/assets/67603284/2ffeac78-ae34-408d-81ad-93385dbc265d)
- **Feature Impact**: Analysis of how different features influence the model's predictions.
- **Model Tuning and Evaluation**: Insights from hyperparameter tuning and their impact on model performance.
  
- **Model Train with Processed Data**
- Decision Tree Regressor with pre-processed data:
- Mean Squared Error Train (MSE Train): 3.43
- Mean Squared Error Test (MSE Test): 120.9
- After doing Decision Tree Regressor with processed data:
  
-![WhatsApp Image 2024-01-19 at 17 37 12](https://github.com/AtaErnam/CS412_Project/assets/67603284/314ac454-85d1-4402-a716-81051670a27b)

(Supporting figures and tables will be included upon availability.)

## Team Contributions
- **[Eren Yiğit Yaşar]**: Implemented feature engineering and Word2Vec model training, also added Decision Tree Regression models.
- **[Ata Ernam]**: Helped with developing and tunining the Decision Tree and Random Forest models.
- **[Melike Soytürk]**: Helped with the pre-processing and worked on the Decision Tree Regressor.
- 
- 

---

